import os
import re
import argparse
import pandas as pd
import time
import logging
from datetime import datetime

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from utils import extract_json_from_response

# ANSIカラー定義
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"

def load_model_and_tokenizer(model_path: str):
    """
    MLX_LMモデルとトークナイザをロードして返す。
    """
    model, tokenizer = load(model_path)
    return model, tokenizer


def build_first_prompt() -> list:
    """
    1つ目のプロンプトを構築し、chat形式で返す。
    """
    user_question = f"{LABEL['疾患']}の患者さんの中で、{LABEL['処置']}が使われた患者は何人？"

    prompt_template = f"""以下はまず出力例を示しています。
心不全の患者の中で、肺水腫の患者は何人？"という質問に答えるために最低限必要な医学的な構造化項目を書いてください。json形式で出力してください。
回答：
[
  {{
    "fieldName": "心不全診断",
    "fieldType": "ブール値",
    "description": "患者が心不全と診断されているかどうか (true または false)",
  }},
  {{
    "fieldName": "肺水腫診断",
    "fieldType": "ブール値",
    "description": "患者が肺水腫と診断されているかどうか (true または false)"
  }}
]
出力例はここまでです。続いて、{user_question}という質問に答えるために最低限必要な医学的な構造化項目を書いてください。jsonとしてparseできるようにjson形式で出力してください。回答のjsonのみを1回だけ出力してください。
回答："""

    formatted_prompt = (
        "あなたは医療に関する質問に答えるAIアシスタントです。以下の質問文を構造化してください。\n"
        + prompt_template
    )
    return formatted_prompt


def run_inference(model, tokenizer, chat: list, temp=0.0, top_p=0.95) -> str:
    """
    引数のchatをもとにテキスト生成を行い、応答文字列を返す。
    """
    # chat_str = json.dumps(chat, ensure_ascii=False, indent=2)
    sampler = make_sampler(temp=temp, top_p=top_p)
    # print(f"chat_str: \n----------\n🍎{chat_str}🍎\n----------")
    response = generate(model, tokenizer, prompt=chat, verbose=False, sampler=sampler)
    return response


def extract_text_before_endoftext(text: str) -> str:
    """
    文字列から <|endoftext|> 以前の文字列を抽出して返す
    """
    pattern = r"(.*?)<\|endoftext\|>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text


def save_response_to_file(response: str, output_file: str, model_path: str):
    """
    応答文字列をファイルに書き込む
    """
    with open(os.path.join(LOG_DIR, output_file), "w", encoding="utf-8") as f:
        f.write(f"{model_path}\n\n{response}")


def build_second_prompt(structured_question: str, document_text: str) -> list:
    """
    structured_question（1つ目の推論で得られたjson的な出力）と
    ドキュメントのテキストを組み合わせて2つ目のプロンプトを返す。
    """
    prompt_template = (
        "json_for_structuring:\n"
        f"{structured_question}\n\n"
        "上記のjsonに従って、以下のデータを構造化してください。\n"
        f"ehr_data:\n{document_text}\n\n"
        "== json出力の例 ==\n"
        # 例を配列ではなく「1つのJSONオブジェクト」に変更
        "```json\n"
        "{\n"
        '  "心停止診断": {\n'
        '    "value": false,\n'
        '    "reason": "理由を書く"\n'
        "  },\n"
        '  "ECMO使用": {\n'
        '    "value": false,\n'
        '    "reason": "理由を書く"\n'
        "  }\n"
        "}\n"
        "```"
        "\n"
        "== json出力の例 ==\n\n"
        "json_for_structuringのそれぞれについて、ehr_dataに該当するかどうかをtrueかfalseで出力してください。\n"
        "また、その理由も書いてください。\n"
        "回答は上の例のように、トップレベルが1つのJSONオブジェクトとなるように出力してください。\n"
        "回答:"
    )

    formatted_prompt = (
        "あなたは医療に関する質問に答えるAIアシスタントです。以下の質問文を構造化してください。\n"
        + prompt_template
    )
    return formatted_prompt


def make_document_text(df_one_document: pd.DataFrame) -> str:
    """
    df_one_documentから各項目をテキスト整形して返す。
    """
    return f"""# 主訴
{df_one_document['主訴']}

# 入院までの経過
{df_one_document['入院までの経過']}

# 入院中経過
{df_one_document['入院中経過']}

# 退院時所見
{df_one_document['退院時所見']}"""


def check_all_true(bool_list):
    """
    ブール値リストを受け取り、全てTrueならTrue、1つでもFalseがあればFalseを返す。
    """
    return all(bool_list)


def main():
    """
    メイン関数。checkpoint機能を全て削除し、毎回 result_df を保存するように修正。
    """
    total_start_time = time.time()

    logger.info("===== START MAIN =====")
    logger.info(f"Using model: {MODEL_PATH}")
    print(f"{COLOR_BLUE}--- Using model: {MODEL_PATH} ---{COLOR_RESET}")
    print()

    # ---------------------------------------------------------------------
    # 1) モデルとトークナイザをロード
    # ---------------------------------------------------------------------
    step1_start = time.time()
    text_1_1 = "[STEP1] Loading model and tokenizer ..."
    logger.info(text_1_1)
    print(f"{COLOR_BLUE}--- {text_1_1} ---{COLOR_RESET}")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    step1_end = time.time()
    text_1_2 = f"[STEP1] Finished in {step1_end - step1_start:.3f} seconds\n"
    logger.info(text_1_2)
    print(f"{COLOR_BLUE}--- {text_1_2} ---{COLOR_RESET}")

    # ---------------------------------------------------------------------
    # 2) 最初のプロンプトを構築し、モデルに問い合わせ
    # ---------------------------------------------------------------------
    step2_start = time.time()
    text_2_1 = "[STEP2] Building first prompt and running inference ..."
    logger.info(text_2_1)
    print(f"{COLOR_BLUE}--- {text_2_1} ---{COLOR_RESET}")
    chat_first = build_first_prompt()
    response1 = run_inference(model, tokenizer, chat_first, temp=0.0, top_p=0.95)
    # print(f"response1: \n----------\n🍎{response1}🍎\n----------")
    response1 = extract_text_before_endoftext(response1)
    text_2_2 = f"[STEP2] First inference result:\n{response1}"
    logger.info(text_2_2)
    print(f"{COLOR_BLUE}--- {text_2_2} ---{COLOR_RESET}")
    step2_end = time.time()
    text_2_3 = f"[STEP2] Finished in {step2_end - step2_start:.3f} seconds\n"
    logger.info(text_2_3)
    print(f"{COLOR_BLUE}--- {text_2_3} ---{COLOR_RESET}")

    # ---------------------------------------------------------------------
    # 3) 応答をファイルに保存 (任意)
    # ---------------------------------------------------------------------
    step3_start = time.time()
    text_3_1 = "[STEP3] Saving first inference result to file ..."
    logger.info(text_3_1)
    print(f"{COLOR_BLUE}--- {text_3_1} ---{COLOR_RESET}")
    save_response_to_file(response1, "response_v01.txt", MODEL_PATH)
    step3_end = time.time()
    text_3_2 = f"[STEP3] Finished in {step3_end - step3_start:.3f} seconds\n"
    logger.info(text_3_2)
    print(f"{COLOR_BLUE}--- {text_3_2} ---{COLOR_RESET}")

    # ---------------------------------------------------------------------
    # 4) CSVの読み込み -> ループで2つ目のプロンプトを生成・推論
    # ---------------------------------------------------------------------
    step4_start = time.time()
    logger.info("[STEP4] Reading CSV and looping over rows ...")

    df = pd.read_csv("40_result_df.csv")
    # df = df[(df['カルテNo'] >= 756) & (df['カルテNo'] <= 758)]
    # df = df[df['カルテNo'] >= 710]
    # df = df[df[f"{LABEL['疾患']}+{LABEL['処置']}"] == 1]  # 正解データが1のもののみを使う
    # df = df.iloc[18:]
    print()
    text_4_1 = f"{COLOR_YELLOW}データ数:{COLOR_RESET} {len(df)}"
    print(text_4_1)
    logger.info(text_4_1)
    text_4_2 = f"{COLOR_YELLOW}患者数:{COLOR_RESET} {df['患者No'].nunique()}"
    print(text_4_2)
    logger.info(text_4_2)
    print()

    result = []
    make_bool_flase_list = []

    elapsed_time_list = []
    for idx in range(len(df)):
        iter_start_time = time.time()

        calte_no = df.iloc[idx]["カルテNo"]
        df_one_document = df.iloc[idx]

        label = df_one_document.get(f"{LABEL['疾患']}+{LABEL['処置']}", False)
        str_one_document = make_document_text(df_one_document)

        chat_second = build_second_prompt(structured_question=response1, document_text=str_one_document)
        response2_raw = run_inference(model, tokenizer, chat_second, temp=0.0, top_p=0.95)
        response2 = extract_text_before_endoftext(response2_raw)

        parsed_json = extract_json_from_response(response2)
        json_parse_success = ("error" not in parsed_json)

        # chat_second_str = json.dumps(chat_second, ensure_ascii=False, indent=2)
        # print(f"chat_second_str: \n----------\n🍎{chat_second_str}🍎\n----------")
        # print(f"response2_raw: \n----------\n🍎{response2_raw}🍎\n----------")
        # print(f"response2: \n----------\n🍎{response2}🍎\n----------")
        # print(f"parsed_json: \n----------\n🍎{parsed_json}🍎\n----------")
        # print(f"json_parse_success: \n----------\n🍎{json_parse_success}🍎\n----------")
        # exit()

        try:
            bool_list = []
            for key in parsed_json.keys():
                val = parsed_json[key]
                if isinstance(val, dict) and ("value" in val):
                    bool_list.append(val["value"])
                elif isinstance(val, bool):
                    bool_list.append(val)
                else:
                    bool_list.append(False)

            result_all_true = check_all_true(bool_list)
        except Exception:
            make_bool_flase_list.append([idx, df_one_document['患者No']])
            print(f"🍎{parsed_json}🍎")
            continue

        print(f"{COLOR_BLUE}--- [Iteration {idx}/{len(df)}] カルテNo: {calte_no} processed ---{COLOR_RESET}")
        parse_success_msg = f"{COLOR_GREEN}SUCCESS{COLOR_RESET}" if json_parse_success else f"{COLOR_RED}FAILED{COLOR_RESET}"
        print(f"{COLOR_YELLOW}json_parse_success:{COLOR_RESET} {parse_success_msg}")
        result_str = "正解" if result_all_true == label else "不正解"
        output_str = f"{COLOR_YELLOW}🍎check_all_true の結果:{COLOR_RESET} {bool_list} -> {result_all_true} | {COLOR_YELLOW}正解データ:{COLOR_RESET} {bool(label)}"
        output_str += f" | {COLOR_YELLOW}結果:{COLOR_RESET} {COLOR_RED if result_all_true != label else COLOR_GREEN}{result_str}{COLOR_RESET}"
        print(output_str)

        logger.info(f"[Iteration {idx+1}/{len(df)}] response2: 🍎{response2}🍎")
        logger.info(f"[Iteration {idx+1}/{len(df)}] parsed_json: 🍎{parsed_json}🍎")
        logger.info(f"[Iteration {idx+1}/{len(df)}] JSON parse success => 🍎{json_parse_success}🍎")
        logger.info(f"[Iteration {idx+1}/{len(df)}] check_all_true => 🍎{output_str}🍎")

        iter_end_time = time.time()
        elapsed_time = iter_end_time - iter_start_time
        elapsed_time_list.append(elapsed_time)
        logger.info(f"[Iteration {idx+1}/{len(df)}] took {elapsed_time:.2f} sec, average {sum(elapsed_time_list)/len(elapsed_time_list):.2f} sec\n")
        print(f"{COLOR_BLUE}Took {elapsed_time:.2f} sec, average {sum(elapsed_time_list)/len(elapsed_time_list):.2f} sec{COLOR_RESET}")

        # 結果を格納
        result.append((
            calte_no,
            response2,
            parsed_json,
            bool_list[0] if len(bool_list) > 0 else False,
            bool_list[1] if len(bool_list) > 1 else False,
            result_all_true,
            label,
            (result_all_true == label),
            json_parse_success
        ))

        # 毎回result_dfを保存
        result_df = pd.DataFrame(
            result,
            columns=[
                "カルテNo",
                "response2",
                "parsed_json",
                "bool_list_0",
                "bool_list_1",
                "result_all_true",
                "label",
                "result_all_true_eq_label",
                "json_parse_success",
            ],
        )
        result_df.to_csv(f"{LOG_DIR}/result_df.csv", index=False)

        # break  # <- 必要に応じてループを途中で切りたい場合に使用

    step4_end = time.time()
    logger.info(f"[STEP4] All loop done in {step4_end - step4_start:.3f} seconds\n")

    total_end_time = time.time()
    logger.info(f"=== TOTAL processing time: {total_end_time - total_start_time:.3f} seconds ===")
    logger.info("===== END MAIN =====\n")

    make_bool_flase_df = pd.DataFrame(make_bool_flase_list, columns=["idx", "患者No"])
    make_bool_flase_df.to_csv(f"{LOG_DIR}/make_bool_flase_df.csv", index=False)

    print(f"{COLOR_GREEN}=== 全体処理時間: {(total_end_time - total_start_time)/60:.2f} 分 ==={COLOR_RESET}")


if __name__ == "__main__":
    global LABEL
    global LOG_DIR
    global LOG_FILENAME
    global MODEL_PATH

    parser = argparse.ArgumentParser(description="ラベルを指定して処理を実行するスクリプト")
    parser.add_argument('--label1', type=str, required=True, help="疾患のラベル")
    parser.add_argument('--label2', type=str, required=True, help="処置のラベル")
    args = parser.parse_args()
    LABEL = {"疾患": args.label1, "処置": args.label2}
    print(f"{COLOR_RED}LABEL: {LABEL}{COLOR_RESET}")

    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    LOG_DIR = f"logs/log_{datetime_str}_Preferred-Qwen-72B-8bit_{LABEL['疾患']}+{LABEL['処置']}"
    os.makedirs(LOG_DIR, exist_ok=True)

    LOG_FILENAME = datetime.now().strftime(f"{LOG_DIR}/log.txt")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILENAME, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    MODEL_PATH = "../../convert2mlx/Preferred-MedLLM-Qwen-72B-mlx-8bit"
    main()