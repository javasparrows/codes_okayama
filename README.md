# end2end_calm3.py 実行環境セットアップ

## 前提条件
- Python 3.11 以上
- uvパッケージマネージャー（要インストール）
- Fish shell

## セットアップ手順

### 1. uvプロジェクトの初期化と依存関係のインストール
```fish
# プロジェクトディレクトリに移動
cd /Users/yukik/Work/Tohoku/tompei-project/paper/codes_okayama

# uvプロジェクトを初期化（すでに実行済み）
uv init --python 3.11

# 必要な依存関係をインストール
uv add mlx-lm pandas
```

### 2. 仮想環境の有効化
```fish
# uvで作成された仮想環境を有効化
source .venv/bin/activate.fish

# または、uvを使って直接実行
uv run python end2end_calm3.py --help
```

### 3. スクリプトの実行
```fish
# ヘルプを表示
uv run python end2end_calm3.py --help

# 実際の実行例（疾患と処置のラベルを指定）
uv run python end2end_calm3.py --label1 心不全 --label2 ECMO

# バッチ実行（複数の疾患・処置の組み合わせを一括実行）
sh run_end2end.sh

# 実行には以下のファイル/ディレクトリが必要:
# - okayama_data/（txtファイルを含むディレクトリ）
# - ../../convert2mlx/calm3-22b-chat-mlx-16bit（モデルファイル）
```

## 必要なファイル構成
```
.
├── end2end_calm3.py      # メインスクリプト
├── utils.py              # ユーティリティ関数
├── okayama_data/         # データディレクトリ
│   ├── okayama_001_*.txt # 入力データ（txtファイル群）
│   ├── okayama_002_*.txt
│   └── ...
├── run_end2end.sh        # バッチ実行スクリプト
├── .venv/                # 仮想環境
├── pyproject.toml        # uvプロジェクト設定
└── README.md             # このファイル
```

## 主な変更点
- CSVファイル（label_data/40_result_df.csv）の代わりに、okayama_data内のtxtファイルを昇順で読み込み
- 各txtファイルの内容を医療文書として処理
- 処理結果をCSVファイルとして出力（ファイル名、カルテNo等を含む）

## 注意事項
- MLXライブラリはApple Siliconでの実行に最適化されています
- モデルファイル（calm3-22b-chat-mlx-16bit）は別途準備が必要です
- okayama_data内のtxtファイルは機密性が高いため、適切に管理してください