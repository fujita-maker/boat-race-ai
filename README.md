# WDJ Boat Race AI V20 Stable

収集・学習・予想・画面を分離したV20の安定版です。

## 安定化した点

- `__pycache__` と `.pyc` を除去
- Python 3.11に固定
- `PYTHONPATH=/app` を明示
- `data/` と `models/` を起動前に自動作成
- GitHubへ空フォルダを保持する `.gitkeep` を追加
- 一度の過去収集を最大200ページに制限
- Streamlit再描画による予想履歴の重複保存を抑止
- Render用ヘルスチェックを追加

## GitHubへ入れるもの

ZIPを解凍し、中身をフォルダ構成のまま全部アップロードしてください。

`__pycache__` や `.pyc` は入っていません。

## Render

アップロード後：

`Manual Deploy → Clear build cache & deploy`

## 重要

外部3サイトの停止、アクセス制限、HTML変更により、データ取得が一部失敗することはあります。
その場合もアプリ全体が落ちにくい構成ですが、取得情報の完全性は外部サイトの状態に依存します。

SQLiteを再デプロイ後も長期保存するには、Render Persistent Diskまたは外部DBが必要です。
