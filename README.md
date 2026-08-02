# WDJ Boat Race AI V30 Playwright

今回アップロードされた app.py を基準に修正したV30完成版です。

## 修正内容

- APP_NAMEと画面タイトルをV30へ統一
- SQLiteファイル名をboat_race_v30.dbへ変更
- Streamlit状態キーをv30へ統一
- BOAT RACE公式はPlaywright Chromiumで取得
- ポセイドン・梅吉AIはrequestsで取得
- 情報源の状態を解析件数で表示
- 取得エラーの詳細表示
- オッズ未公開時は期待値や払戻額を推測しない

## GitHubへ上書きするファイル

- app.py
- Dockerfile
- requirements.txt
- render.yaml
- README.md

アップロード後、Renderで
`Manual Deploy → Clear build cache & deploy`
を実行してください。
