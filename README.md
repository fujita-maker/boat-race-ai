# WDJ Boat Race AI V22 Stable

Segmentation fault対策版です。

## 主な変更
- pandasを削除
- BeautifulSoupを削除
- HTML処理を標準ライブラリ中心に変更
- requirementsをStreamlitとrequestsだけに削減
- 過去収集を1回100ページ以下に制限
- 単一app.py構成を維持

GitHubへ以下の5ファイルを上書きしてください。

- app.py
- Dockerfile
- requirements.txt
- render.yaml
- README.md
