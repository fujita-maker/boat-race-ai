# WDJ Boat Race AI Web v9 完成版

## V9の主要改善
- 見送りでも参考買い目6〜8点を必ず表示
- 買い目を判定の直下・画面上部へ表示
- Streamlitのsession_stateで予想結果を保持
- 画面が再描画されても買い目が消えない
- DB保存失敗時も予想表示を継続
- 外部サイト取得失敗時は枠順ベースの8点を自動生成
- pandas.read_html / lxmlを不使用
- Renderのstatus 139・502対策を継続

## GitHubへの入れ方
ZIPを解凍し、以下の5ファイルをリポジトリ直下へ上書きしてください。

- app.py
- Dockerfile
- README.md
- render.yaml
- requirements.txt

その後、Renderで次を実行します。

Manual Deploy → Clear build cache & deploy
