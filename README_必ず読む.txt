★このフォルダの app.py が最新・完全版です（約31KB）。
　含まれる機能: /api/race, /api/train, /api/backtest, 学習モデル(L_BETA)

【重要】ダウンロードフォルダに古い app.py が複数あるはずです。
　古い版は約19KBで /api/backtest が入っていません。
　必ず「このフォルダの app.py（31KB）」をGitHubに上げてください。

手順:
 1. GitHub の boat-race-ai で app.py を、このフォルダの app.py で上書き
 2. Render → Manual Deploy → Deploy latest commit
 3. 確認: https://boat-race-ai-cxwk.onrender.com/api/backtest?days=6&step=3
    → JSON（回収率など）が返れば成功。404なら古いまま。
