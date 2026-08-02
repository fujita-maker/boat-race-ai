# WDJ Boat Race AI Web v8

## V8の改善点
- Renderのstatus 139対策
- pandas.read_html / lxmlによるHTMLテーブル解析を完全停止
- 外部ページの取得サイズを最大1.5MBに制限
- 接続・読込タイムアウトを短縮
- 公式、梅吉AI、ポセイドンの一部が失敗してもアプリを落とさない
- 購入見送りでも6〜8点の参考買い目を必ず表示
- WDJ115-v2条件に適合した場合だけ5,000円で仮想購入
- 買い目、オッズ、購入額、的中時払戻、純利益を表示

## GitHubへの入れ方
このZIPを展開し、次の5ファイルをリポジトリ直下へ上書きしてください。

- app.py
- Dockerfile
- README.md
- render.yaml
- requirements.txt

アップロード後、Renderで以下を実行してください。

Manual Deploy → Clear build cache & deploy
