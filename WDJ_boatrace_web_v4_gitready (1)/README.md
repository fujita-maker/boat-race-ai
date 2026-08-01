# WDJ Boat Race AI Web v4

## 表示仕様
- 毎レース必ず参考予想を表示
- 最終判断は「買い・少額・見送り」
- 自信度は★1〜5
- 公式・梅吉AI・ポセイドンの取得状況を反映
- データ不足時も参考予想は表示
- 購入条件を満たした場合だけ資金配分
- 実購入と検証予測を分離
- 締切前に予想を固定
- 場別・季節別・風向別の成績を集計

## GitHubの正しい配置
リポジトリ直下に次の5ファイルを置きます。

- app.py
- Dockerfile
- render.yaml
- requirements.txt
- README.md

フォルダごと置かないでください。

## Render
RuntimeはDockerを使用します。
GitHubのmainブランチへpushすると自動デプロイされます。
