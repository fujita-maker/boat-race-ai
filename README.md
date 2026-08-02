# WDJ Boat Race AI V30 Playwright

BOAT RACE公式をChromiumで表示してから本文を取得する版です。

## 取得方式

- BOAT RACE公式：Playwright Chromium
- ポセイドン：requests
- 梅吉AI：requests

## 安定化

- 画像・動画・フォントを読み込まない
- Chromiumは予想ボタンを押した時だけ起動
- 公式3ページを取得したら必ず終了
- 12秒のページタイムアウト
- オッズ未公開と通信失敗を分けて表示
- オッズが無い場合は期待値を捏造しない

## GitHubへ上書きする5ファイル

- app.py
- Dockerfile
- requirements.txt
- render.yaml
- README.md

アップロード後、Renderで
`Manual Deploy → Clear build cache & deploy`
を実行してください。

## 注意

Chromiumを使うためV22よりメモリ使用量が増えます。
Render Starterでプロセスが落ちる場合は、公式取得だけを別サービスへ分離する必要があります。
