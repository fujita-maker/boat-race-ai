# ボートレース 妙味スコア予想（自動更新版）

場・レース・日付を選ぶと、出走表と直前情報（展示タイム・ST・進入コース）を自動取得し、
「1着力」で頭を自動判定 → 2着の妙味（人気薄なのに走れる艇）を抽出して買い目を提案します。
2連単オッズは締切まで自動更新（30秒ごと）。オッズが取れない時は級別・勝率からの推定に自動フォールバックします。

## データの出どころ

- **出走表・直前情報** … `boatraceopenapi`（GitHub Pages配信のJSON、約3分間隔更新）。安定して取れます。
- **2連単オッズ** … `boatrace.jp` を best-effort でスクレイプ。公式にオッズAPIが無いため、ここだけHTML解析です。

## ローカルで動かす

```bash
pip install -r requirements.txt
uvicorn app:app --reload
# ブラウザで http://127.0.0.1:8000
```

## Render にデプロイ（GitHub経由）

1. このフォルダを GitHub のリポジトリに push する。
2. Render → New → **Web Service** → 対象リポジトリを選択。
3. 設定（`render.yaml` があるので自動で埋まります）
   - Runtime: Python
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Plan: Free
4. Deploy。発行された `https://xxx.onrender.com` を開けば使えます。

> 無料プランはアクセスが無いとスリープし、初回アクセスに数十秒かかります。

## 動作確認とオッズ解析の検証（重要）

`/api/race?jcd=19&rno=5&hd=YYYYMMDD` を直接開くとJSONが返ります。`odds_status` を確認してください。

- `ライブ（boatrace.jp）` … オッズ取得成功。**返ってきた `odds_all` を、公式サイトの実際のオッズ表と数レース見比べてください。**
- `推定（オッズ未取得）` … 取得失敗。締切前の発売中のレースでこれが続く場合は、boatrace.jp 側の
  HTML構造（`td.oddsPoint` の並び順）が想定と違う可能性があります。

オッズの数値が公式と食い違う／常に推定になる場合は、`app.py` の `fetch_exacta_odds()` の
セル並び順マッピングを直す必要があります（1か所）。実際のオッズ頁を見れば即修正できるので、その旨伝えてください。

## デプロイが `streamlit: not found`（status 127）で失敗する時

このサービスは以前 **streamlit で起動する設定**のまま残っています。中身は FastAPI/uvicorn なので、起動を直します。

1. リポジトリに、この **`Dockerfile`（uvicorn で起動）を追加**する。古い `Dockerfile` があれば**置き換える**。
   古い streamlit 用ファイル（例：`streamlit_app.py` など）が残っていれば削除。
2. Render → **Settings** を開き、**Docker Command / Start Command** の欄を確認。
   `streamlit run ...` と入っていたら、**空欄にする**（Dockerfile の CMD を使わせる）か、
   `uvicorn app:app --host 0.0.0.0 --port $PORT` に書き換える。
3. 右上 **Manual Deploy → “Clear build cache & deploy”** で再デプロイ。
4. 起動ログに `Uvicorn running on ...` と出れば成功。発行URLを開く。

> Docker タイプのままでOKです（Dockerfile を置けば動きます）。Python タイプで作り直したい場合は、
> サービスを削除して New → Web Service → Runtime=Python、Build=`pip install -r requirements.txt`、
> Start=`uvicorn app:app --host 0.0.0.0 --port $PORT` で作成します。

## 忘れてはいけない前提

- 競艇の払戻率は75%固定（控除率25%）。このツールは**勝ちを保証しません**。
- オッズは変動します。締切直前に必ず最終確認を。
- 価値があるのは「買い/見送りを一貫させ、当たり外れ両方を記録し、通算回収率で検証すること」。
  ◎本命妙味だけを買った回収率が、100件以上で安定して100%超なら、初めて“エッジがあるかも”と言えます。
