# WDJ Boat Race AI Web v13 過去学習版

## 新機能
- 過去データCSVを読み込み
- 現在ルールの過去検証
- ポセイドンと梅吉AIの比重を自動最適化
- 購入期待値基準を自動最適化
- 最大購入点数を自動最適化
- 学習結果を今後のライブ予想へ反映
- 学習前後の収支・回収率・的中率を表示

## 必要な学習CSV列
race_date, venue, race_no, combo, odds,
poseidon_prob, umepyon_prob, fallback_prob,
actual_combo, payout_100

## 注意
RenderではSQLiteデータが再デプロイで消える場合があります。
本運用ではPersistent Diskまたは外部DBを推奨します。
