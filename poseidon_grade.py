#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""poseidon_grade.py — 「1号艇軸＋高オッズ＋EVプラス」で3連単6-8点を選び、実結果で精算して台帳化。

入力: data/poseidon/{hd}.json (poseidon_scrape.py がオッズ付きで保存したもの)
      OpenAPI結果 (autorun.fetch_openapi 経由)
選定(1レース): 1着=1号艇(--axis) の組で オッズ>=--odds-min かつ EV=AI確率×オッズ>=--ev-min を満たすものを
      EV降順に最大 --points 点。無ければ見送り。
精算: 実際の3連単配当で精算 → data/poseidon_ledger.csv に追記(重複除外)。通算ROIを表示。
使い方: python poseidon_grade.py --hd 20260821
"""
import argparse, csv, json, os
import datetime as dt
from autorun import fetch_openapi, find_race, _payouts

LCOLS = ["ts","date","jcd","rno","combo","p","odds","kaijin","ev",
         "stake","status","win_combo","hit","payout","ret"]

def select(race_probs, axis, odds_min, ev_min, points):
    cands = []
    for combo, v in race_probs.items():
        p, odds, kaijin = (v + [None, None])[:3]
        if odds is None:                       # オッズ未取得はEV計算不可
            continue
        if axis and not combo.startswith(f"{axis}-"):
            continue
        if odds < odds_min:
            continue
        ev = (p / 100.0) * odds
        if ev < ev_min:
            continue
        cands.append({"combo": combo, "p": p, "odds": odds,
                      "kaijin": kaijin, "ev": round(ev, 3)})
    cands.sort(key=lambda c: -c["ev"])
    return cands[:points]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default="")
    ap.add_argument("--pose", default="data/poseidon")
    ap.add_argument("--ledger", default="data/poseidon_ledger.csv")
    ap.add_argument("--axis", default="1")        # 1着=1号艇
    ap.add_argument("--odds-min", type=float, default=20.0)
    ap.add_argument("--ev-min", type=float, default=1.0)
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--stake", type=float, default=100.0)
    a = ap.parse_args()
    hd = a.hd or dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d")

    pj = os.path.join(a.pose, f"{hd}.json")
    if not os.path.exists(pj):
        print(f"[grade] {pj} が無い。先に poseidon_scrape.py を。終了。"); return
    pose = json.load(open(pj))

    data = fetch_openapi(hd)
    if data is None:
        print(f"[grade] {hd} の結果データ無し(未確定)。終了。"); return

    rows = []
    if os.path.exists(a.ledger):
        rows = list(csv.DictReader(open(a.ledger, newline="", encoding="utf-8")))
    seen = {(r["date"], r["jcd"], r["rno"], r["combo"]) for r in rows}
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")

    bought = races = skipped = 0
    for key, race_probs in pose.items():
        jcd, rno = key.split("_")
        jcd, rno = int(jcd), int(rno)
        race = find_race(data, jcd, rno)
        if race is None:
            continue
        ex, tri = _payouts(race)
        if tri is None:                 # 未確定
            continue
        races += 1
        picks = select(race_probs, a.axis, a.odds_min, a.ev_min, a.points)
        if not picks:
            skipped += 1
            continue
        wtc, amt = tri
        for pk in picks:
            k = (str(hd), str(jcd), str(rno), pk["combo"])
            if k in seen:
                continue
            seen.add(k)
            hit = (pk["combo"] == wtc)
            ret = round(a.stake * amt / 100.0, 1) if hit else 0.0
            rows.append({"ts": now, "date": hd, "jcd": jcd, "rno": rno,
                         "combo": pk["combo"], "p": pk["p"], "odds": pk["odds"],
                         "kaijin": pk["kaijin"], "ev": pk["ev"], "stake": a.stake,
                         "status": "settled", "win_combo": wtc,
                         "hit": int(hit), "payout": (amt if hit else 0), "ret": ret})
            bought += 1

    os.makedirs(os.path.dirname(a.ledger) or ".", exist_ok=True)
    rows.sort(key=lambda r: (str(r["date"]), int(r["jcd"]), int(r["rno"]), str(r["combo"])))
    with open(a.ledger, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LCOLS); w.writeheader(); w.writerows(rows)

    s = [r for r in rows if r.get("status") == "settled"]
    stk = sum(float(r["stake"]) for r in s)
    ret = sum(float(r["ret"] or 0) for r in s)
    h = sum(int(float(r["hit"] or 0)) for r in s)
    # レース(束)単位の的中
    grp = {}
    for r in s:
        g = (r["date"], r["jcd"], r["rno"])
        grp[g] = grp.get(g, 0) or int(float(r["hit"] or 0))
    roi = (ret / stk * 100) if stk else 0
    print(f"[grade] {hd} 対象{races}レース 購入{bought}点 見送り{skipped}レース", flush=True)
    print(f"[通算] {len(s)}点 / {len(grp)}レース(束的中{sum(grp.values())}={sum(grp.values())/len(grp)*100 if grp else 0:.1f}%) "
          f"投資{stk:,.0f}円 払戻{ret:,.0f}円 純益{ret-stk:,.0f}円 回収率{roi:.1f}%", flush=True)

if __name__ == "__main__":
    main()
