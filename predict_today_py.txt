#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict_today.py — 今日の未開催レースについて、両モデルの推奨買い目を today_picks.json に出力。

- 的中モデル(GBM): hit_daily.py を再利用して直近120日で学習→上位3点＋自信度＋見送りゲート(上位20%)。
- 高オッズモデル: 焼き込み条件付きロジット＋boatrace.jpの2連単オッズで 買い/見送り 判定(1-4/1-5・オッズ≥20・EV≥1.2)。
- 未開催(=まだ着順が無い)レースだけ対象。GitHub Actionsで1日数回走らせ、締切に近いほどオッズが最終値に近づく。

出力: data/today_picks.json
使い方: pip install numpy scikit-learn requests beautifulsoup4 lxml
        python predict_today.py --hd 20260815 --train-days 120 --topn 3 --out data/today_picks.json
"""
import argparse, json, os, time
import datetime as dt
import numpy as np
import requests
from bs4 import BeautifulSoup
import hit_daily as H          # 実証済みの的中モデル部品を再利用

UA = {"User-Agent": "Mozilla/5.0 (compatible; boatrace-today/1.0)"}
# ---- 高オッズ(焼き込み条件付きロジット) ----
_LB = [0.6958, 0.1224, 0.728, 0.0152, 0.1583, -0.0137, -0.421]
_LM = [0.1668, 0.5436, 5.2401, 4.6892, 32.7627, 0.088, 6.826]
_LSD = [0.1764, 0.3059, 1.3348, 2.1123, 11.0123, 0.1053, 0.1136]
_CBASE = {1: 0.55, 2: 0.15, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.02}
_CLASSN = {"A1": 1.0, "A2": 0.66, "B1": 0.33, "B2": 0.0}


def _num(v, d):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def extract_boats(race):
    ents = []
    H._collect(race, "national_win_rate", ents)
    ents = {int(e.get("entry_number", 0)): e for e in ents if e.get("entry_number")}
    prevs = []
    H._collect(race, "exhibition_time", prevs)
    prevs = {int(p.get("entry_number", 0)): p for p in prevs if p.get("entry_number")}
    boats = []
    for n in range(1, 7):
        e = ents.get(n, {}); p = prevs.get(n, {})
        course = p.get("course_number") or n
        st = p.get("start_timing")
        if st is None:
            st = e.get("average_start_timing")
        boats.append({"frame": n, "course": int(course),
                      "cls": e.get("rank_number_source") or "B1",
                      "nat": e.get("national_win_rate"), "loc": e.get("local_win_rate"),
                      "motor": e.get("motor_top_2_percent"), "ex": p.get("exhibition_time"), "st": st})
    return boats


def predict_power(boats):
    sc = []
    for b in boats:
        f = [_CBASE.get(b["course"], 0.05), _CLASSN.get(b["cls"], 0.33),
             _num(b["nat"], np.nan), _num(b["loc"], np.nan), _num(b["motor"], np.nan),
             _num(b["st"], np.nan), _num(b["ex"], np.nan)]
        s = 0.0
        for k in range(7):
            z = (f[k] - _LM[k]) / _LSD[k]
            if not np.isnan(z):
                s += _LB[k] * z
        sc.append(s)
    s = np.asarray(sc); s -= s.max(); e = np.exp(s); p = e / e.sum()
    return {b["frame"]: float(pi) for b, pi in zip(boats, p)}


def _conf_gap(power):
    ps = sorted(power.items(), key=lambda kv: kv[1], reverse=True)
    pv = [v for _, v in ps]
    gap = (pv[0] - pv[1]) / pv[0] if pv and pv[0] > 0 else 0.0
    conf = 0.0
    if len(ps) >= 3:
        pi, pj, pk = pv[0], pv[1], pv[2]
        d1, d2 = 1 - pi, 1 - pi - pj
        if d1 > 1e-9 and d2 > 1e-9:
            conf = pi * (pj / d1) * (pk / d2)
    return conf, gap


def _exacta_probs(power):
    out = {}
    for i in range(1, 7):
        for j in range(1, 7):
            if i == j:
                continue
            pi, pj = power.get(i, 0.0), power.get(j, 0.0); d = 1 - pi
            out[(i, j)] = pi * (pj / d) if d > 1e-9 else 0.0
    return out


def fetch_exacta_odds(jcd, rno, hd, retries=3, timeout=15):
    url = f"https://boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd:02d}&hd={hd}"
    for _ in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code != 200:
                time.sleep(1.0); continue
            soup = BeautifulSoup(r.text, "lxml")
            cells = soup.select("td.oddsPoint")
            vals = []
            for c in cells:
                t = c.get_text(strip=True).replace(",", "")
                try:
                    vals.append(float(t))
                except ValueError:
                    vals.append(None)
            if len(vals) < 30:
                return None
            vals = vals[:30]
            sec = {f: [s for s in range(1, 7) if s != f] for f in range(1, 7)}
            odds = {}; idx = 0
            for row_i in range(5):
                for first in range(1, 7):
                    odds[f"{first}-{sec[first][row_i]}"] = vals[idx]; idx += 1
            return odds
        except Exception:
            time.sleep(1.2)
    return None


def highodds_pick(boats, odds_map, odds_min=20.0, ev_min=1.20, p_floor=0.05,
                  min_conf=0.12, min_gap=0.30, max_n=2, focus={"1-4", "1-5"}):
    power = predict_power(boats)
    conf, gap = _conf_gap(power)
    if conf < min_conf or gap < min_gap:
        return {"decision": "見送り", "reason": "レースゲート未通過", "picks": []}
    ex = _exacta_probs(power); cand = []
    for (i, j), p in ex.items():
        cs = f"{i}-{j}"
        if focus and cs not in focus:
            continue
        o = odds_map.get(cs) if odds_map else None
        if o is None or o < odds_min or p < p_floor:
            continue
        ev = p * o
        if ev < ev_min:
            continue
        cand.append({"combo": cs, "P": round(p, 4), "odds": o, "EV": round(ev, 3)})
    cand.sort(key=lambda c: c["P"], reverse=True)
    picks = cand[:max_n]
    return {"decision": "買い" if picks else "見送り",
            "reason": "高オッズEV＋" if picks else "候補なし", "picks": picks}


def race_closed_at(race):
    for k in ("race_closed_at", "closed_at", "race_close_time"):
        if isinstance(race, dict) and race.get(k):
            return str(race.get(k))
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default="")
    ap.add_argument("--train-days", type=int, default=120)
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--conf-quantile", type=float, default=0.80)
    ap.add_argument("--out", default="data/today_picks.json")
    a = ap.parse_args()

    jst = dt.timezone(dt.timedelta(hours=9))
    hd = a.hd or dt.datetime.now(jst).strftime("%Y%m%d")

    # --- 的中GBMを学習（hit_daily再利用） ---
    day0 = dt.datetime.strptime(hd, "%Y%m%d").date()
    train_hds = [(day0 - dt.timedelta(days=k)).strftime("%Y%m%d") for k in range(1, a.train_days + 1)]
    print(f"[today] 的中GBM学習: 直近{a.train_days}日 ...", flush=True)
    tr = H.collect_days(train_hds, need_results=True)
    gb = conf_thr = winfr = appfr = Fa = None
    if len(tr) >= 500:
        Xa_tr, winfr, appfr = H.augment([r["X"] for r in tr], [r["boats"] for r in tr],
                                        [r["jcd"] for r in tr], build_prior=True, A_list=[r["a"] for r in tr])
        Fa = Xa_tr.shape[2]
        yw = np.zeros((len(tr), 6), int)
        for k, r in enumerate(tr):
            yw[k, r["a"]] = 1
        gb = H.HistGradientBoostingClassifier(max_iter=350, max_depth=5, learning_rate=0.07,
                                              l2_regularization=1.0, min_samples_leaf=40)
        gb.fit(Xa_tr.reshape(-1, Fa), yw.reshape(-1))
        pw_tr = gb.predict_proba(Xa_tr.reshape(-1, Fa))[:, 1].reshape(len(tr), 6)
        v_tr = np.log(np.clip(pw_tr, 1e-6, 1))
        confs = np.array([H.topn_conf(v_tr[k], a.topn)[1] for k in range(len(tr))])
        conf_thr = float(np.quantile(confs, a.conf_quantile))
        print(f"[today] GBM学習完了。見送りゲート自信度>={conf_thr:.3f}", flush=True)
    else:
        print(f"[today] 学習データ不足({len(tr)})。的中予想はスキップ。", flush=True)

    data = H.fetch_openapi(hd)
    if data is None:
        json.dump({"hd": hd, "generated_at": dt.datetime.now(jst).strftime("%Y-%m-%d %H:%M"),
                   "races": [], "note": "本日データ無し"}, open(a.out, "w"), ensure_ascii=False)
        print("[today] 本日データ無し。空で出力。", flush=True)
        return

    races = []
    ho_buy = hit_go = up = 0
    for jcd in H.TARGET_VENUES:
        for rno in range(1, 13):
            rc = H.find_race(data, jcd, rno)
            if rc is None:
                continue
            if H.placements(rc) is not None:      # 既に確定=未開催でない→対象外
                continue
            Xf, boats_n = H.extract_features(rc)
            if Xf is None:
                continue
            up += 1
            rec = {"jcd": jcd, "rno": rno, "closed_at": race_closed_at(rc),
                   "hit": None, "ho": None}
            # 的中モデル
            if gb is not None:
                Xa, _, _ = H.augment([Xf], [boats_n], [jcd], winfr=winfr, appfr=appfr, build_prior=False)
                pw = gb.predict_proba(Xa.reshape(-1, Fa))[:, 1].reshape(1, 6)
                v = np.log(np.clip(pw, 1e-6, 1))[0]
                picks, conf = H.topn_conf(v, a.topn)
                combos = [f"{boats_n[i]}-{boats_n[j]}" for (i, j), _ in picks]
                gate = conf >= conf_thr
                rec["hit"] = {"combos": combos, "conf": round(float(conf), 3), "buy": bool(gate)}
                if gate:
                    hit_go += 1
            # 高オッズモデル
            boats = extract_boats(rc)
            odds_map = fetch_exacta_odds(jcd, rno, hd)
            res = highodds_pick(boats, odds_map)
            rec["ho"] = {"decision": res["decision"],
                         "picks": [{"combo": p["combo"], "odds": p["odds"], "EV": p["EV"]} for p in res["picks"]]}
            if res["decision"] == "買い":
                ho_buy += 1
            time.sleep(0.25)
            races.append(rec)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out = {"hd": hd, "generated_at": dt.datetime.now(jst).strftime("%Y-%m-%d %H:%M"),
           "topn": a.topn, "races": races}
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print(f"[today] 未開催{up}R → 的中買い{hit_go}R / 高オッズ買い{ho_buy}R を {a.out} に出力", flush=True)


if __name__ == "__main__":
    main()
