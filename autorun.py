#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""autorun.py — 毎日「予想 → 結果照合 → 台帳追記」をGitHub Actions上で完結させる。

Renderアプリ(app.py)の autorun と同一ロジックを移植した単体版。
  ・データ元: BoatraceOpenAPI(api/v1 日次JSON) … 出走表+直前+結果+払戻を含む。
  ・オッズ  : boatrace.jp odds2tf(締切時=最終オッズ。過去日も残る)。
  ・対象    : ゲート(自信度×本命の抜け)通過レースの 1-4 / 1-5 で
              オッズ>=20倍 かつ EV(=P×オッズ)>=1.20 の2連単だけ「買い」。
  ・精算    : 確定結果(2連単の1着-2着と払戻)で即精算し ho_ledger.csv に追記。

Renderに叩かせず全部ここでやるので、リクエストのタイムアウトが起きない。
GitHub Actions は boatrace.jp に到達でき、実行時間の上限も緩い(collect-oddsで実証済み)。

使い方:
  pip install numpy requests beautifulsoup4 lxml
  python autorun.py --hd 20260814 --ledger data/ho_ledger.csv
  # --hd 省略で当日JST
"""
import argparse, csv, os, sys, time
import datetime as dt
from typing import Any, Optional
import numpy as np
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (compatible; boatrace-autorun/1.0)"}
OPENAPI = "https://boatraceopenapi.github.io/api/v1/{y}/{hd}.json"
TARGET_VENUES = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 21, 22, 24]

# ---- 焼き込み条件付きロジット(app.pyと同一) ----
_LB = [0.6958, 0.1224, 0.728, 0.0152, 0.1583, -0.0137, -0.421]
_LM = [0.1668, 0.5436, 5.2401, 4.6892, 32.7627, 0.088, 6.826]
_LSD = [0.1764, 0.3059, 1.3348, 2.1123, 11.0123, 0.1053, 0.1136]
_CLASS = {"A1": 1.0, "A2": 0.66, "B1": 0.33, "B2": 0.0}
_CBASE = {1: 0.55, 2: 0.15, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.02}
LCOLS = ["ts", "date", "jcd", "rno", "combo", "P", "odds", "market_p", "EV",
         "stake", "status", "payout", "ret", "hit", "star", "women"]


def _num(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fetch_openapi(hd: str, retries=4, timeout=20) -> Optional[Any]:
    url = OPENAPI.format(y=hd[:4], hd=hd)
    for _ in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 404:
                return None            # その日は開催データ無し
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(2)
    return None


def _as_items(node):
    if isinstance(node, dict):
        return list(node.values())
    if isinstance(node, list):
        return node
    return []


def find_race(data: Any, jcd: int, rno: int) -> Optional[dict]:
    progs = data.get("programs", data) if isinstance(data, dict) else data
    stadiums = progs.get("stadiums") if isinstance(progs, dict) else None
    if stadiums is None:
        return None
    stadium = None
    for s in _as_items(stadiums):
        if isinstance(s, dict) and int(s.get("stadium_number", -1)) == jcd:
            stadium = s
            break
    if stadium is None and isinstance(stadiums, dict) and str(jcd) in stadiums:
        stadium = stadiums[str(jcd)]
    if not isinstance(stadium, dict):
        return None
    races = stadium.get("races")
    for rc in _as_items(races):
        if isinstance(rc, dict) and int(rc.get("race_number", -1)) == rno:
            return rc
    if isinstance(races, dict) and str(rno) in races:
        return races[str(rno)]
    return None


def _collect_with_key(node, key, out):
    if isinstance(node, dict):
        if key in node:
            out.append(node)
        for v in node.values():
            _collect_with_key(v, key, out)
    elif isinstance(node, list):
        for v in node:
            _collect_with_key(v, key, out)


def extract_boats(race: dict) -> list:
    entries = []
    _collect_with_key(race, "national_win_rate", entries)
    entries = {int(e.get("entry_number", 0)): e for e in entries if e.get("entry_number")}
    prevs = []
    _collect_with_key(race, "exhibition_time", prevs)
    prevs = {int(p.get("entry_number", 0)): p for p in prevs if p.get("entry_number")}
    boats = []
    for n in range(1, 7):
        e = entries.get(n, {})
        p = prevs.get(n, {})
        course = p.get("course_number") or n
        st = p.get("start_timing")
        if st is None:
            st = e.get("average_start_timing")
        boats.append({
            "frame": n, "course": int(course),
            "cls": e.get("rank_number_source") or "B1",
            "nat": e.get("national_win_rate"), "loc": e.get("local_win_rate"),
            "motor": e.get("motor_top_2_percent"), "ex": p.get("exhibition_time"),
            "st": st, "name": e.get("name"), "odds": None,
        })
    return boats


def _payouts(race):
    combos = []
    _collect_with_key(race, "combination", combos)
    ex = tri = None
    for c in combos:
        s = str(c.get("combination", "")); amt = c.get("amount")
        if amt is None:
            continue
        if s.count("-") == 2 and tri is None:
            tri = (s, float(amt))
        elif s.count("-") == 1 and ex is None:
            ex = (s, float(amt))
    return ex, tri


def fetch_exacta_odds(jcd: int, rno: int, hd: str, retries=3, timeout=15) -> Optional[dict]:
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
            seconds_by_first = {f: [s for s in range(1, 7) if s != f] for f in range(1, 7)}
            odds = {}
            idx = 0
            for row_i in range(5):
                for first in range(1, 7):
                    second = seconds_by_first[first][row_i]
                    odds[f"{first}-{second}"] = vals[idx]
                    idx += 1
            return odds
        except Exception:
            time.sleep(1.5)
    return None


def predict_power(boats):
    sc = []
    for b in boats:
        f = [_CBASE.get(b["course"], 0.05), _CLASS.get(b["cls"], 0.33),
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


def highodds_pick(boats, odds_map, odds_min, ev_min, p_floor, min_conf, min_gap, max_n, focus=None):
    power = predict_power(boats)
    conf, gap = _conf_gap(power)
    if conf < min_conf or gap < min_gap:
        return None
    ex = _exacta_probs(power); cand = []
    for (i, j), p in ex.items():
        cs0 = f"{i}-{j}"
        if focus and cs0 not in focus:
            continue
        o = odds_map.get(cs0) if odds_map else None
        if o is None or o < odds_min or p < p_floor:
            continue
        ev = p * o
        if ev < ev_min:
            continue
        cand.append({"combo": cs0, "P": round(p, 4), "odds": o,
                     "market_p": round(1 / o, 4), "EV": round(ev, 3)})
    cand.sort(key=lambda c: c["P"], reverse=True)
    return cand[:max_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default="")
    ap.add_argument("--ledger", default="data/ho_ledger.csv")
    ap.add_argument("--odds-min", type=float, default=20.0)
    ap.add_argument("--ev-min", type=float, default=1.20)
    ap.add_argument("--p-floor", type=float, default=0.05)
    ap.add_argument("--min-conf", type=float, default=0.12)
    ap.add_argument("--min-gap", type=float, default=0.30)
    ap.add_argument("--max-n", type=int, default=2)
    ap.add_argument("--stake", type=float, default=100.0)
    ap.add_argument("--focus", default="1-4,1-5")
    ap.add_argument("--sleep", type=float, default=0.6)
    a = ap.parse_args()

    hd = a.hd or dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d")
    focus_set = {c.strip() for c in a.focus.split(",") if c.strip()} or None
    print(f"[autorun] hd={hd} 対象{len(TARGET_VENUES)}場 focus={sorted(focus_set) if focus_set else '全'}", flush=True)

    data = fetch_openapi(hd)
    if data is None:
        print(f"[autorun] {hd} のOpenAPIデータ無し(開催なし/未確定)。終了。", flush=True)
        return

    # 既存台帳を読み込み(重複防止)
    rows = []
    if os.path.exists(a.ledger):
        with open(a.ledger, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    seen = {(r["date"], r["jcd"], r["rno"], r["combo"]) for r in rows}
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")

    races = gated = logged = 0
    for jcd in TARGET_VENUES:
        vlog = 0
        for rno in range(1, 13):
            race = find_race(data, jcd, rno)
            if race is None:
                continue
            ex, tri = _payouts(race)
            if ex is None:                       # 未確定→スキップ
                continue
            races += 1
            boats = extract_boats(race)
            conf, gap = _conf_gap(predict_power(boats))
            if conf < a.min_conf or gap < a.min_gap:   # ゲート未通過は見送り
                continue
            gated += 1
            odds_map = fetch_exacta_odds(jcd, rno, hd)
            time.sleep(a.sleep)
            if not odds_map:
                continue
            picks = highodds_pick(boats, odds_map, a.odds_min, a.ev_min, a.p_floor,
                                  a.min_conf, a.min_gap, a.max_n, focus_set)
            if not picks:
                continue
            wc, amt = ex
            for p in picks:
                key = (str(hd), str(jcd), str(rno), str(p["combo"]))
                if key in seen:
                    continue
                seen.add(key)
                hit = (p["combo"] == wc)
                ret = round(a.stake * amt / 100.0, 1) if hit else 0.0
                rows.append({"ts": now, "date": hd, "jcd": jcd, "rno": rno, "combo": p["combo"],
                             "P": p["P"], "odds": p["odds"], "market_p": p["market_p"], "EV": p["EV"],
                             "stake": a.stake, "status": "settled", "payout": (amt if hit else 0),
                             "ret": ret, "hit": int(hit), "star": 0, "women": 0})
                logged += 1; vlog += 1
        if vlog:
            print(f"  {jcd}場: {vlog}件記録", flush=True)

    # 保存(日付→場→R→組でソート)
    os.makedirs(os.path.dirname(a.ledger) or ".", exist_ok=True)
    rows.sort(key=lambda r: (str(r["date"]), int(r["jcd"]), int(r["rno"]), str(r["combo"])))
    with open(a.ledger, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LCOLS); w.writeheader(); w.writerows(rows)

    s = [r for r in rows if r.get("status") == "settled"]
    stk = sum(float(r["stake"]) for r in s)
    ret = sum(float(r["ret"] or 0) for r in s)
    h = sum(int(float(r["hit"] or 0)) for r in s)
    roi = (ret / stk * 100) if stk else 0
    print(f"[autorun] hd={hd} 確定レース{races} ゲート通過{gated} 今回記録{logged}", flush=True)
    print(f"[通算] 精算{len(s)}買い目 的中{h}({(h/len(s)*100 if s else 0):.1f}%) "
          f"投資{stk:,.0f}円 払戻{ret:,.0f}円 純益{ret-stk:,.0f}円 回収率{roi:.1f}%", flush=True)


if __name__ == "__main__":
    main()
