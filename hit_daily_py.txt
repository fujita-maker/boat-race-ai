#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hit_daily.py — 2連単「当てにいく」特化システム（高オッズ系とは完全に別建て）。

方針:
  ・的中率重視。GBMで各艇の強さを学習し、2連単30通りの確率→上位N点(既定3点)を購入。
  ・学習も本番も同じ BoatraceOpenAPI api/v1 から同じ抽出で作る＝train/serveのズレ無し。
  ・毎回、直近 train-days 日で学習（pickle不要・sklearnバージョン非依存）→当日を予想＆精算。
  ・精算は api/v1 の着順(1着-2着)と2連単払戻(exacta)で完結（boatrace.jp不要）。
  ・台帳は data/hit_ledger.csv（高オッズの ho_ledger.csv とは別ファイル）。

使い方:
  pip install numpy scikit-learn requests
  python hit_daily.py --hd 20260814 --train-days 120 --topn 3 --ledger data/hit_ledger.csv
"""
import argparse, csv, os, sys, time, json
import datetime as dt
from typing import Any, Optional
import numpy as np
import requests
from sklearn.ensemble import HistGradientBoostingClassifier

UA = {"User-Agent": "Mozilla/5.0 (compatible; boatrace-hit/1.0)"}
OPENAPI = "https://boatraceopenapi.github.io/api/v1/{y}/{hd}.json"
TARGET_VENUES = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 21, 22, 24]
_CLASS = {"A1": 1.0, "A2": 0.66, "B1": 0.33, "B2": 0.0}
LCOLS = ["ts", "date", "jcd", "rno", "combo", "P", "rank", "payout", "stake",
         "status", "ret", "hit", "actual"]
# 特徴量（api/v1から取得できるもの）: frame, class, 全国勝率, 当地勝率, モーター2連率, 平均ST
NBASE = 6


def fetch_openapi(hd, retries=4, timeout=25):
    url = OPENAPI.format(y=hd[:4], hd=hd)
    for _ in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 404:
                return None
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


def find_race(data, jcd, rno):
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


def _collect(node, key, out):
    if isinstance(node, dict):
        if key in node:
            out.append(node)
        for v in node.values():
            _collect(v, key, out)
    elif isinstance(node, list):
        for v in node:
            _collect(v, key, out)


def _num(v, d):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def extract_features(race):
    """6艇×NBASE の特徴 + 枠番配列。api/v1のracerエントリから取得。取れない特徴はNaN。"""
    entries = []
    _collect(race, "national_win_rate", entries)
    entries = {int(e.get("entry_number", 0)): e for e in entries if e.get("entry_number")}
    if len(entries) < 6:
        return None, None
    X = []
    boats = []
    for n in range(1, 7):
        e = entries.get(n, {})
        cls = _CLASS.get(e.get("rank_number_source") or e.get("class") or "B1", 0.33)
        X.append([
            float(n),                                   # frame(枠番)
            cls,                                        # 級別
            _num(e.get("national_win_rate"), np.nan),   # 全国勝率
            _num(e.get("local_win_rate"), np.nan),      # 当地勝率
            _num(e.get("motor_top_2_percent"), np.nan), # モーター2連率
            _num(e.get("average_start_timing"), np.nan),# 平均ST
        ])
        boats.append(n)
    return np.array(X, float), np.array(boats, int)


def placements(race):
    """(1着枠index, 2着枠index) を返す。無ければ None。"""
    pls = []
    _collect(race, "place_number", pls)
    order = {}
    for p in pls:
        pn = p.get("place_number"); en = p.get("entry_number")
        if pn in (1, 2) and en:
            order[int(pn)] = int(en) - 1
    if 1 in order and 2 in order:
        return order[1], order[2]
    return None


def _payouts_exacta(race):
    """2連単の (combo, 払戻額) を返す。無ければ None。"""
    combos = []
    _collect(race, "combination", combos)
    for c in combos:
        s = str(c.get("combination", "")); amt = c.get("amount")
        if amt is None:
            continue
        if s.count("-") == 1:      # 2連単
            return s, float(amt)
    return None


# ---------- 特徴量エンジニアリング（レース内相対 + 場×枠事前勝率） ----------
def augment(Xraw, boats_list, st_list, winfr=None, appfr=None, build_prior=False, A_list=None):
    """Xraw: list of (6,NBASE). 返り値: (R,6,Fa) 拡張特徴, prior tables。"""
    Xf = np.nan_to_num(np.stack(Xraw))                 # (R,6,NBASE)
    m = Xf.mean(1, keepdims=True); sd = Xf.std(1, keepdims=True) + 1e-9
    Zin = (Xf - m) / sd
    rk = np.argsort(np.argsort(Xf, 1), 1).astype(float)
    R = len(Xf)
    if build_prior:
        winfr = np.zeros((30, 7)); appfr = np.zeros((30, 7))
        for k in range(R):
            s = int(st_list[k])
            for pos in range(6):
                appfr[s, int(boats_list[k][pos])] += 1
            winfr[s, int(boats_list[k][A_list[k]])] += 1
    vf = np.zeros((R, 6, 1))
    for k in range(R):
        s = int(st_list[k])
        for pos in range(6):
            fr = int(boats_list[k][pos])
            ap = appfr[s, fr] if s < 30 else 0
            vf[k, pos, 0] = winfr[s, fr] / ap if ap > 0 else 0.15
    Xa = np.nan_to_num(np.concatenate([Xf, Zin, rk, vf], axis=2))
    return Xa, winfr, appfr


def collect_days(hd_list, need_results):
    """複数日のapi/v1を取得し、レース群を返す。need_results=Trueは着順必須(学習/精算)。"""
    races = []
    for i, hd in enumerate(hd_list):
        data = fetch_openapi(hd)
        if data is None:
            continue
        for jcd in TARGET_VENUES:
            for rno in range(1, 13):
                rc = find_race(data, jcd, rno)
                if rc is None:
                    continue
                X, boats = extract_features(rc)
                if X is None:
                    continue
                pl = placements(rc)
                if need_results and pl is None:
                    continue
                races.append({"hd": hd, "jcd": jcd, "rno": rno, "X": X, "boats": boats,
                              "a": pl[0] if pl else None, "b": pl[1] if pl else None,
                              "race": rc})
        if (i + 1) % 20 == 0:
            print(f"  fetched {i+1}/{len(hd_list)}日 races={len(races)}", flush=True)
        time.sleep(0.15)
    return races


def smax(a):
    a = a - a.max(1, keepdims=True); e = np.exp(a); return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default="")
    ap.add_argument("--train-days", type=int, default=120)
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--stake", type=float, default=100.0)
    ap.add_argument("--ledger", default="data/hit_ledger.csv")
    a = ap.parse_args()

    jst = dt.timezone(dt.timedelta(hours=9))
    hd = a.hd or dt.datetime.now(jst).strftime("%Y%m%d")
    day0 = dt.datetime.strptime(hd, "%Y%m%d").date()

    # --- 学習データ収集（直近 train-days 日、当日は除く） ---
    train_hds = [(day0 - dt.timedelta(days=k)).strftime("%Y%m%d") for k in range(1, a.train_days + 1)]
    print(f"[hit] 学習データ収集: 直近{a.train_days}日 ...", flush=True)
    tr = collect_days(train_hds, need_results=True)
    if len(tr) < 500:
        print(f"[hit] 学習レース不足({len(tr)})。中止。", flush=True)
        return
    # 特徴診断（api/v1で何が取れているか）
    Xall = np.stack([r["X"] for r in tr])
    avail = (~np.isnan(Xall)).mean(0).mean(0)
    print(f"[hit] 学習{len(tr)}レース 特徴取得率 {[round(float(x),2) for x in avail]} "
          f"(frame,class,全国,当地,モ2,ST)", flush=True)

    st_tr = [r["jcd"] for r in tr]; bo_tr = [r["boats"] for r in tr]; A_tr = [r["a"] for r in tr]
    Xa_tr, winfr, appfr = augment([r["X"] for r in tr], bo_tr, st_tr, build_prior=True, A_list=A_tr)
    Fa = Xa_tr.shape[2]
    yw = np.zeros((len(tr), 6), int)
    for k, r in enumerate(tr):
        yw[k, r["a"]] = 1
    gb = HistGradientBoostingClassifier(max_iter=350, max_depth=5, learning_rate=0.07,
                                        l2_regularization=1.0, min_samples_leaf=40)
    gb.fit(Xa_tr.reshape(-1, Fa), yw.reshape(-1))
    print(f"[hit] GBM学習完了 (拡張特徴{Fa})", flush=True)

    # --- 当日: 確定レースを予想＆精算 ---
    data = fetch_openapi(hd)
    if data is None:
        print(f"[hit] {hd} データ無し。終了。", flush=True)
        return
    today = []
    for jcd in TARGET_VENUES:
        for rno in range(1, 13):
            rc = find_race(data, jcd, rno)
            if rc is None:
                continue
            pl = placements(rc)
            if pl is None:                    # 未確定→スキップ
                continue
            X, boats = extract_features(rc)
            if X is None:
                continue
            today.append({"jcd": jcd, "rno": rno, "X": X, "boats": boats,
                          "a": pl[0], "b": pl[1], "race": rc})
    if not today:
        print(f"[hit] {hd} 確定レース無し。終了。", flush=True)
        return
    Xa_td, _, _ = augment([r["X"] for r in today], [r["boats"] for r in today],
                          [r["jcd"] for r in today], winfr=winfr, appfr=appfr, build_prior=False)
    pw = gb.predict_proba(Xa_td.reshape(-1, Fa))[:, 1].reshape(len(today), 6)
    v = np.log(np.clip(pw, 1e-6, 1)); p1 = smax(v)

    # 既存台帳
    rows = []
    if os.path.exists(a.ledger):
        with open(a.ledger, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    seen = {(r["date"], r["jcd"], r["rno"], r["combo"]) for r in rows}
    now = dt.datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S")
    logged = 0
    for k, r in enumerate(today):
        vk = v[k]
        pr = []
        for i in range(6):
            v2 = vk.copy(); v2[i] = -1e9; p2 = smax(v2[None])[0]
            for j in range(6):
                if i != j:
                    pr.append(((i, j), p1[k, i] * p2[j]))
        pr.sort(key=lambda t: t[1], reverse=True)
        picks = pr[:a.topn]
        b = r["boats"]
        actual_idx = (r["a"], r["b"])
        actual_cs = f"{b[r['a']]}-{b[r['b']]}"
        expay = _payouts_exacta(r["race"])   # (combo, amount) or None
        for rankpos, (idx, prob) in enumerate(picks, 1):
            combo = f"{b[idx[0]]}-{b[idx[1]]}"
            key = (hd, str(r["jcd"]), str(r["rno"]), combo)
            if key in seen:
                continue
            seen.add(key)
            hit = (idx == actual_idx)
            payout = expay[1] if (hit and expay and expay[0] == combo) else 0
            ret = round(a.stake * payout / 100.0, 1) if hit and payout else 0.0
            rows.append({"ts": now, "date": hd, "jcd": r["jcd"], "rno": r["rno"], "combo": combo,
                         "P": round(float(prob), 4), "rank": rankpos, "payout": payout,
                         "stake": a.stake, "status": "settled", "ret": ret,
                         "hit": int(hit), "actual": actual_cs})
            logged += 1

    os.makedirs(os.path.dirname(a.ledger) or ".", exist_ok=True)
    rows.sort(key=lambda r: (str(r["date"]), int(r["jcd"]), int(r["rno"]), str(r["combo"])))
    with open(a.ledger, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LCOLS); w.writeheader(); w.writerows(rows)

    s = [r for r in rows if r.get("status") == "settled"]
    stk = sum(float(r["stake"]) for r in s)
    ret = sum(float(r["ret"] or 0) for r in s)
    h = sum(int(float(r["hit"] or 0)) for r in s)
    # レース単位の的中率（そのレースの上位N点のどれかが当たったか）
    byrace = {}
    for r in s:
        kk = (r["date"], r["jcd"], r["rno"])
        byrace[kk] = byrace.get(kk, 0) or int(float(r["hit"] or 0))
    race_hit = sum(byrace.values()); race_n = len(byrace)
    roi = (ret / stk * 100) if stk else 0
    print(f"[hit] hd={hd} 当日{len(today)}確定 記録{logged}点(上位{a.topn})", flush=True)
    print(f"[通算] レース{race_n} 的中{race_hit}({(race_hit/race_n*100 if race_n else 0):.1f}%) "
          f"投資{stk:,.0f}円 払戻{ret:,.0f}円 純益{ret-stk:,.0f}円 回収率{roi:.1f}%", flush=True)


if __name__ == "__main__":
    main()
