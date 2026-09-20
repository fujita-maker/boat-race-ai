"""
ボートレース 妙味スコア予想 バックエンド
- 出走表・直前情報（展示/ST/コース）: boatraceopenapi の JSON（安定）
- 2連単オッズ: boatrace.jp を best-effort スクレイプ（取れなければ null → フロントは推定にフォールバック）
FastAPI / Render 対応。
"""
import os
import time
import json
import gzip
import base64
import math
import re
import datetime as dt
from typing import Any, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Boatrace Myoumi API")

UA = {"User-Agent": "Mozilla/5.0 (compatible; boatrace-myoumi/1.0)"}
OPENAPI = "https://boatraceopenapi.github.io/api/v1/{y}/{hd}.json"

# ---- openapi 日次ファイルの簡易キャッシュ（3分）----
_cache: dict[str, tuple[float, Any]] = {}

def fetch_openapi(hd: str) -> Any:
    now = time.time()
    if hd in _cache and now - _cache[hd][0] < 180:
        return _cache[hd][1]
    url = OPENAPI.format(y=hd[:4], hd=hd)
    r = requests.get(url, headers=UA, timeout=15)
    r.raise_for_status()
    data = r.json()
    _cache[hd] = (now, data)
    return data

# ---- 日次JSONから対象レースを取り出す（構造に多少の揺れがあっても拾えるように）----
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
    if stadium is None:
        # dict が番号キーの場合
        if isinstance(stadiums, dict) and str(jcd) in stadiums:
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

def extract_boats(race: dict) -> list[dict]:
    # 出走表エントリ（national_win_rate を持つ dict が各艇）
    entries: list[dict] = []
    _collect_with_key(race, "national_win_rate", entries)
    entries = {int(e.get("entry_number", 0)): e for e in entries if e.get("entry_number")}
    # 直前(preview: exhibition_time を持つ dict)
    prevs: list[dict] = []
    _collect_with_key(race, "exhibition_time", prevs)
    prevs = {int(p.get("entry_number", 0)): p for p in prevs if p.get("entry_number")}

    boats = []
    for n in range(1, 7):
        e = entries.get(n, {})
        p = prevs.get(n, {})
        course = p.get("course_number") or n
        st = p.get("start_timing")
        if st is None:
            st = e.get("average_start_timing")  # 直前が未確定なら平均STで代用
        boats.append({
            "frame": n,
            "course": int(course),
            "cls": e.get("rank_number_source") or "B1",
            "nat": e.get("national_win_rate"),
            "loc": e.get("local_win_rate"),
            "motor": e.get("motor_top_2_percent"),
            "ex": p.get("exhibition_time"),
            "st": st,
            "name": e.get("name"),
            "odds": None,  # 後で埋める
        })
    return boats

# ---- boatrace.jp から 2連単オッズ（best-effort）----
# 軸(1着) -> 各2着 のオッズを返す。取得/解析に失敗したら None を返す。
def fetch_exacta_odds(jcd: int, rno: int, hd: str) -> Optional[dict]:
    url = f"https://boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd:02d}&hd={hd}"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    cells = soup.select("td.oddsPoint")
    # 2連単は 30通り。
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
    # boatrace.jp の2連単表は「1着1〜6の6列」を横に並べた1枚の表で、oddsPoint セルは
    # 行方向（各行=6列分）に並ぶ。各列(1着番号)内は「自分以外の艇番を昇順」で1行ずつ。
    # row-major で復元する（実オッズと数値一致を確認済み）。
    seconds_by_first = {f: [s for s in range(1, 7) if s != f] for f in range(1, 7)}
    odds = {}
    idx = 0
    for row_i in range(5):
        for first in range(1, 7):
            second = seconds_by_first[first][row_i]
            odds[(first, second)] = vals[idx]
            idx += 1
    return {f"{k[0]}-{k[1]}": v for k, v in odds.items()}

@app.get("/api/race")
def api_race(jcd: int = Query(...), rno: int = Query(...),
             hd: str = Query(...), with_odds: bool = Query(True)):
    try:
        data = fetch_openapi(hd)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"openapi取得失敗: {e}"}, status_code=502)
    race = find_race(data, jcd, rno)
    if race is None:
        return JSONResponse({"ok": False, "error": "該当レースが見つかりません（日付/場/レース番号を確認、または未発売）"}, status_code=404)
    boats = extract_boats(race)

    axis_guess = min(boats, key=lambda b: b["course"])  # 進入1コース
    odds_map = None
    odds_status = "推定（オッズ未取得）"
    if with_odds:
        odds_map = fetch_exacta_odds(jcd, rno, hd)
        if odds_map:
            odds_status = "ライブ（boatrace.jp）"
            first = axis_guess["frame"]
            for b in boats:
                if b["frame"] != first:
                    b["odds"] = odds_map.get(f"{first}-{b['frame']}")
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    return {
        "ok": True,
        "jcd": jcd, "rno": rno, "hd": hd,
        "boats": boats,
        "odds_status": odds_status,
        "odds_all": odds_map,
        "updated_at": now.strftime("%H:%M:%S"),
        "note": "出走表/直前はopenapi(約3分更新)、オッズはboatrace.jp best-effort",
    }

# ================= 過去データ学習（条件付きロジットで1着確率を較正）=================
_CLASS = {"A1": 1.0, "A2": 0.66, "B1": 0.33, "B2": 0.0}
_CBASE = {1: 0.55, 2: 0.15, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.02}

def _num(v, default):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return default

def race_to_features(race: dict):
    """1レースを (6艇×7特徴, 勝者index) に変換。結果が無ければ None。"""
    boats = extract_boats(race)  # frame,course,cls,nat,loc,motor,ex,st
    # 勝者（place_number==1 の艇）
    places = []
    _collect_with_key(race, "place_number", places)
    winner = None
    for p in places:
        if p.get("place_number") == 1 and p.get("entry_number"):
            winner = int(p["entry_number"])
            break
    if winner is None:
        return None
    X = []
    for b in boats:
        X.append([
            _CBASE.get(b["course"], 0.05),
            _CLASS.get(b["cls"], 0.33),
            _num(b["nat"], 5.0),
            _num(b["loc"], 5.0),
            _num(b["motor"], 35.0),
            _num(b["st"], 0.16),
            _num(b["ex"], 6.80),
        ])
    return X, winner - 1  # index 0-5

def collect_dataset(days: int, step: int):
    """今日からdays日分をstep間隔でサンプルし、(X, winners) を集める。"""
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()
    Xs, ws, used_days = [], [], 0
    for d in range(1, days + 1, max(1, step)):
        day = today - dt.timedelta(days=d)
        hd = day.strftime("%Y%m%d")
        try:
            data = fetch_openapi(hd)
        except Exception:
            continue
        progs = data.get("programs", data) if isinstance(data, dict) else data
        stadiums = progs.get("stadiums") if isinstance(progs, dict) else None
        if not stadiums:
            continue
        got = 0
        for s in _as_items(stadiums):
            for rc in _as_items(s.get("races") if isinstance(s, dict) else None):
                if not isinstance(rc, dict):
                    continue
                fx = race_to_features(rc)
                if fx:
                    Xs.append(fx[0]); ws.append(fx[1]); got += 1
        if got:
            used_days += 1
    return Xs, ws, used_days

def train_conditional_logit(X, winners, iters=300, lr=0.4):
    X = np.asarray(X, float); winners = np.asarray(winners, int)
    R = len(X)
    flat = X.reshape(-1, X.shape[2])
    mu = flat.mean(0); sd = flat.std(0) + 1e-9
    Z = (X - mu) / sd
    cut = max(1, int(R * 0.8))
    beta = np.zeros(X.shape[2])
    for _ in range(iters):
        s = Z[:cut] @ beta; s -= s.max(1, keepdims=True)
        p = np.exp(s); p /= p.sum(1, keepdims=True)
        chosen = Z[:cut][np.arange(cut), winners[:cut]]
        exp_feat = (p[:, :, None] * Z[:cut]).sum(1)
        beta += lr * (chosen - exp_feat).mean(0)
    def probs(Zt):
        s = Zt @ beta; s -= s.max(1, keepdims=True)
        p = np.exp(s); p /= p.sum(1, keepdims=True); return p
    pte = probs(Z[cut:]); wte = winners[cut:]
    top = pte.argmax(1); ptop = pte.max(1)
    acc = float((top == wte).mean()) if len(wte) else 0.0
    base = float((wte == 0).mean()) if len(wte) else 0.0
    cal = {}
    for b in range(3, 10):
        m = (ptop >= b / 10) & (ptop < (b + 1) / 10)
        if m.sum() >= 10:
            cal[f"{b*10}-{b*10+10}%"] = round(float((top[m] == wte[m]).mean() * 100), 1)
    # 推奨・見送りしきい値: 予測トップ確率が高いほど買い。実的中率60%超えの帯の下限を目安に。
    thr = None
    for b in range(3, 10):
        key = f"{b*10}-{b*10+10}%"
        if cal.get(key, 0) >= 60:
            thr = round(b / 10, 2); break
    return {
        "beta": [round(x, 4) for x in beta.tolist()],
        "mu": [round(x, 4) for x in mu.tolist()],
        "sd": [round(x, 4) for x in sd.tolist()],
        "feature_order": ["course_base", "class", "nat", "loc", "motor", "st", "ex"],
        "test_acc": round(acc * 100, 1),
        "baseline_course1": round(base * 100, 1),
        "n_races": int(R),
        "calibration": cal,
        "suggest_buy_threshold": thr,
    }

@app.get("/api/train")
def api_train(days: int = Query(84), step: int = Query(3), iters: int = Query(300)):
    t0 = time.time()
    Xs, ws, used_days = collect_dataset(days, step)
    if len(Xs) < 200:
        return JSONResponse({"ok": False, "error": f"データ不足（{len(Xs)}レース）。daysを増やすか開催日を確認。"}, status_code=422)
    res = train_conditional_logit(Xs, ws, iters=iters)
    res.update({"ok": True, "used_days": used_days, "seconds": round(time.time() - t0, 1)})
    return res

# ================= バックテスト（時系列分割・実結果/実配当で 見送り×点数 を総当たり）=================
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

def race_full(race):
    fx = race_to_features(race)
    if fx is None:
        return None
    X, winner = fx
    ex, tri = _payouts(race)
    if tri is None:
        return None
    try:
        order = [int(x) for x in tri[0].split("-")]
    except ValueError:
        return None
    if len(order) != 3:
        return None
    return {"X": X, "winner": winner, "order": order,
            "ex_amt": (ex[1] if ex else None), "tri_amt": tri[1]}

def collect_full(days, step):
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()
    recs = []
    for d in range(1, days + 1, max(1, step)):
        day = today - dt.timedelta(days=d)
        hd = day.strftime("%Y%m%d")
        try:
            data = fetch_openapi(hd)
        except Exception:
            continue
        progs = data.get("programs", data) if isinstance(data, dict) else data
        stadiums = progs.get("stadiums") if isinstance(progs, dict) else None
        if not stadiums:
            continue
        for s in _as_items(stadiums):
            for rc in _as_items(s.get("races") if isinstance(s, dict) else None):
                if not isinstance(rc, dict):
                    continue
                r = race_full(rc)
                if r:
                    r["day"] = d
                    recs.append(r)
    return recs

def api_backtest_core(days, step, iters):
    recs = collect_full(days, step)
    if len(recs) < 300:
        return {"ok": False, "error": f"データ不足({len(recs)})"}
    recs.sort(key=lambda r: -r["day"])   # 古い順
    n = len(recs); cut = int(n * 0.7)
    train, test = recs[:cut], recs[cut:]
    m = train_conditional_logit([r["X"] for r in train], [r["winner"] for r in train], iters=iters)
    beta = np.array(m["beta"]); mu = np.array(m["mu"]); sd = np.array(m["sd"])
    for r in test:
        Z = (np.asarray(r["X"], float) - mu) / sd
        sc = Z @ beta
        e = np.exp(sc - sc.max()); p = e / e.sum()
        axis = int(sc.argmax())
        r["axis"] = axis + 1; r["headP"] = float(p[axis])
        others = sorted([i for i in range(6) if i != axis], key=lambda i: -sc[i])
        r["secs"] = [i + 1 for i in others]
    THRS = [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    TICKETS = [("2連単1点", 2, 1), ("2連単2点", 2, 2),
               ("3連単F 軸-2-全", 3, 2), ("3連単F 軸-3-全", 3, 3), ("3連単1点", 3, 1)]
    N = len(test); results = []
    for thr in THRS:
        for name, kind, k in TICKETS:
            staked = returned = hits = nb = 0; profits = []
            for r in test:
                if r["headP"] < thr:
                    continue
                picks = r["secs"][:k]
                if kind == 2:
                    if r["ex_amt"] is None:
                        continue
                    cost = k * 100
                    win = (r["order"][0] == r["axis"] and r["order"][1] in picks)
                    pay = r["ex_amt"] if win else 0
                elif name == "3連単1点":
                    second = r["secs"][0]; third = r["secs"][1]
                    cost = 100
                    win = (r["order"] == [r["axis"], second, third])
                    pay = r["tri_amt"] if win else 0
                else:
                    cost = k * 4 * 100
                    win = (r["order"][0] == r["axis"] and r["order"][1] in picks)
                    pay = r["tri_amt"] if win else 0
                staked += cost; returned += pay; nb += 1
                if win:
                    hits += 1
                profits.append(pay - cost)
            if nb < 20:
                continue
            cum = peak = dd = 0
            for pf in profits:
                cum += pf; peak = max(peak, cum); dd = min(dd, cum - peak)
            arr = np.array(profits, float)
            results.append({
                "見送り条件": (f"1着確率>={int(thr*100)}%" if thr > 0 else "全レース購入"),
                "点数": name, "購入数": nb, "見送り率": round((N - nb) / N * 100, 1),
                "的中率": round(hits / nb * 100, 1), "回収率": round(returned / staked * 100, 1) if staked else 0,
                "総利益": int(returned - staked), "最大DD": int(dd),
                "シャープ": round(float(arr.mean() / (arr.std() + 1e-9)), 3),
            })
    results.sort(key=lambda x: -x["回収率"])
    return {"ok": True, "n_train": len(train), "n_test": N,
            "model_test_acc": m["test_acc"], "baseline_course1": m["baseline_course1"],
            "top": results[:12], "strategies_tested": len(results)}

@app.get("/api/backtest")
def api_backtest(days: int = Query(180), step: int = Query(6), iters: int = Query(250)):
    t0 = time.time()
    res = api_backtest_core(days, step, iters)
    if isinstance(res, dict) and res.get("ok"):
        res["seconds"] = round(time.time() - t0, 1)
    return JSONResponse(res)

# ================= 高オッズEV 前向き検証（ゲート付き）2連単 =================
# 依存追加なし(numpyのみ)。アプリ内蔵の条件付きロジット(焼き込み係数)をサーバ側で再現し、
# 「信頼できるレースだけ(自信度×本命の抜け)通すゲート → 高オッズ×EVプラスの2連単」を推奨・記録・精算する。
import csv as _csv
_LB = [0.6958, 0.1224, 0.728, 0.0152, 0.1583, -0.0137, -0.421]
_LM = [0.1668, 0.5436, 5.2401, 4.6892, 32.7627, 0.088, 6.826]
_LSD = [0.1764, 0.3059, 1.3348, 2.1123, 11.0123, 0.1053, 0.1136]
DATA_DIR = os.environ.get("APP_DATA_DIR", "/app/data" if os.path.isdir("/app/data") else "./data")
LEDGER = os.path.join(DATA_DIR, "ho_ledger.csv")
LCOLS = ["ts", "date", "jcd", "rno", "combo", "P", "odds", "market_p", "EV",
         "stake", "status", "payout", "ret", "hit", "star", "women"]


def predict_power(boats):
    """各艇のP(1着)。フロントJSと同一の焼き込み条件付きロジット。"""
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
    """レース自信度(3連単最尤P相当)と本命の抜け。ゲート判定用。"""
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
    # focus: 買い目を絞る組の集合（例 {"1-4","1-5"}）。None なら全2連単が対象。
    power = predict_power(boats)
    conf, gap = _conf_gap(power)
    if conf < min_conf or gap < min_gap:
        return {"decision": "見送り", "reason": f"レースゲート未通過（自信度{conf*100:.0f}%/抜け{gap*100:.0f}%）",
                "conf": round(conf, 4), "gap": round(gap, 4), "picks": []}
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
        cand.append({"combo": f"{i}-{j}", "P": round(p, 4), "odds": o,
                     "market_p": round(1 / o, 4), "EV": round(ev, 3)})
    cand.sort(key=lambda c: c["P"], reverse=True)
    picks = cand[:max_n]
    if not picks:
        return {"decision": "見送り", "reason": "高オッズEV＋の買い目なし",
                "conf": round(conf, 4), "gap": round(gap, 4), "picks": []}
    return {"decision": "買い", "reason": "ゲート通過・高オッズEV＋",
            "conf": round(conf, 4), "gap": round(gap, 4), "picks": picks}


def _append_ledger(hd, jcd, rno, picks, stake, star=0, women=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    exists = os.path.exists(LEDGER)
    seen = set()
    if exists:
        with open(LEDGER, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                seen.add((row["date"], row["jcd"], row["rno"], row["combo"]))
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=LCOLS)
        if not exists:
            w.writeheader()
        for p in picks:
            if (str(hd), str(jcd), str(rno), p["combo"]) in seen:
                continue
            w.writerow({"ts": now, "date": hd, "jcd": jcd, "rno": rno, "combo": p["combo"],
                        "P": p["P"], "odds": p["odds"], "market_p": p["market_p"], "EV": p["EV"],
                        "stake": stake, "status": "pending", "payout": "", "ret": "", "hit": "",
                        "star": star, "women": int(bool(women))})


def _ledger_stats(rows):
    s = [r for r in rows if r["status"] == "settled"]
    stk = sum(float(r["stake"]) for r in s)
    ret = sum(float(r["ret"] or 0) for r in s)
    h = sum(int(r["hit"] or 0) for r in s)
    return {"settled": len(s), "pending": sum(1 for r in rows if r["status"] == "pending"),
            "hits": h, "staked": round(stk), "returned": round(ret), "net": round(ret - stk),
            "roi_pct": round(ret / stk * 100, 1) if stk else 0}


@app.get("/api/highodds")
def api_highodds(jcd: int = Query(...), rno: int = Query(...), hd: str = Query(...),
                 odds_min: float = Query(20.0), ev_min: float = Query(1.20), p_floor: float = Query(0.05),
                 min_conf: float = Query(0.12), min_gap: float = Query(0.30), max_n: int = Query(2),
                 stake: float = Query(100.0), log: bool = Query(False),
                 star: int = Query(0), women: bool = Query(False),
                 focus: str = Query("1-4,1-5")):
    try:
        data = fetch_openapi(hd)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"openapi取得失敗: {e}"}, status_code=502)
    race = find_race(data, jcd, rno)
    if race is None:
        return JSONResponse({"ok": False, "error": "該当レースが見つかりません"}, status_code=404)
    boats = extract_boats(race)
    odds_map = fetch_exacta_odds(jcd, rno, hd)
    focus_set = {c.strip() for c in (focus or "").split(",") if c.strip()} or None
    res = highodds_pick(boats, odds_map, odds_min, ev_min, p_floor, min_conf, min_gap, max_n, focus_set)
    res.update({"ok": True, "jcd": jcd, "rno": rno, "hd": hd,
                "odds_status": "ライブ（boatrace.jp）" if odds_map else "オッズ未取得（見送り扱い推奨）"})
    if log and res["decision"] == "買い" and odds_map:
        _append_ledger(hd, jcd, rno, res["picks"], stake, star, women)
        res["logged"] = len(res["picks"])
    return res


@app.get("/api/highodds/settle")
def api_highodds_settle(hd: str = Query(...)):
    if not os.path.exists(LEDGER):
        return {"ok": False, "error": "台帳がありません"}
    try:
        data = fetch_openapi(hd)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"openapi取得失敗: {e}"}, status_code=502)
    with open(LEDGER, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    upd = 0
    for row in rows:
        if row["status"] != "pending" or row["date"] != hd:
            continue
        race = find_race(data, int(row["jcd"]), int(row["rno"]))
        if race is None:
            continue
        ex, tri = _payouts(race)
        if ex is None:
            continue
        wc, amt = ex
        hit = (row["combo"] == wc)
        row["status"] = "settled"; row["payout"] = amt if hit else 0
        row["ret"] = round(float(row["stake"]) * amt / 100.0, 1) if hit else 0.0
        row["hit"] = int(hit); upd += 1
    with open(LEDGER, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=LCOLS); w.writeheader(); w.writerows(rows)
    return {"ok": True, "settled_now": upd, **_ledger_stats(rows)}


@app.get("/api/highodds/ledger")
def api_highodds_ledger():
    if not os.path.exists(LEDGER):
        return {"ok": True, "rows": [], "settled": 0, "pending": 0, "hits": 0,
                "staked": 0, "returned": 0, "net": 0, "roi_pct": 0}
    with open(LEDGER, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    return {"ok": True, "rows": rows[-200:], **_ledger_stats(rows)}


# 自動: その日の対象レースを全部 判定→(結果があれば即)精算 して台帳に追記
TARGET_VENUES = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 21, 22, 24]  # 平和島劇場で買える約13場


@app.get("/api/highodds/autorun")
def api_highodds_autorun(hd: str = Query(None),
                         odds_min: float = Query(20.0), ev_min: float = Query(1.20),
                         p_floor: float = Query(0.05), min_conf: float = Query(0.12),
                         min_gap: float = Query(0.30), max_n: int = Query(2),
                         stake: float = Query(100.0), focus: str = Query("1-4,1-5"),
                         venue: int = Query(None)):
    if not hd:
        hd = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d")
    venues = [venue] if venue else TARGET_VENUES   # venue指定で1場だけ処理(タイムアウト回避)
    try:
        data = fetch_openapi(hd)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"openapi: {e}"}, status_code=502)
    focus_set = {c.strip() for c in focus.split(",") if c.strip()} or None
    seen = set()
    if os.path.exists(LEDGER):
        with open(LEDGER, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                seen.add((str(row["date"]), str(row["jcd"]), str(row["rno"]), str(row["combo"])))
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(DATA_DIR, exist_ok=True)
    exists = os.path.exists(LEDGER)
    races = gated = logged = 0
    today_rows = []          # ← この実行で記録した行(GitHub側で永続台帳に追記する用)
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=LCOLS)
        if not exists:
            w.writeheader()
        for jcd in venues:
            for rno in range(1, 13):
                race = find_race(data, jcd, rno)
                if race is None:
                    continue
                ex, tri = _payouts(race)      # 確定結果(2連単勝ち組と払戻)。無ければ未確定→skip
                if ex is None:
                    continue
                races += 1
                boats = extract_boats(race)
                power = predict_power(boats)
                conf, gap = _conf_gap(power)
                if conf < min_conf or gap < min_gap:
                    continue
                gated += 1
                odds_map = fetch_exacta_odds(jcd, rno, hd)
                if not odds_map:
                    continue
                res = highodds_pick(boats, odds_map, odds_min, ev_min, p_floor,
                                    min_conf, min_gap, max_n, focus_set)
                if res["decision"] != "買い":
                    continue
                wc, amt = ex
                for p in res["picks"]:
                    key = (str(hd), str(jcd), str(rno), str(p["combo"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    hit = (p["combo"] == wc)
                    ret = round(stake * amt / 100.0, 1) if hit else 0.0
                    rowd = {"ts": now, "date": hd, "jcd": jcd, "rno": rno, "combo": p["combo"],
                            "P": p["P"], "odds": p["odds"], "market_p": p["market_p"], "EV": p["EV"],
                            "stake": stake, "status": "settled", "payout": (amt if hit else 0),
                            "ret": ret, "hit": int(hit), "star": 0, "women": 0}
                    w.writerow(rowd)
                    today_rows.append(rowd)
                    logged += 1
                time.sleep(0.2)
    with open(LEDGER, newline="", encoding="utf-8") as f:
        allrows = list(_csv.DictReader(f))
    return {"ok": True, "hd": hd, "確定レース": races, "ゲート通過": gated, "今回記録": logged,
            "records": today_rows, "cols": LCOLS, **_ledger_stats(allrows)}


_HIGHODDS_HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>高オッズEV 前向き検証</title>
<style>
:root{--bg:#eef1f4;--card:#fff;--ink:#12263a;--sub:#5b6b7d;--line:#dbe2ea;--buy:#1c7a38;--skip:#8a97a5;--accent:#3a5bd0;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"Hiragino Kaku Gothic ProN",sans-serif;padding:14px;}
h1{font-size:20px;margin:2px 0 2px}.sub{color:var(--sub);font-size:13px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.warn{background:#fff9ec;border:1px solid #f0e0b8;border-radius:10px;padding:10px 12px;font-size:12.5px;color:#7a5b12;margin-bottom:12px}
label{font-size:12px;color:var(--sub);display:block;margin:6px 0 2px}
select,input{padding:8px;border:1px solid var(--line);border-radius:8px;font-size:15px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:15px;font-weight:700;cursor:pointer}
button.sec{background:#e7edf5;color:var(--ink)}
button.rec{background:var(--buy)}
.verdict{font-size:22px;font-weight:800;margin:6px 0}
.badge{display:inline-block;padding:3px 12px;border-radius:20px;color:#fff;font-size:15px;font-weight:800}
.b-buy{background:var(--buy)}.b-skip{background:var(--skip)}
table{width:100%;border-collapse:collapse;margin-top:8px}th,td{border-bottom:1px solid var(--line);padding:7px 6px;font-size:13px;text-align:right}
th:first-child,td:first-child{text-align:left}
.stat{display:flex;gap:14px;flex-wrap:wrap}.stat div{font-size:13px;color:var(--sub)}.stat b{display:block;font-size:20px;color:var(--ink)}
.roi{color:var(--buy)}.roi.neg{color:#c0392b}
small{color:var(--sub)}
.detail{font-size:12px;color:var(--sub);margin-top:6px}
details summary{cursor:pointer;color:var(--accent);font-size:13px}
</style></head><body>
<h1>&#127937; 高オッズEV 前向き検証</h1>
<div class="sub">信頼できるレースだけ通し（自信度×本命の抜け）、高オッズ×EVプラスの2連単を"紙トレ"で記録→通算回収率で検証する道具。</div>
<div class="warn">&#9888; これは利益を保証しません。控除率25%は不変。買い/見送りを一貫させ、結果を記録して<b>通算回収率</b>が本当にプラスに乗るかを、実弾ゼロで数週間かけて検証するための道具です。オッズは締切直前が最終値。</div>

<div class="card">
  <b>&#9312; レースを選んで判定</b>
  <div class="row" style="margin-top:8px">
    <div><label>競艇場</label><select id="jcd"></select></div>
    <div><label>レース</label><select id="rno"></select></div>
    <div><label>日付</label><input type="date" id="hd"></div>
    <div><label>梅吉★</label><select id="star"><option value="0">-</option><option value="1">★1</option><option value="2">★2</option><option value="3">★3</option></select></div>
    <div><label>女子戦</label><input type="checkbox" id="women" style="width:20px;height:20px;margin-top:8px"></div>
    <div><button onclick="judge(false)">&#9654; 判定する</button></div>
  </div>
  <div class="detail">※ 梅吉★（あなたが umepyon.com を見て入力）と女子戦フラグは、買い目と一緒に台帳に記録され、後で「★別・女子別」の通算回収率で検証できます。</div>
  <details style="margin-top:10px"><summary>詳細設定（しきい値）</summary>
    <div class="row" style="margin-top:8px">
      <div><label>オッズ下限(倍)</label><input type="number" id="odds_min" value="20" style="width:80px"></div>
      <div><label>EV下限</label><input type="number" id="ev_min" value="1.20" step="0.05" style="width:80px"></div>
      <div><label>自信度下限</label><input type="number" id="min_conf" value="0.12" step="0.01" style="width:80px"></div>
      <div><label>本命抜け下限</label><input type="number" id="min_gap" value="0.30" step="0.05" style="width:80px"></div>
      <div><label>最大点数</label><input type="number" id="max_n" value="2" style="width:70px"></div>
      <div><label>1点賭け金(円)</label><input type="number" id="stake" value="100" style="width:90px"></div>
      <div><label>買い目を絞る（カンマ区切り・空=全部）</label><input type="text" id="focus" value="1-4,1-5" style="width:150px"></div>
    </div>
  </details>
</div>

<div class="card" id="resultCard" style="display:none">
  <div id="verdict"></div>
  <div class="detail" id="reason"></div>
  <div id="picks"></div>
  <div style="margin-top:10px" id="recWrap"></div>
</div>

<div class="card">
  <div class="row" style="justify-content:space-between">
    <b>&#9313; 通算成績（紙トレ台帳）</b>
    <div class="row">
      <input type="date" id="settleDate">
      <button class="sec" onclick="settle()">この日を精算</button>
      <button class="sec" onclick="loadLedger()">更新</button>
    </div>
  </div>
  <div class="stat" id="stats" style="margin-top:10px"></div>
  <div id="ledger"></div>
</div>

<script>
const VENUES={1:"桐生",2:"戸田",3:"江戸川",4:"平和島",5:"多摩川",6:"浜名湖",7:"蒲郡",8:"常滑",9:"津",10:"三国",11:"びわこ",12:"住之江",13:"尼崎",14:"鳴門",15:"丸亀",16:"児島",17:"宮島",18:"徳山",19:"下関",20:"若松",21:"芦屋",22:"福岡",23:"唐津",24:"大村"};
const $=id=>document.getElementById(id);
for(const k in VENUES) $("jcd").insertAdjacentHTML("beforeend",`<option value="${k}">${String(k).padStart(2,'0')} ${VENUES[k]}</option>`);
for(let i=1;i<=12;i++) $("rno").insertAdjacentHTML("beforeend",`<option value="${i}">${i}R</option>`);
const jst=new Date(Date.now()+ (new Date().getTimezoneOffset()+540)*60000);
$("hd").value=jst.toISOString().slice(0,10); $("settleDate").value=$("hd").value;
$("jcd").value="22";

function hdc(v){return v.replace(/-/g,'');}
async function judge(log){
  const q=new URLSearchParams({jcd:$("jcd").value,rno:$("rno").value,hd:hdc($("hd").value),
    odds_min:$("odds_min").value,ev_min:$("ev_min").value,min_conf:$("min_conf").value,
    min_gap:$("min_gap").value,max_n:$("max_n").value,stake:$("stake").value,log:log?"true":"false",
    star:$("star").value,women:$("women").checked?"true":"false",focus:$("focus").value});
  $("resultCard").style.display="block";
  $("verdict").innerHTML="判定中…"; $("reason").innerHTML=""; $("picks").innerHTML=""; $("recWrap").innerHTML="";
  try{
    const r=await fetch("/api/highodds?"+q.toString()); const j=await r.json();
    if(!j.ok){$("verdict").innerHTML="取得失敗"; $("reason").innerHTML=j.error||""; return;}
    const buy=j.decision==="買い";
    $("verdict").innerHTML=`<span class="badge ${buy?'b-buy':'b-skip'}">${j.decision}</span> <small>${VENUES[$("jcd").value]} ${$("rno").value}R</small>`;
    $("reason").innerHTML=`理由: ${j.reason}　/　自信度 ${(j.conf*100).toFixed(0)}%　本命の抜け ${(j.gap*100).toFixed(0)}%　/　オッズ:${j.odds_status}`;
    if(j.picks&&j.picks.length){
      let h='<table><tr><th>買い目(2連単)</th><th>AI予測P</th><th>市場</th><th>オッズ</th><th>EV</th><th>賭け金</th></tr>';
      for(const p of j.picks) h+=`<tr><td><b>${p.combo}</b></td><td>${(p.P*100).toFixed(1)}%</td><td>${(p.market_p*100).toFixed(1)}%</td><td>${p.odds}倍</td><td style="color:var(--buy);font-weight:700">${p.EV}</td><td>${$("stake").value}円</td></tr>`;
      $("picks").innerHTML=h+'</table>';
      if(log){$("recWrap").innerHTML=`<b style="color:var(--buy)">&#10003; ${j.logged}件を台帳に記録しました。</b>`; loadLedger();}
      else $("recWrap").innerHTML=`<button class="rec" onclick="judge(true)">&#128221; この買い目を台帳に記録する</button>`;
    }else{ $("picks").innerHTML='<div class="detail">買い目なし（見送り）。</div>'; }
  }catch(e){ $("verdict").innerHTML="通信エラー"; $("reason").innerHTML=String(e); }
}
async function loadLedger(){
  try{
    const j=await (await fetch("/api/highodds/ledger")).json();
    const neg=j.net<0?"neg":"";
    $("stats").innerHTML=`<div>精算済<b>${j.settled}</b></div><div>未精算<b>${j.pending}</b></div><div>的中<b>${j.hits}</b></div>
      <div>投資<b>${(j.staked||0).toLocaleString()}円</b></div><div>払戻<b>${(j.returned||0).toLocaleString()}円</b></div>
      <div>純益<b class="roi ${neg}">${(j.net||0).toLocaleString()}円</b></div><div>回収率<b class="roi ${neg}">${j.roi_pct}%</b></div>`;
    const settled=(j.rows||[]).filter(r=>r.status=="settled");
    function grp(keyfn,lab){const g={};for(const r of settled){const k=keyfn(r);(g[k]=g[k]||{s:0,ret:0,n:0,h:0});g[k].s+=+r.stake;g[k].ret+=+(r.ret||0);g[k].n++;g[k].h+=+(r.hit||0);}return Object.keys(g).sort().map(k=>`${lab(k)} ${g[k].n}件 的中${g[k].h} ROI ${g[k].s?Math.round(g[k].ret/g[k].s*100):0}%`).join('　/　')||'—';}
    const hyp = settled.length? `<div class="detail" style="margin:8px 0;line-height:1.7"><b>&#128300; 仮説チェック（精算済み）</b><br>★別: ${grp(r=>r.star||"0",k=>k=="0"?"★なし":"★"+k)}<br>女子: ${grp(r=>r.women=="1"?"女子戦":"一般",k=>k)}</div>` : '';
    const rows=(j.rows||[]).slice().reverse().slice(0,30);
    let tbl;
    if(rows.length){
      let h='<table><tr><th>日付</th><th>場R</th><th>買い目</th><th>オッズ</th><th>EV</th><th>★</th><th>女</th><th>状態</th><th>払戻</th></tr>';
      for(const r of rows) h+=`<tr><td>${r.date}</td><td>${r.jcd}-${r.rno}</td><td>${r.combo}</td><td>${r.odds}</td><td>${r.EV}</td><td>${r.star&&r.star!="0"?"★"+r.star:""}</td><td>${r.women=="1"?"女":""}</td><td>${r.status=="settled"?(r.hit=="1"?"的中":"外れ"):"未"}</td><td>${r.ret||""}</td></tr>`;
      tbl=h+'</table>';
    }else tbl='<div class="detail">まだ記録がありません。判定→記録で貯まります。</div>';
    $("ledger").innerHTML=hyp+tbl;
  }catch(e){ $("ledger").innerHTML='<div class="detail">台帳取得エラー: '+e+'</div>'; }
}
async function settle(){
  const hd=hdc($("settleDate").value);
  const j=await (await fetch("/api/highodds/settle?hd="+hd)).json();
  if(j.ok) alert(`精算しました（今回${j.settled_now}件）。通算 ROI ${j.roi_pct}%`); else alert(j.error||"精算失敗");
  loadLedger();
}
loadLedger();
</script></body></html>"""


@app.get("/highodds", response_class=HTMLResponse)
def highodds_page():
    return _HIGHODDS_HTML


# ---- フロント（HTMLを埋め込み：別ファイル不要で確実に配信）----
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ボートレース 妙味スコア予想（自動更新版）</title>
<style>
  *{box-sizing:border-box;}
  body{font-family:"Hiragino Kaku Gothic ProN","Yu Gothic",Meiryo,Arial,sans-serif;margin:0;background:#0f1720;color:#1a2330;}
  .wrap{max-width:1080px;margin:0 auto;padding:0 16px 60px;}
  header{background:linear-gradient(135deg,#12263a,#1f3d5c);color:#fff;padding:20px 24px;}
  header h1{margin:0 0 4px;font-size:20px;} header p{margin:0;font-size:13px;color:#b9cbe0;}
  .card{background:#fff;border-radius:12px;padding:16px 20px;margin:14px 0;box-shadow:0 4px 18px rgba(0,0,0,.25);}
  .warn{background:#fff8e6;border-left:5px solid #e0a800;font-size:12.5px;line-height:1.6;color:#5a4600;}
  h2{font-size:15px;margin:2px 0 12px;color:#12263a;border-bottom:2px solid #e5ebf1;padding-bottom:6px;}
  table{border-collapse:collapse;width:100%;font-size:13px;} th,td{border:1px solid #dbe2ea;padding:5px 6px;text-align:center;}
  th{background:#eef3f8;font-weight:700;color:#33475b;} td input,td select{width:100%;border:1px solid #cfd8e3;border-radius:5px;padding:5px 4px;font-size:13px;text-align:center;background:#fbfdff;}
  td.frame{font-weight:800;}
  .chip.b1,td.frame.b1{background:#ffffff;color:#222;border:1.5px solid #b7bec6;}
  .chip.b2,td.frame.b2{background:#1a1a1a;color:#fff;}
  .chip.b3,td.frame.b3{background:#e6303a;color:#fff;}
  .chip.b4,td.frame.b4{background:#1f6fd0;color:#fff;}
  .chip.b5,td.frame.b5{background:#f4c430;color:#333;}
  .chip.b6,td.frame.b6{background:#3aa655;color:#fff;}
  .oddscol{background:#fff9ec!important;}
  button{background:#1f6fd0;color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:14px;font-weight:700;cursor:pointer;}
  button:hover{background:#175bb0;} .btn2{background:#5a6b7d;font-size:13px;padding:8px 14px;}
  .fetchbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
  .fetchbar select,.fetchbar input{padding:7px 8px;border:1px solid #cfd8e3;border-radius:6px;font-size:14px;}
  .status{font-size:12.5px;color:#33475b;background:#eef3f8;padding:6px 12px;border-radius:20px;}
  .status b{color:#12263a;} .live{color:#1c7a38;font-weight:800;} .est{color:#9a6a10;font-weight:800;}
  .controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:10px;}
  .dial,.axisbox{font-size:13px;color:#33475b;background:#f4f7fa;padding:8px 12px;border-radius:8px;}
  .conf{font-size:14px;font-weight:800;padding:11px 14px;border-radius:9px;margin:2px 0 12px;}
  .cHi{background:#e3f6e8;color:#166a30;border:2px solid #3aa655;} .cMid{background:#fdf3e0;color:#8a5a00;border:2px solid #e0a800;} .cLow{background:#fdecec;color:#a02525;border:2px solid #e07272;}
  .rrow{display:grid;grid-template-columns:44px 1fr 76px;gap:10px;align-items:center;padding:9px 0;border-bottom:1px solid #eef2f6;}
  .chip{width:30px;height:30px;border-radius:6px;color:#fff;font-weight:800;display:flex;align-items:center;justify-content:center;}
  .barline{display:flex;align-items:center;gap:8px;margin:2px 0;font-size:11.5px;color:#556;} .barlabel{width:92px;text-align:right;color:#667;}
  .bar{height:12px;border-radius:3px;} .bJ{background:#1f6fd0;}.bM{background:#9aa7b4;}.bY{background:#3aa655;}.bYn{background:#e64c4c;}.bP{background:#7a5cd0;}
  .myo{font-size:19px;font-weight:800;text-align:right;}
  .flag{display:inline-block;font-size:11px;font-weight:800;padding:2px 8px;border-radius:20px;}
  .fAxis{background:#e7eefc;color:#1b4a9c;border:1px solid #9cb8e8;}.fHon{background:#e3f6e8;color:#1c7a38;border:1px solid #7fce9a;}.fSpec{background:#fdf3e0;color:#9a6a10;border:1px solid #e6c47a;}.fSkip{background:#f0f3f6;color:#7a8794;border:1px solid #cdd6df;}
  .verdict{font-size:17px;font-weight:800;padding:14px 16px;border-radius:10px;margin-top:6px;background:#e3f6e8;color:#166a30;border:2px solid #3aa655;}
  .roiNote{font-size:12.5px;margin-top:8px;padding:8px 12px;border-radius:8px;background:#f4f7fa;color:#455;line-height:1.6;}
  .expl{font-size:13px;line-height:1.7;color:#37485a;margin-top:10px;} .small{font-size:11.5px;color:#8794a2;}
</style>
</head>
<body>
<header>
  <h1>🚤 妙味スコア予想（自動更新版）</h1>
  <p>場・R・日付を選ぶと出走表と直前情報を自動取得。オッズは締切まで自動更新（取得できない時は推定にフォールバック）</p>
</header>
<div class="wrap">

  <div class="card warn">
    ⚠ 控除率25%は不変で、これは勝ちを保証しません。買い/見送りを一貫させ、結果を記録して<b>通算回収率</b>で検証する道具です。オッズは目安（取得タイミングで変動）。締切直前に最終確認を。
  </div>

  <div class="card">
    <h2>① レースを選んで取得</h2>
    <div class="fetchbar">
      <select id="jcd"></select>
      <select id="rno"></select>
      <input type="date" id="hd">
      <button onclick="fetchRace()">🔄 データ取得</button>
      <label class="status"><input type="checkbox" id="auto" onchange="toggleAuto()"> 自動更新(30秒)</label>
      <span class="status" id="status">未取得</span>
    </div>
    <p class="small">日付は開催日を選択。取得後は各セルを手で微調整もできます。オッズ列に値が入れば「軸→その艇」の実オッズで妙味を計算します。</p>
  </div>

  <div class="card">
    <h2>② 出走データ</h2>
    <table id="inputTable"><thead><tr>
      <th>艇</th><th>進入</th><th>級別</th><th>全国<br>勝率</th><th>当地<br>勝率</th><th>モーター<br>2連率</th><th>展示<br>タイム</th><th>展示<br>ST</th>
      <th class="oddscol">2連単<br>オッズ</th>
    </tr></thead><tbody id="tbody"></tbody></table>
    <div class="controls">
      <button onclick="calc()">▶ 計算する</button>
      <span class="axisbox">👑 1着軸：
        <select id="axisSel" onchange="if(lastData)calc()">
          <option value="auto">自動判定</option>
          <option value="1">1号艇</option><option value="2">2号艇</option><option value="3">3号艇</option>
          <option value="4">4号艇</option><option value="5">5号艇</option><option value="6">6号艇</option>
        </select></span>
      <span class="dial">🎯 カバー範囲：<input type="range" id="width" min="1" max="5" step="1" value="2" oninput="wv.textContent=this.value; if(lastData)calc()"><b id="wv">2</b> 艇</span>
    </div>
  </div>

  <div class="card">
    <h2>④ 資金管理・実戦判定（バックテスト最適ルール）</h2>
    <div class="controls" style="gap:16px;">
      <span class="axisbox">💰 総資金：<input id="bank" type="number" value="30000" style="width:100px;padding:5px;border:1px solid #cfd8e3;border-radius:5px;"> 円</span>
      <span class="axisbox">賭け方：
        <select id="stakeMethod" onchange="if(lastData)calc()" style="padding:5px;border-radius:5px;border:1px solid #cfd8e3;">
          <option value="fixed">固定額</option><option value="pct">資金の割合</option>
        </select>
        <input id="stakeFixed" type="number" value="1000" style="width:75px;padding:5px;border:1px solid #cfd8e3;border-radius:5px;" onchange="if(lastData)calc()"> 円 ／
        <input id="stakePct" type="number" value="1" step="0.5" style="width:50px;padding:5px;border:1px solid #cfd8e3;border-radius:5px;" onchange="if(lastData)calc()"> %
      </span>
      <span class="axisbox">買い条件：1着確率 <input id="buyThr" type="number" value="80" style="width:50px;padding:5px;border:1px solid #cfd8e3;border-radius:5px;" onchange="if(lastData)calc()"> % 以上</span>
      <span class="axisbox">1日の損切り：<input id="stopLoss" type="number" value="10000" style="width:80px;padding:5px;border:1px solid #cfd8e3;border-radius:5px;" onchange="renderSess()"> 円</span>
    </div>
    <div style="margin-top:10px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
      <span class="status">現在資金：<b id="curBank">—</b> 円</span>
      <span class="status">本日収支：<b id="sessPnl">±0</b> 円</span>
      <button class="btn2" style="background:#2e7d46;" onclick="logResult(true)">的中(配当入力)</button>
      <button class="btn2" style="background:#a24b4b;" onclick="logResult(false)">外れ</button>
      <button class="btn2" onclick="resetSession()">資金リセット</button>
      <span class="small" id="stopMsg"></span>
    </div>
    <p class="small">資金管理は「勝ち」を作れません（回収率94.5%＝長期は負け）。役割は破産防止・損失速度の抑制・1日の損失上限。的中/外れを押すと現在資金が更新されます（この端末に保存）。</p>
  </div>

  <div class="card" id="resultCard" style="display:none;">
    <h2>③ 判定結果</h2>
    <div id="conf"></div><div id="verdict"></div><div id="roiNote" class="roiNote"></div>
    <div id="rows" style="margin-top:14px;"></div><div class="expl" id="expl"></div>
  </div>
</div>

<script>
const VENUES={1:'桐生',2:'戸田',3:'江戸川',4:'平和島',5:'多摩川',6:'浜名湖',7:'蒲郡',8:'常滑',9:'津',10:'三国',11:'びわこ',12:'住之江',13:'尼崎',14:'鳴門',15:'丸亀',16:'児島',17:'宮島',18:'徳山',19:'下関',20:'若松',21:'芦屋',22:'福岡',23:'唐津',24:'大村'};
const CC=['b1','b2','b3','b4','b5','b6'];
const CLASS_SCORE={'A1':1.0,'A2':0.66,'B1':0.33,'B2':0.0};
const COURSE_BASE={1:0.55,2:0.15,3:0.12,4:0.10,5:0.06,6:0.02};
let lastData=null, autoTimer=null, lastOddsAll=null;

function initSelectors(){
  const j=document.getElementById('jcd');
  for(const[k,v]of Object.entries(VENUES)) j.insertAdjacentHTML('beforeend',`<option value="${k}">${String(k).padStart(2,'0')} ${v}</option>`);
  j.value='19';
  const r=document.getElementById('rno');
  for(let i=1;i<=12;i++) r.insertAdjacentHTML('beforeend',`<option value="${i}">${i}R</option>`);
  r.value='5';
  const d=new Date(); const jst=new Date(d.getTime()+ (d.getTimezoneOffset()+540)*60000);
  document.getElementById('hd').value=jst.toISOString().slice(0,10);
}
function hdCompact(){return document.getElementById('hd').value.replace(/-/g,'');}

function blankRow(c){return{course:c,cls:'B1',nat:'',loc:'',motor:'',ex:'',st:'',odds:''};}
function buildRows(data){const tb=document.getElementById('tbody');tb.innerHTML='';
 for(let i=0;i<6;i++){const d=data[i];const tr=document.createElement('tr');
  tr.innerHTML=`<td class="frame ${CC[i]}">${i+1}</td>
   <td><input type="number" min="1" max="6" value="${d.course}" data-k="course" data-i="${i}"></td>
   <td><select data-k="cls" data-i="${i}">${['A1','A2','B1','B2'].map(c=>`<option ${c===d.cls?'selected':''}>${c}</option>`).join('')}</select></td>
   <td><input type="number" step="0.01" value="${d.nat}" data-k="nat" data-i="${i}"></td>
   <td><input type="number" step="0.01" value="${d.loc}" data-k="loc" data-i="${i}"></td>
   <td><input type="number" step="0.01" value="${d.motor}" data-k="motor" data-i="${i}"></td>
   <td><input type="number" step="0.01" value="${d.ex}" data-k="ex" data-i="${i}"></td>
   <td><input type="number" step="0.01" value="${d.st}" data-k="st" data-i="${i}"></td>
   <td class="oddscol"><input type="number" step="0.1" value="${d.odds}" placeholder="—" data-k="odds" data-i="${i}"></td>`;
  tb.appendChild(tr);}}
function readInputs(){const data=[...Array(6)].map(()=>({}));
 document.querySelectorAll('#tbody [data-k]').forEach(el=>{const i=+el.dataset.i,k=el.dataset.k;
  if(k==='cls')data[i][k]=el.value;else if(k==='odds')data[i][k]=el.value===''?null:parseFloat(el.value);
  else data[i][k]=parseFloat(el.value);});return data;}

async function fetchRace(){
  const jcd=document.getElementById('jcd').value, rno=document.getElementById('rno').value, hd=hdCompact();
  const st=document.getElementById('status'); st.innerHTML='取得中…';
  try{
    const res=await fetch(`/api/race?jcd=${jcd}&rno=${rno}&hd=${hd}`);
    const j=await res.json();
    if(!j.ok){st.innerHTML='⚠ '+ (j.error||'取得失敗'); return;}
    lastOddsAll=j.odds_all||null;
    buildRows(j.boats.map(b=>({course:b.course, cls:b.cls||'B1',
      nat:b.nat??'', loc:b.loc??'', motor:b.motor??'', ex:b.ex??'', st:b.st??'', odds:b.odds??''})));
    const cls = j.odds_status.startsWith('ライブ')?'live':'est';
    st.innerHTML=`<b>${VENUES[jcd]} ${rno}R</b> 更新 ${j.updated_at} ／ オッズ:<span class="${cls}">${j.odds_status}</span>`;
    calc();
  }catch(e){ st.innerHTML='⚠ 通信エラー: '+e; }
}
function toggleAuto(){
  if(document.getElementById('auto').checked){ fetchRace(); autoTimer=setInterval(fetchRace,30000); }
  else { clearInterval(autoTimer); autoTimer=null; }
}

function norm(a,inv){const v=a.filter(x=>x!=null&&!isNaN(x));const mn=Math.min(...v),mx=Math.max(...v);
 if(!v.length||mx===mn)return a.map(()=>0.5);return a.map(x=>(x==null||isNaN(x))?0.5:(inv?(mx-x)/(mx-mn):(x-mn)/(mx-mn)));}

function calc(){
 const data=readInputs();lastData=data;
 const motorN=norm(data.map(d=>d.motor)),stN=norm(data.map(d=>d.st),true),exN=norm(data.map(d=>d.ex),true),
       locN=norm(data.map(d=>d.loc)),natN=norm(data.map(d=>d.nat));
 const clsN=data.map(d=>CLASS_SCORE[d.cls]??0.33),inner=data.map(d=>(6-d.course)/5);
 // ===== 学習済みモデル（過去約4530レースで較正した1着確率）=====
 // 特徴順: [course_base, class, nat, loc, motor, st, ex]
 const L_BETA=[0.6958,0.1224,0.728,0.0152,0.1583,-0.0137,-0.421];
 const L_MU=[0.1668,0.5436,5.2401,4.6892,32.7627,0.088,6.826];
 const L_SD=[0.1764,0.3059,1.3348,2.1123,11.0123,0.1053,0.1136];
 const scores=data.map(d=>{
   const f=[COURSE_BASE[d.course]??0.05, CLASS_SCORE[d.cls]??0.33, d.nat, d.loc, d.motor, d.st, d.ex];
   let s=0; for(let k=0;k<7;k++){const z=(f[k]-L_MU[k])/L_SD[k]; if(!isNaN(z)) s+=L_BETA[k]*z;} return s;
 });
 const _mx=Math.max(...scores),_e=scores.map(s=>Math.exp(s-_mx)),_sm=_e.reduce((a,b)=>a+b,0)||1;
 const power=_e.map(e=>e/_sm);   // 1着力 ＝ 学習済み1着確率（合計1）
 const pRank=[...power.keys()].sort((a,b)=>power[b]-power[a]);
 const sel=document.getElementById('axisSel').value;
 let axisIdx=(sel==='auto')?pRank[0]:(+sel-1);
 const relGap=(power[pRank[0]]-power[pRank[1]])/power[pRank[0]];
 const axisFrame=axisIdx+1;
 // 選択中の軸に対応する実オッズを全30通り(lastOddsAll)から都度引く（軸を変えると追従）
 const liveOdds=data.map((d,i)=>{
   if(i===axisIdx) return null;
   if(lastOddsAll){const v=lastOddsAll[`${axisFrame}-${i+1}`]; if(v!=null) return v;}
   return d.odds;
 });
 const impl=liveOdds.map(v=>v!=null?1/v:null),implN=norm(impl,false);
 const res=data.map((d,i)=>{const J=0.40*motorN[i]+0.25*stN[i]+0.20*exN[i]+0.15*locN[i];
   const oi=liveOdds[i];
   const M=oi!=null?implN[i]:0.45*clsN[i]+0.35*natN[i]+0.20*inner[i];
   return{i,frame:i+1,course:d.course,power:power[i],J,M,Y:J-M,byOdds:oi!=null,odds:oi};});
 const axis=res[axisIdx];
 const width=parseInt(document.getElementById('width').value);
 const cand=res.filter(r=>r.i!==axis.i).sort((a,b)=>b.Y-a.Y);
 const buys=cand.slice(0,width);const HON=0.15;
 // ---- 期待値(EV)判定：ライブオッズがある時だけ計算 ----
 const psum=power.reduce((a,b)=>a+b,0);
 const p1=power.map(x=>x/psum);                       // 各艇の1着確率(暫定)
 const others=[...Array(6).keys()].filter(i=>i!==axisIdx);
 const osum=others.reduce((s,i)=>s+power[i],0)||1;
 let evList=[];
 for(const i of others){
   const o=liveOdds[i]; if(o==null) continue;
   const p=p1[axisIdx]*(power[i]/osum);               // P(軸=1着 かつ i=2着)
   evList.push({frame:i+1, p, o, ev:p*o-1});
 }
 evList.sort((a,b)=>b.ev-a.ev);
 const evBuys=evList.filter(e=>e.ev>0);
 const confEl=document.getElementById('conf');let cCls;const autoTop=res[pRank[0]],second=res[pRank[1]];
 const headProb=power[pRank[0]]*100;  // 学習モデルの頭の1着確率
 if(headProb>=70){cCls='cHi';confEl.innerHTML=`1着信頼度：<b>高</b>（${autoTop.frame}号艇の1着確率 <b>${headProb.toFixed(0)}%</b>＝学習モデルで堅い）`;}
 else if(headProb>=60){cCls='cMid';confEl.innerHTML=`1着信頼度：<b>中</b>（${autoTop.frame}号艇の1着確率 <b>${headProb.toFixed(0)}%</b>）`;}
 else{cCls='cLow';confEl.innerHTML=`1着信頼度：<b>低 ⚠ 見送り推奨</b>（${autoTop.frame}号艇でも1着確率 <b>${headProb.toFixed(0)}%</b>＝学習モデルの推奨しきい値60%未満）`;}
 if(sel!=='auto'&&axisIdx!==pRank[0])confEl.innerHTML+=`　／ 自動推奨頭は${autoTop.frame}号艇（手動で${axis.frame}号艇指定中）`;
 confEl.className='conf '+cCls;
 const order=[axis,...cand];const rowsEl=document.getElementById('rows');rowsEl.innerHTML='';
 const maxAbs=Math.max(0.5,...res.map(r=>Math.abs(r.Y)));const maxP=Math.max(...power);
 order.forEach(r=>{const isAxis=r.i===axis.i,picked=buys.includes(r);let tag;
  if(isAxis)tag=`<span class="flag fAxis">1着 軸</span>`;
  else if(picked&&r.Y>=HON)tag=`<span class="flag fHon">◎本命妙味</span>`;
  else if(picked)tag=`<span class="flag fSpec">○投機カバー</span>`;else tag=`<span class="flag fSkip">見送り</span>`;
  const yPct=Math.round(Math.abs(r.Y)/maxAbs*100);
  const inner_html=isAxis
   ?`<div class="barline"><span class="barlabel">1着力</span><div class="bar bP" style="width:${Math.round(r.power/maxP*100)}%"></div><span>${(r.power/maxP*100).toFixed(0)}</span></div>`
   :`<div class="barline"><span class="barlabel">実力</span><div class="bar bJ" style="width:${Math.round(r.J*100)}%"></div><span>${(r.J*100).toFixed(0)}</span></div>
     <div class="barline"><span class="barlabel">市場評価</span><div class="bar bM" style="width:${Math.round(r.M*100)}%"></div><span>${(r.M*100).toFixed(0)}</span></div>
     <div class="barline"><span class="barlabel">妙味</span><div class="bar ${r.Y>=0?'bY':'bYn'}" style="width:${yPct}%"></div><span>${r.Y>=0?'+':''}${(r.Y*100).toFixed(0)}</span></div>`;
  rowsEl.insertAdjacentHTML('beforeend',`<div class="rrow"><div class="chip ${CC[r.i]}">${r.frame}</div>
    <div><div style="margin-bottom:3px;">${tag} <span class="small">${isAxis?'頭(1着)':'2着候補'}${r.byOdds?' ・実オッズ':''}</span></div>${inner_html}</div>
    <div class="myo" style="color:${isAxis?'#5a3fb0':(r.Y>=0?'#1c7a38':'#c0392b')}">${isAxis?(r.power/maxP*100).toFixed(0):(r.Y>=0?'+':'')+(r.Y*100).toFixed(0)}</div></div>`);});
 const legs=buys.map(b=>b.frame).join('・');const combos=buys.length*4,cost=combos*100;
 // ---- 期待値(EV)判定の表示 ----
 let evHtml='';
 if(evList.length){
   const lines=evList.map(e=>`<div style="font-size:12px;padding:1px 0;">${axisFrame}-${e.frame}：オッズ${e.o.toFixed(1)} × 推定${(e.p*100).toFixed(1)}% → EV <b style="color:${e.ev>=0?'#1c7a38':'#c0392b'}">${e.ev>=0?'+':''}${(e.ev*100).toFixed(0)}%</b></div>`).join('');
   const highOdds=evBuys.filter(e=>e.o>=15).length;
   const rec=evBuys.length
     ? `EV試算プラス：${evBuys.map(e=>axisFrame+'-'+e.frame).join('、')}`+(highOdds?`　⚠ <b style="color:#a02525;">高オッズ艇が含まれます＝モデルが穴を過大評価している可能性大。鵜呑み禁物。</b>`:'')
     : `EV試算プラスの買い目なし。`;
   evHtml=`<div style="background:#eef7ff;border:1px solid #b8d4ef;border-radius:8px;padding:10px;margin-bottom:10px;">
     <div style="font-weight:800;color:#12263a;margin-bottom:4px;">📊 期待値(EV)の試算 ＜実験中・未較正＞</div>
     ${lines}<div style="margin-top:5px;font-size:12.5px;">${rec}</div>
     <div style="font-size:11px;color:#a02525;margin-top:4px;">⚠ 確率は未較正の暫定モデル。EVプラスが高オッズ艇に偏るのは「穴の過大評価」の典型で、これは買い推奨ではありません。数百件記録して較正するまでは<b>参考値</b>として見てください。</div></div>`;
 } else {
   evHtml=`<div style="font-size:12px;color:#9a6a10;background:#fdf3e0;border-radius:8px;padding:8px 10px;margin-bottom:10px;">📊 EV試算：ライブオッズ未取得のため計算不可（下は推定妙味による目安）。</div>`;
 }
 // ---- 実戦判定（バックテスト最適ルール：1着確率≥しきい値 → 2連単1点[1着力1位→2位]、賭け金=資金管理）----
 const buyThr=nz('buyThr',80), stake=calcStake();
 const p1st=res[pRank[0]].frame, p2nd=res[pRank[1]].frame;
 let actionHtml;
 if(headProb>=buyThr){
   actionHtml=`<div class="verdict" style="margin-bottom:10px;">✅ 【買い】2連単 <span style="font-size:26px;">${p1st}-${p2nd}</span> を1点／賭け金 <b>¥${stake.toLocaleString()}</b><br><span style="font-size:12px;font-weight:600;">1着確率 ${headProb.toFixed(0)}% ≥ ${buyThr}%（最適ルールの買い条件クリア。買い目＝1着力1位→2位）</span></div>`;
 } else {
   actionHtml=`<div class="verdict" style="margin-bottom:10px;background:#fdecec;color:#a02525;border-color:#e07272;">🚫 【見送り】1着確率 ${headProb.toFixed(0)}% ＜ ${buyThr}%（買い条件を満たさない＝賭けない）</div>`;
 }
 document.getElementById('verdict').innerHTML=`${actionHtml}${evHtml}<div style="opacity:.75;font-size:14px;">【参考】妙味フォーメーション：${axis.frame} → ${legs} → 全（${combos}点${cost.toLocaleString()}円）</div>`;
 const spec=buys.filter(b=>b.Y<HON).length;const roi=document.getElementById('roiNote');
 if(cCls==='cLow')roi.innerHTML=`⚠ <b>1着信頼度が低いレース。</b>頭自体が飛ぶ危険が高いので、買う前に「勝負するか見送るか」を先に判断してください。`;
 else if(spec>0)roi.innerHTML=`⚖️ 網に<b>「○投機カバー」が${spec}艇</b>（根拠薄・オッズ頼み）。的中率は上がるが回収率は75%側へ。`;
 else roi.innerHTML=`✅ 買い目は全て<b>「◎本命妙味」（裏付けあり）</b>。一番濃い狙い方です。`;
 document.getElementById('expl').innerHTML=`1着力順：${pRank.map(i=>res[i].frame+'号艇').join(' > ')}。頭＝${axis.frame}号艇、2着に ${legs} を流します。<br><span class="small">結果は記録シートへ。当たり外れ両方を残して通算回収率で検証を。</span>`;
 document.getElementById('resultCard').style.display='block';
}
// ===== 資金管理 =====
function nz(id,d){const v=parseFloat(document.getElementById(id).value);return isNaN(v)?d:v;}
function calcStake(){
  const bank=nz('bank',30000);
  if(document.getElementById('stakeMethod').value==='pct'){
    return Math.max(100, Math.round(bank*nz('stakePct',1)/100/100)*100);
  }
  return Math.max(100, Math.round(nz('stakeFixed',1000)/100)*100);
}
function loadSess(){
  let s={start:nz('bank',30000),cur:nz('bank',30000)};
  try{const j=localStorage.getItem('brSess'); if(j) s=JSON.parse(j);}catch(e){}
  return s;
}
function saveSess(s){try{localStorage.setItem('brSess',JSON.stringify(s));}catch(e){}}
function renderSess(){
  const s=loadSess();
  document.getElementById('curBank').textContent=s.cur.toLocaleString();
  const pnl=s.cur-s.start;
  document.getElementById('sessPnl').textContent=(pnl>=0?'+':'')+pnl.toLocaleString();
  const stop=nz('stopLoss',10000), msg=document.getElementById('stopMsg');
  if(pnl<=-stop){ msg.innerHTML='🛑 <b style="color:#a02525;">損切りライン到達。今日はやめましょう。</b>'; }
  else { msg.innerHTML=`損切りまであと ${(stop+pnl).toLocaleString()} 円`; }
}
function logResult(hit){
  const s=loadSess(), stake=calcStake();
  if(hit){ const pay=parseFloat(prompt('受け取った配当（円）を入力：','')); if(isNaN(pay))return; s.cur+=(pay-stake); }
  else { s.cur-=stake; }
  saveSess(s); renderSess();
}
function resetSession(){ const s={start:nz('bank',30000),cur:nz('bank',30000)}; saveSess(s); renderSess(); }
initSelectors();
buildRows([1,2,3,4,5,6].map(blankRow));
renderSess();
</script>
</body>
</html>
'''

@app.get("/")
def index():
    # トップ=今日の高オッズEV買い目（自動一覧）。手動1レース判定は /highodds、妙味は /myoumi。
    return today_page()


@app.get("/myoumi")
def myoumi_page():
    return HTMLResponse(INDEX_HTML)


# ================= 両モデル ダッシュボード（GitHubの台帳をサーバー側取得）=================
GH_RAW = "https://raw.githubusercontent.com/fujita-maker/boat-race-ai/main/data/"
_VEN = {1:"桐生",2:"戸田",3:"江戸川",4:"平和島",5:"多摩川",6:"浜名湖",7:"蒲郡",8:"常滑",9:"津",
        10:"三国",11:"びわこ",12:"住之江",13:"尼崎",14:"鳴門",15:"丸亀",16:"児島",17:"宮島",
        18:"徳山",19:"下関",20:"若松",21:"芦屋",22:"福岡",23:"唐津",24:"大村"}


def _gh_ledger(name):
    try:
        r = requests.get(GH_RAW + name + "?t=" + str(int(time.time())), headers=UA, timeout=15)
        if r.status_code != 200:
            return None
        return list(_csv.DictReader(r.text.splitlines()))
    except Exception:
        return None


def _gh_json(name):
    try:
        r = requests.get(GH_RAW + name + "?t=" + str(int(time.time())), headers=UA, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# =====================================================================
# v65 (2026-09-07): 進入（スタート展示のコース）を扱う仕組み
#   背景: 3連単モデルの最大特徴量は「コース」(_CBASE)。進入が枠なりでないレースでは
#   前提そのものが変わるため、進入を見ないで判定すると別のレースを予想しているのと同じ。
#   過去249日(2026-01-01〜09-06・38,534R)の実測:
#     ・本番が枠なりでないレース = 19.3%（場によって1.4%〜27.9%）
#     ・展示進入と本番進入が全艇一致 = 87.8%（展示が枠なり→本番も枠なり 92.4%）
#   dec40混戦の買い(170日・2,052件)を進入で層別すると:
#     ・本番が枠なり     ROI 100.3%(1,637件)
#     ・本番が進入崩れ   ROI  70.2%(  415件)  ← 構造的に負ける
#   ただし「朝の段階で前づけ率から荒れそうなレースを外す」は回収率が改善しない(検証済)。
#   効くのは展示進入を見てからの足切りだけ。→ dec40_lane（下記）。
# =====================================================================
_MZ_CACHE = {"t": 0.0, "d": None}

# 前づけ率マスタの内蔵コピー(gzip+base64)。2026-01-01〜09-06 の249日・1,630選手から作成。
# data/maezuke_master.json がリポジトリに無くても動くようにするための同梱データ。
# リポジトリに data/maezuke_master.json を置けばそちらが優先される(月次Actionで更新できる)。
_MZ_EMBED = (
    "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"
)


def _maezuke_embedded():
    try:
        return json.loads(gzip.decompress(base64.b64decode(_MZ_EMBED)).decode("utf-8"))
    except Exception:
        return None


def _maezuke_master():
    """選手別の内寄せ率(前づけ率)マスタ。1時間キャッシュ。
    優先順: data/maezuke_master.json (GitHub) → app.py 内蔵コピー。
    どちらも読めなければ None（進入リスク表示だけが消える。買い判定には影響しない）。"""
    now = time.time()
    # 重要: 取得に失敗した場合もタイムスタンプを更新してキャッシュする。
    #   失敗時にキャッシュしないと、レースごとに毎回GitHubへ取りに行ってしまい
    #   /final の生成が最大288回のHTTPリクエストになり Internal Server Error になる(2026-09-07に発生)。
    if _MZ_CACHE["t"] and now - _MZ_CACHE["t"] < 3600:
        return _MZ_CACHE["d"]
    _MZ_CACHE["t"] = now          # 先に更新して、以降1時間は必ず1回だけにする
    d = None
    try:
        d = _gh_json("maezuke_master.json")
    except Exception:
        d = None
    if not d:
        d = _maezuke_embedded()   # app.py 内蔵コピー
    _MZ_CACHE["d"] = d
    return d


def _lane_map(race):
    """展示進入(スタート展示のコース)。{枠番: コース} を返す。未取得なら値がNone。
    公式の直前情報は締切のおよそ10〜13分前に出る。"""
    pr = ((race.get("preview") or {}).get("racers") or {})
    out = {}
    for f in range(1, 7):
        v = (pr.get(str(f)) or pr.get(f) or {}).get("course_number")
        try:
            out[f] = int(v) if v else None
        except (TypeError, ValueError):
            out[f] = None
    return out


def _lane_known(race):
    lm = _lane_map(race)
    return all(lm.get(f) for f in range(1, 7))


def _lane_map_result(race):
    """本番の進入(レース結果に載るコース)。{枠番: コース}。レース確定後にしか入らない。"""
    rr = ((race.get("result") or {}).get("racers") or {})
    out = {}
    for f in range(1, 7):
        v = (rr.get(str(f)) or rr.get(f) or {}).get("course_number")
        try:
            out[f] = int(v) if v else None
        except (TypeError, ValueError):
            out[f] = None
    return out


def _lane_real(race, key_frame, honmei_frame):
    """精算時に本番進入を記録する。展示(スタート展示)と本番は約12%のレースで食い違う
    (過去249日実測: 全艇一致87.8%。展示枠なり→本番も枠なり92.4% / 展示崩れ→本番も崩れ84.1%)。
    本番進入は締切後・スタート直前にしか決まらないので、事前に知る方法は無い＝残余リスクの記録。"""
    lm = _lane_map_result(race)
    if not all(lm.get(f) for f in range(1, 7)):
        return None
    inv = {c: f for f, c in lm.items()}
    if len(inv) != 6:
        return None
    out = {"known": True, "text": "-".join(str(inv[c]) for c in range(1, 7)),
           "wakunari": all(lm[f] == f for f in range(1, 7))}
    try:
        X = int(key_frame); out["key_out"] = lm[X] > X; out["key_in"] = lm[X] < X
    except (TypeError, ValueError):
        out["key_out"] = out["key_in"] = False
    try:
        P = int(honmei_frame); out["honmei_moved"] = lm[P] != P
    except (TypeError, ValueError):
        out["honmei_moved"] = False
    return out


def _lane_text(race):
    """進入隊形を『1-2-4-3-5-6』の形（内側コース順の枠番）で返す。未取得なら空。"""
    lm = _lane_map(race)
    if not all(lm.get(f) for f in range(1, 7)):
        return ""
    inv = {}
    for f, c in lm.items():
        inv[c] = f
    if len(inv) != 6:
        return ""
    return "-".join(str(inv[c]) for c in range(1, 7))


def _shinnyu_risk(race, key_frame=None, honmei_frame=None):
    """出走表段階(朝)で分かる進入リスク。選手別の内寄せ率から算出。
    返り値: {"race": 隊形が崩れる確率, "key_out": キーが外に出される確率,
             "honmei_move": 本命が動く確率, "max": 上2つの大きい方, "names": [前づけ常習者]}
    ※ 検証では『これで朝に足切りしても回収率は上がらない』。表示・注意喚起のための情報。"""
    m = _maezuke_master()
    if not m:
        return None
    g = float(m.get("global_inside_rate") or 0.05)
    rm = m.get("racers") or {}
    ents = []
    _collect_with_key(race, "national_win_rate", ents)
    by = {}
    for e in ents:
        n = e.get("entry_number")
        if n:
            by[int(n)] = e

    def p(f):
        e = by.get(f) or {}
        v = rm.get(str(e.get("number")))
        return float(v[0]) if v else g

    def anyof(frames):
        q = 1.0
        for f in frames:
            q *= (1 - p(f))
        return 1 - q
    names = []
    for f in range(2, 7):
        if p(f) >= 0.20:
            names.append("%d号%s(%.0f%%)" % (f, (by.get(f) or {}).get("name") or "", p(f) * 100))
    out = {"race": anyof(range(2, 7)), "names": names,
           "key_out": (anyof(range(int(key_frame) + 1, 7)) if key_frame else None),
           "honmei_move": None}
    if honmei_frame:
        hf = int(honmei_frame)
        hm = anyof(range(hf + 1, 7))
        if hf > 1:
            hm = 1 - (1 - hm) * (1 - p(hf))
        out["honmei_move"] = hm
    cands = [v for v in (out["key_out"], out["honmei_move"]) if v is not None]
    out["max"] = max(cands) if cands else out["race"]
    return out


def _shinnyu_label(r):
    if not r:
        return "—"
    v = r.get("max") or 0
    if v >= 0.35:
        return "高"
    if v >= 0.20:
        return "中"
    return "低"


def _lane_line(uv):
    """レースカードに出す進入の1行。uv_json の lane 情報から作る(買い/見送りに関係なく毎レース表示)。"""
    ln = (uv or {}).get("lane") or {}
    X = ((uv or {}).get("key") or {}).get("frame")
    P = (uv or {}).get("honmei")
    if not ln:
        return ('<div class="tsub" style="margin-top:4px">進入 —'
                '<span style="color:#8a97a5">（この判定より前のデータのため記録なし）</span></div>')
    if not ln.get("known"):
        return ('<div class="tsub" style="margin-top:4px">進入 <b>未取得</b>'
                '<span style="color:#b8860b">（展示進入が出る前に判定＝枠なり前提の計算）</span></div>')
    txt = ln.get("text") or ""
    if ln.get("wakunari"):
        return ('<div class="tsub" style="margin-top:4px">進入 '
                '<b style="color:#1c7a38">%s</b> <span style="color:#1c7a38">枠なり</span></div>') % txt
    notes = []
    if ln.get("key_out"):
        notes.append('<span style="color:#b23b3b">キー%s号艇が外へ回された＝この賭けの前提が崩れる型'
                     '（過去ROI66%%）</span>' % X)
    elif ln.get("key_in"):
        notes.append('<span style="color:#1c7a38">キー%s号艇が内に入った＝キーに有利な型'
                     '（過去ROI101%%）</span>' % X)
    if ln.get("honmei_moved"):
        notes.append('<span style="color:#b23b3b">本命%s号艇が動いた＝並びが読めない型'
                     '（過去ROI約50%%）</span>' % P)
    if not notes:
        notes.append('<span style="color:#5c6b7a">キー・本命は枠番どおり</span>')
    return ('<div class="tsub" style="margin-top:4px">進入 <b style="color:#b8860b">%s</b> '
            '<span style="color:#b8860b">枠なり崩れ</span>　%s</div>') % (txt, "／".join(notes))


def _midev_line(uv):
    """中オッズEV(記録のみ)の1行。買い目が立った時だけ出す。"""
    mv = (uv or {}).get("midev") or {}
    pk = mv.get("picks") or []
    if _MIDEV_DISPLAY:
        pk = midev_shadow_picks(mv, _MIDEV_DISPLAY)
    if not pk:
        return ""
    body = " / ".join('<span class="combo">%s</span>(%s倍・EV%s)' % (p.get("combo"), p.get("odds"), p.get("ev"))
                      for p in pk)
    tail = ""
    if mv.get("hit") is not None and str(mv.get("hit")) != "":
        tail = ('　<span class="res-hit">的中（払戻%s円）</span>' % f"{int(_fnum(mv.get('ret'))):,}"
                if str(mv.get("hit")) == "1" else '　<span class="res-miss">不的中</span>')
    if _MIDEV_DISPLAY:
        return ('<div class="tsub" style="margin-top:2px"><b style="color:#1c7a38">C案 買い</b>'
                '<span style="color:#8a97a5">（%s の1点×%s円）</span>'
                '　%s</div>') % (dict(_MIDEV_SHADOWS)[_MIDEV_DISPLAY], _MIDEV_UNIT, body)
    tags = "".join('　<span style="background:#e8ecf1;color:#5c6b7a;border-radius:6px;padding:0 6px;'
                   'font-size:11px">影:%s</span>' % lab
                   for nm, lab in _MIDEV_SHADOWS if midev_shadow_picks(mv, nm))
    return ('<div class="tsub" style="margin-top:2px">中オッズEV<span style="color:#8a97a5">（記録のみ・'
            '%s〜%s倍かつEV≥%s を%s点×%s円）</span>　%s%s%s</div>') % (
        int(_MIDEV_LO), int(_MIDEV_HI), _MIDEV_EV, len(pk), _MIDEV_UNIT, body, tail, tags)


def _lane_real_line(uv):
    """レース確定後に、本番の進入を1行で。展示と違っていた場合だけ目立たせる。"""
    lr = (uv or {}).get("lane_real") or {}
    if not lr.get("known"):
        return ""
    X = ((uv or {}).get("key") or {}).get("frame")
    P = (uv or {}).get("honmei")
    if not lr.get("changed"):
        return ('<div class="tsub" style="margin-top:2px">本番進入 <b>%s</b> '
                '<span style="color:#8a97a5">＝展示どおり</span></div>') % lr.get("text")
    notes = []
    if lr.get("key_out"):
        notes.append('<span style="color:#b23b3b">キー%s号艇が外へ</span>' % X)
    elif lr.get("key_in"):
        notes.append('<span style="color:#1c7a38">キー%s号艇が内へ</span>' % X)
    if lr.get("honmei_moved"):
        notes.append('<span style="color:#b23b3b">本命%s号艇が動いた</span>' % P)
    tail = ("　" + "／".join(notes)) if notes else ""
    return ('<div class="tsub" style="margin-top:2px">本番進入 <b style="color:#b23b3b">%s</b> '
            '<span style="color:#b23b3b">＝展示から変化（判定後に変わったので事前には防げない型・全体の約12%%）</span>'
            '%s</div>') % (lr.get("text"), tail)


_UV_DERIVED = ("dec40_stable", "dec40_in", "dec40_in_ev", "dec_in_ev12")
# v80/v81: 既存フィールドから合成する派生バケット(旧行のdecisionで代用しない)

_UV_LIVE_IDX = (1, 3, 4, 5)   # 8点並び(P-Q-X/Q-P-X/P-R-X/P-X-Q/Q-X-P/P-X-R/X-P-Q/X-Q-P)のうち実弾4点


def _uv_live4(u):
    """uv_json から「実際に買う4点」を取り出す。点数構成はv58→v59→v62で変わっているが
    combos の生成順は不変なので、位置で拾えば全期間を同じ土俵に揃えられる。"""
    pa = u.get("picks_all") or []
    pk = u.get("picks") or []
    src = pa if len(pa) == 8 else pk
    if len(src) == 4:
        return src
    if len(src) in (6, 8):
        return [src[i] for i in _UV_LIVE_IDX]
    return []


def _uv_ev1(u):
    """実弾4点の1点あたり期待値(モデルP×判定時オッズの平均)。取れなければ None。
    現行の basket_ev8 は「買わない4点も含めた8点」のEVなので別物。"""
    live = _uv_live4(u)
    if len(live) != 4:
        return None
    tot = 0.0
    for q in live:
        o = _fnum(q.get("odds")); pr = _fnum(q.get("P"))
        if not o:
            return None
        tot += pr * o
    return tot / 4.0


def _uv_base(u):
    """過小評価キーの土台ゲート(gap>=3・モーター2連率>=40・全国2連率>=5・買い目最大オッズ>=15)。
    ※6点記録の時期(v59)は1着2点のオッズが台帳に無く最大オッズを過小評価しうる(29行のみ)。"""
    k = u.get("key") or {}
    if (_fnum(k.get("gap")) < 3 or _fnum(k.get("motor2")) < 40
            or _fnum(k.get("nat2")) < 5):
        return False
    pa = u.get("picks_all") or []
    pk = u.get("picks") or []
    ods = [_fnum(q.get("odds")) for q in (pa if len(pa) == 8 else pk)]
    ods = [o for o in ods if o]
    return bool(ods and max(ods) >= 15)


# v84: 「買い目の組み方」を点単位で見直す影バケット。
#   実弾は形で4点決め打ち(Q-P-X / P-X-Q / Q-X-P / P-X-R)しており、その点のオッズを見ていない。
#   ライブ458レース1,832点の実測:
#     ・的中50本は全て150倍以下に入っていた。150倍超の324点(=324,000円)は的中ゼロ。
#       150倍以下だけ買うと 回収率111.7%→135.7%、大穴3本抜き89.2%→108.3%、前後半とも改善。
#     ・1点ごとのEV(モデルP×判定時オッズ)≥1.5 で選ぶと 600点・186.2%・抜3 118.5%。
#       同じ点数をランダムに抜いた場合(3,000回)の95%点163.3%を上回る(p=0.010)。
#   いずれも記録のみ。実弾の4点構成は変えない。
_UVP_UNIT = 1000.0   # 1点あたりの賭け金(undervalue_pick 内の _UV_UNIT と同じ値)
_UVP_ODDS_CAP = 150.0
_UVP_EV_FLOOR = 1.5

# v85(2026-09-19 藤田指示「③に今日から変更して」): 実弾の買い目を「形で4点」から
#   「4点のうち オッズ≤150倍 かつ 1点のEV(モデルP×判定時オッズ)≥1.5 の点だけ」に変更。
#   ライブ458レース1,832点を買い方ごとに再精算した実測:
#     現行4点 : 買458R/1832点/的中50/的中率10.9%/回収率111.7%/大穴3本抜き89.2%/収支+214,400円
#     ③      : 買240R/ 396点/的中15/的中率 6.2%/回収率282.2%/大穴3本抜き179.6%/収支+721,400円
#     投資は1/5以下(1日83,273円→18,000円)、最大DDは284,900円→133,100円。
#   代償: 的中率が下がるぶん連敗が伸びる。最大53連敗・平均16.1回・中央値10回
#         (現行は41回/9.5回/6回)。日数換算で最大4.9日ぶん当たらない。
#         ただし偶然の95%点は84連敗なので、53連敗自体は異常ではない。
#   注意: 2条件を重ねて選んでいるぶん選択バイアスが乗っている。前向きの成績で要確認。
#   ゲート計算(odds_ok の max_odds、basket_ev8、pgap)は従来どおり8点ベースのまま＝
#   「買うレースの集合」は変えない。変わるのは「そのレースで何点買うか」だけ。
#   None にすれば従来の4点固定に戻る。
_UV_LEG_ODDS_CAP = 150.0
_UV_LEG_EV_FLOOR = 1.5
_UVP_FILTERS = (
    ("uvp_o150", "オッズ150倍以下の点だけ買う", lambda q: _fnum(q.get("odds")) <= _UVP_ODDS_CAP),
    ("uvp_ev15", "1点のEV≥1.5の点だけ買う",
     lambda q: _fnum(q.get("P")) * _fnum(q.get("odds")) >= _UVP_EV_FLOOR),
    ("uvp_both", "150倍以下 かつ 1点のEV≥1.5",
     lambda q: _fnum(q.get("odds")) <= _UVP_ODDS_CAP
     and _fnum(q.get("P")) * _fnum(q.get("odds")) >= _UVP_EV_FLOOR),
)


def uvp_result(r, u, name):
    """実弾4点のうち条件を満たす点だけ買った場合の (点数, 投資, 払戻)。対象外は (0,0,0)。"""
    if u.get("dec40") != "買い" and u.get("dec_stable") != "買い":
        return 0, 0.0, 0.0
    live = _uv_live4(u)
    if len(live) != 4 or any(not _fnum(q.get("odds")) for q in live):
        return 0, 0.0, 0.0
    f = dict((n, fn) for n, _, fn in _UVP_FILTERS).get(name)
    if f is None:
        return 0, 0.0, 0.0
    win = r.get("win_combo") or ""
    pts = [q for q in live if f(q)]
    if not pts:
        return 0, 0.0, 0.0
    stake = _UVP_UNIT * len(pts)
    ret = sum(_UVP_UNIT * _fnum(q.get("odds")) for q in pts if q.get("combo") == win)
    return len(pts), stake, ret


def _uv_key_inner(u):
    """キー艇の枠が2〜3か。base1,012レースの実測で的中率 内枠14.2% vs 外枠(4-6)6.1%。
    モデル予測Pで層別しても差が残る＝モデルが外枠キーの2・3着確率を過大評価している。"""
    return (u.get("key") or {}).get("frame") in (2, 3)


def _uv_dec(u, field):
    """uv_json から各バケットの判定を取り出す。
    v80: 'dec40_stable'(B案≥40混戦 ∧ A案安定型)は新しい記録項目を足さず、既存の
         dec40 と dec_stable から合成する派生バケット。両方は 2026-08-28 の台帳開始時から
         記録されているので、過去の全レースに遡って集計できる(0件から貯め直す必要がない)。
         A案とB案は同じ undervalue_pick の買い目を共有するため、これは券を増やす話ではなく
         「1枚の券に両ゲートが掛かったレースだけを抜き出した部分集合」の成績。
    """
    if field == "dec40_stable":
        d40 = u.get("dec40"); dst = u.get("dec_stable")
        if d40 is None or dst is None:
            return None          # 旧行(両フィールドが無い)は集計から除外
        return "買い" if (d40 == "買い" and dst == "買い") else "見送り"
    # v81: キーの枠と「実弾4点のEV」による派生バケット。どちらも判定時の記録から合成できる。
    if field in ("dec40_in", "dec40_in_ev", "dec_in_ev12"):
        if not (u.get("key") or {}).get("frame"):
            return None
        inner = _uv_key_inner(u)
        if field == "dec40_in":
            d40 = u.get("dec40")
            if d40 is None:
                return None
            return "買い" if (d40 == "買い" and inner) else "見送り"
        ev1 = _uv_ev1(u)
        if ev1 is None:
            return None
        if field == "dec40_in_ev":
            d40 = u.get("dec40")
            if d40 is None:
                return None
            return "買い" if (d40 == "買い" and inner and ev1 >= 1.0) else "見送り"
        # dec_in_ev12: 混戦ゲートを使わない別系統(土台ゲートのみ＋内枠＋4点EV>=1.2)
        return "買い" if (_uv_base(u) and inner and ev1 >= 1.2) else "見送り"
    return u.get(field)


def _uv_dec_or_decision(u, field):
    """判定を取り出す。従来フィールドが無い旧行は decision で代用するが、
    派生バケットは合成元が無い＝集計対象外(None)にする(旧行が買い側に混入しないように)。"""
    d = _uv_dec(u, field)
    if d is None and field not in _UV_DERIVED:
        d = u.get("decision")
    return d


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _dash_stats(rows):
    s = [r for r in rows if r.get("status") == "settled"]
    stake = sum(_fnum(r.get("stake")) for r in s)
    ret = sum(_fnum(r.get("ret")) for r in s)
    races = {}
    for r in s:
        k = (r.get("date"), r.get("jcd"), r.get("rno"))
        races[k] = races.get(k, 0) or (1 if _fnum(r.get("hit")) else 0)
    rn = len(races); rh = sum(races.values())
    # 日別累積純益
    by = {}
    for r in s:
        by[r.get("date")] = by.get(r.get("date"), 0.0) + (_fnum(r.get("ret")) - _fnum(r.get("stake")))
    cum = []; c = 0.0
    for d in sorted(by):
        c += by[d]; cum.append(c)
    return dict(picks=len(s), races=rn, rhit=rh, rhitpct=(rh/rn*100 if rn else 0),
                stake=stake, ret=ret, net=ret-stake, roi=(ret/stake*100 if stake else 0), cum=cum)


def _yen(n):
    n = round(n)
    return ("−¥" if n < 0 else "¥") + "{:,}".format(abs(int(n)))


def _spark(cumA, cumB):
    W, H, pad = 640, 150, 30
    allv = list(cumA) + list(cumB) + [0.0]
    if len(allv) <= 1:
        return '<div class="tsub" style="padding:8px">データがたまると累積収支グラフが出ます。</div>'
    mn, mx = min(allv), max(allv); rng = (mx - mn) or 1
    idx = max(len(cumA), len(cumB), 2)
    def X(i): return pad + (W - pad - 8) * (0 if idx < 2 else i/(idx-1))
    def Y(v): return 12 + (H - pad) * (1 - (v - mn)/rng)
    def line(pts, col):
        if not pts:
            return ""
        d = " ".join(("M" if i == 0 else "L") + "%.1f %.1f" % (X(i), Y(v)) for i, v in enumerate(pts))
        return '<path d="%s" fill="none" stroke="%s" stroke-width="2.4"/><circle cx="%.1f" cy="%.1f" r="3.5" fill="%s"/>' % (d, col, X(len(pts)-1), Y(pts[-1]), col)
    z = Y(0)
    return ('<svg viewBox="0 0 %d %d" width="100%%" style="max-width:%dpx">'
            '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#dbe2ea" stroke-dasharray="4 4"/>'
            '<text x="%d" y="%.1f" text-anchor="end" font-size="10" fill="#8a97a5">0</text>'
            '%s%s</svg>') % (W, H, W, pad, z, W-8, z, pad-5, z+4,
                             line(cumA, "#3a5bd0"), line(cumB, "#1c7a38"))


def _dash_table(rows, kind):
    s = [r for r in rows if r.get("status") == "settled"][-16:][::-1]
    if not s:
        return '<div class="tsub" style="padding:6px">まだ記録がありません。</div>'
    h = '<table><tr><th>日</th><th>場</th><th>R</th><th>買い目</th><th>結果</th><th>払戻</th></tr>'
    for r in s:
        win = _fnum(r.get("hit"))
        d = str(r.get("date", "")); md = d[4:6] + "/" + d[6:8] if len(d) >= 8 else d
        ven = _VEN.get(int(r.get("jcd", 0)) if str(r.get("jcd", "")).isdigit() else 0, r.get("jcd"))
        act = ""
        if kind == "hit" and r.get("actual") and not win:
            act = ' <span class="tsub">(' + str(r.get("actual")) + ')</span>'
        h += ('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
              '<td class="%s">%s%s</td><td>%s</td></tr>') % (
            md, ven, r.get("rno"), r.get("combo"),
            "hit" if win else "miss", "的中" if win else "—", act,
            _yen(_fnum(r.get("ret"))) if win else "—")
    return h + "</table>"


_DASH_CSS = """
*{box-sizing:border-box}body{margin:0;background:#eef1f4;color:#12263a;
font-family:-apple-system,"Hiragino Kaku Gothic ProN",sans-serif;padding:16px}
h1{font-size:19px;margin:0 0 2px}.sub{color:#5b6b7d;font-size:12.5px;margin-bottom:14px}
.bar{margin-bottom:12px}.bar a{background:#3a5bd0;color:#fff;text-decoration:none;border-radius:8px;padding:8px 14px;font-size:13px;font-weight:700}
.bar a.g{background:#e7edf5;color:#12263a;margin-left:8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:720px){.grid{grid-template-columns:1fr}}
.card{background:#fff;border:1px solid #dbe2ea;border-radius:14px;padding:16px}
.card h2{font-size:15px;margin:0 0 2px;display:flex;align-items:center;gap:8px}
.dot{width:10px;height:10px;border-radius:50%}
.tag{font-size:11px;color:#5b6b7d;font-weight:600}
.kpis{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:12px}
.kpi{background:#f4f7fa;border:1px solid #dbe2ea;border-radius:10px;padding:10px}
.kpi .lab{font-size:11px;color:#5b6b7d}.kpi .val{font-size:20px;font-weight:800;margin-top:2px}
.good{color:#129b63}.bad{color:#c0392b}
.small{font-size:11.5px;color:#5b6b7d;margin-top:8px}
.chartwrap{background:#fff;border:1px solid #dbe2ea;border-radius:14px;padding:16px;margin-top:14px}
.legend{font-size:12px;color:#5b6b7d;margin-bottom:6px}
.legend b{display:inline-block;width:12px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px}
table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}
th,td{border-bottom:1px solid #dbe2ea;padding:6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
.hit{color:#1c7a38;font-weight:700}.miss{color:#8a97a5}.tsub{color:#8a97a5}
.err{color:#7a5b12;font-size:12px}
details summary{cursor:pointer;color:#3a5bd0;font-size:12.5px;margin-top:8px}
"""


def _card(title, color, tag, rows):
    if rows is None:
        st = None
    else:
        st = _dash_stats(rows)
    if st is None:
        inner = '<div class="err">台帳がまだありません。ワークフローが1回動くと表示されます。</div>'
        return ('<div class="card"><h2><span class="dot" style="background:%s"></span>%s '
                '<span class="tag">%s</span></h2>%s</div>' % (color, title, tag, inner)), []
    roic = "good" if st["roi"] >= 100 else "bad"
    netc = "good" if st["net"] >= 0 else "bad"
    kind = "hit" if "的中" in title else "ho"
    kpis = ('<div class="kpis">'
            '<div class="kpi"><div class="lab">レース数(買い)</div><div class="val">%s</div></div>'
            '<div class="kpi"><div class="lab">的中率(レース)</div><div class="val">%.1f%%</div></div>'
            '<div class="kpi"><div class="lab">回収率</div><div class="val %s">%.1f%%</div></div>'
            '<div class="kpi"><div class="lab">投資</div><div class="val" style="font-size:15px">%s</div></div>'
            '<div class="kpi"><div class="lab">払戻</div><div class="val" style="font-size:15px">%s</div></div>'
            '<div class="kpi"><div class="lab">純益</div><div class="val %s" style="font-size:15px">%s</div></div>'
            '</div>') % (st["races"], st["rhitpct"], roic, st["roi"],
                         _yen(st["stake"]), _yen(st["ret"]), netc, _yen(st["net"]))
    sub = '<div class="small">精算 %d点 / 的中レース %d/%d</div>' % (st["picks"], st["rhit"], st["races"])
    tbl = '<details><summary>最近の買い目を見る</summary>%s</details>' % _dash_table(rows, kind)
    html = ('<div class="card"><h2><span class="dot" style="background:%s"></span>%s '
            '<span class="tag">%s</span></h2>%s%s%s</div>' % (color, title, tag, kpis, sub, tbl))
    return html, st["cum"]


@app.get("/dashboard")
def dashboard():
    ho = _gh_ledger("ho_ledger.csv")
    hit = _gh_ledger("hit_ledger.csv")
    cardA, cumA = _card("高オッズEV", "#3a5bd0", "1-4/1-5・オッズ≥20・EV≥1.2", ho)
    cardB, cumB = _card("的中(見送り)", "#1c7a38", "GBM上位2点・自信度上位20%", hit)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
    html = ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="refresh" content="300">'
            '<title>競艇AI ダッシュボード</title><style>%s</style></head><body>'
            '<h1>&#128675; 競艇AI ダッシュボード</h1>'
            '<div class="sub">高オッズEV と 的中(見送り) の前向き記録。GitHubの台帳から取得。'
            '控除率25%%の壁を越えられるかを実測で追う道具です。更新: %s（5分ごと自動）</div>'
            '<div class="bar"><a href="/dashboard">&#8635; 最新に更新</a>'
            '<a class="g" href="/today">今日の買い目予想</a>'
            '<a class="g" href="/trifecta">&#127919; 3連単 的中率重視</a>'
            '<a class="g" href="/highodds">手動で1レース判定</a></div>'
            '<div class="grid">%s%s</div>'
            '<div class="chartwrap"><div class="legend">'
            '<span><b style="background:#3a5bd0"></b>高オッズ 累積収支</span>&nbsp;&nbsp;'
            '<span><b style="background:#1c7a38"></b>的中 累積収支</span>'
            '&nbsp;&nbsp;<span class="tsub">（0円ラインより上＝プラス）</span></div>%s</div>'
            '</body></html>') % (_DASH_CSS, now, cardA, cardB, _spark(cumA, cumB))
    return HTMLResponse(html)


# ==================== /today 予想生成をRender常駐で自動化（cron非依存）====================
# daily-predict(GitHub cron)が飛んでも /today が固まらないよう、Render常駐スレッドが約20分ごとに
# 当日の未開催レースを再評価して買い目を作り直す。表示はこのRender生成を優先し、無ければGitHubにフォールバック。
import threading as _threading_today  # noqa
_TODAY_CACHE = {"hd": None, "generated_at": None, "races": None}
_today_lock = _threading_today.Lock()


def generate_today_picks():
    now = _tri_now()
    hd = now.strftime("%Y%m%d")
    day0 = now.date()
    try:
        data = fetch_openapi(hd)
    except Exception:
        return 0
    if not data:
        return 0
    out = []
    for jcd in range(1, 25):
        for rno in range(1, 13):
            rc = find_race(data, jcd, rno)
            if rc is None:
                continue
            ex, tri = _payouts(rc)
            if ex is not None:      # 確定済み=未開催でない
                continue
            closed = _tri_parse_closed(_tri_closed_at(rc), day0)
            if closed is None:
                continue
            if (closed - now).total_seconds() <= 0:   # 締切済みは出さない
                continue
            boats = extract_boats(rc)
            power = predict_power(boats)
            conf, gap = _conf_gap(power)
            if conf < 0.12 or gap < 0.30:   # 安いゲートで先に弾く(オッズ取得を節約)
                continue
            odds_map = fetch_exacta_odds(jcd, rno, hd)
            if not odds_map:
                continue
            res = highodds_pick(boats, odds_map, 20.0, 1.20, 0.05, 0.12, 0.30, 2, {"1-4", "1-5"})
            if res["decision"] != "買い" or not res.get("picks"):
                continue
            out.append({"jcd": jcd, "rno": rno,
                        "closed_at": closed.strftime("%Y-%m-%dT%H:%M:00+09:00"),
                        "ho": {"decision": "買い",
                               "picks": [{"combo": p["combo"], "odds": p["odds"], "EV": p["EV"]} for p in res["picks"]]}})
    with _today_lock:
        _TODAY_CACHE["hd"] = hd
        _TODAY_CACHE["generated_at"] = now.strftime("%Y-%m-%d %H:%M")
        _TODAY_CACHE["races"] = out
    return len(out)


def _today_loop():
    while True:
        try:
            if 8 <= _tri_now().hour <= 23:
                generate_today_picks()
        except Exception as e:
            print("[today-gen]", e, flush=True)
        time.sleep(1200)


@app.get("/api/today/gen")
def api_today_gen():
    """手動で当日の買い目を今すぐ再生成する。"""
    n = generate_today_picks()
    return {"ok": True, "buys": n, "generated_at": _TODAY_CACHE.get("generated_at")}


@app.get("/today")
def today_page():
    now_dt = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    hd_today = now_dt.strftime("%Y%m%d")
    now = now_dt.strftime("%Y-%m-%d %H:%M")
    d = None
    with _today_lock:
        if _TODAY_CACHE.get("hd") == hd_today and _TODAY_CACHE.get("races") is not None:
            d = {"hd": _TODAY_CACHE["hd"], "generated_at": _TODAY_CACHE.get("generated_at"),
                 "races": list(_TODAY_CACHE["races"])}
    if d is None:
        d = _gh_json("today_picks.json")
    head = ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="refresh" content="600">'
            '<title>今日の高オッズEV買い目</title><style>%s'
            '.rowsec{background:#fff;border:1px solid #dbe2ea;border-radius:14px;padding:14px;margin-top:14px}'
            '.pill{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700;color:#fff}'
            '.pbuy{background:#3a5bd0}'
            '.combo{font-weight:700;font-size:15px}</style></head><body>'
            '<h1>&#127919; 今日の高オッズEV買い目</h1>'
            '<div class="sub" style="margin-top:-6px">条件＝1号艇軸(1-4/1-5)・オッズ≥20倍・EV≥1.2 のみ表示</div>') % _DASH_CSS
    bar = ('<div class="bar"><a href="/today">&#8635; 最新に更新</a>'
           '<a class="g" href="/trifecta">&#127919; 3連単 的中率重視</a>'
           '<a class="g" href="/final">&#128337; 締切5分前 自動判定</a>'
           '<a class="g" href="/undervalue">&#127775; 過小評価キー</a>'
           '<a class="g" href="/dashboard">成績ダッシュボード</a>'
           '<a class="g" href="/highodds">手動で1レース判定</a></div>')
    if d is None:
        body = ('<div class="sub">まだ今日の予想がありません。Render常駐が約20分ごとに自動生成します'
                '（起動直後は数分お待ちを）。更新: %s</div>' % now)
        return HTMLResponse(head + bar + body + "</body></html>")
    gen = d.get("generated_at", "?"); hd = d.get("hd", "")
    races = d.get("races") or []

    def _hhmm(r):
        ca = str(r.get("closed_at") or "")
        return ca.split("T")[1][:5] if "T" in ca else ca[:5]
    now_hm = now_dt.strftime("%H:%M")
    # 高オッズEVが「買い」かつ締切がまだ先のレースだけ抽出し、締切順に
    buys = [r for r in races if r.get("ho") and r["ho"].get("decision") == "買い"
            and r["ho"].get("picks") and _hhmm(r) > now_hm]
    buys.sort(key=lambda r: str(r.get("closed_at") or "z"))

    def ven(j):
        try:
            return _VEN.get(int(j), j)
        except Exception:
            return j

    def render_rows(rows):
        h = ('<table><tr><th>締切</th><th>場</th><th>R</th>'
             '<th style="text-align:left">買い目（オッズ / EV）</th></tr>')
        for r in rows:
            ca = str(r.get("closed_at") or "")
            ct = ca[11:16] if len(ca) >= 16 else ca
            ps = " / ".join('<span class="combo">%s</span> <span class="tsub">%sx&nbsp;EV%s</span>'
                            % (p["combo"], p["odds"], p["EV"]) for p in r["ho"]["picks"])
            h += ('<tr><td>%s</td><td>%s</td><td>%s</td>'
                  '<td style="text-align:left"><span class="pill pbuy">買い</span> %s</td></tr>') % (
                  ct, ven(r["jcd"]), r["rno"], ps)
        return h + "</table>"

    ho_n = len(buys)
    sub = ('<div class="sub">%s の未開催レース。高オッズEVモデルの「買い」%d件。'
           '予想生成: %s（Render常駐が約20分ごとに自動生成・締切済みは自動で除外）。'
           '<br><span class="tsub">※オッズは生成時点の値。締切直前が最終値なので、賭ける直前に「手動で1レース判定」で再確認を。'
           '控除率25%%の壁は不変で、これは勝ちを保証しません。実弾ゼロで通算回収率を測る道具です。</span>'
           '更新: %s</div>') % (hd, ho_n, gen, now)
    if not buys:
        table = ('<div class="rowsec"><div class="tsub">今のところ高オッズEVの「買い」推奨はありません'
                 '（全レース見送り）。時間をおくと更新されます。</div></div>')
    else:
        table = '<div class="rowsec">' + render_rows(buys) + '</div>'
    legend = ('<div class="small">高オッズEV＝焼き込み条件付きロジットで各艇のP(1着)を出し、'
              '1号艇軸の 1-4/1-5 に絞って オッズ≥20倍 かつ EV(=P×オッズ)≥1.2 の点だけを「買い」。1レース最大2点。</div>')
    return HTMLResponse(head + bar + sub + table + legend + "</body></html>")


# ==================== 🎯 3連単 的中率重視モデル（既存とは別建て）====================
# 半年超(13か月・54,831R)ウォークフォワードBTの最適設定を適用:
#   自信度「上位約5%」ゲート(=conf≥0.18, 学習データ95分位) / 上位5点 / 1レース総額1000円。
#   BTのOOS実績: 的中率≒48% / 回収率≒83%(均等), 確率比例で≒85%。※どれも100%未満(控除率25%の壁)。
# 買い目の順位付けは本サイト共通の焼き込みモデル(logit)→Plackett-Luceで120通り。
import itertools as _it
_TRI_PERMS = list(_it.permutations([1, 2, 3, 4, 5, 6], 3))  # 120通り
TRI_CONF_GATE = 0.18   # 自信度ゲート（学習データ95分位＝上位約5%のレースだけ買う＝見送り多め）
TRI_N = 5              # 買い目数（BTバランス最適）
TRI_TOTAL = 1000.0     # 1レース総額


def trifecta_probs_pl(power):
    """{frame:P(1着)} → 120通り {(i,j,k):P}（Plackett-Luce, 正規化）。"""
    out = {}
    for (i, j, k) in _TRI_PERMS:
        pi = power.get(i, 0.0); pj = power.get(j, 0.0); pk = power.get(k, 0.0)
        d1 = 1.0 - pi; d2 = 1.0 - pi - pj
        out[(i, j, k)] = pi * (pj / d1) * (pk / d2) if (d1 > 1e-9 and d2 > 1e-9) else 0.0
    s = sum(out.values()) or 1.0
    return {c: v / s for c, v in out.items()}


def fetch_trifecta_odds(jcd, rno, hd):
    """boatrace.jp から3連単オッズ(120通り)best-effort。失敗時 None。
    2連単(odds2tf)と同じ row-major 構造を3連単(odds3t)へ拡張。"""
    url = f"https://boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={hd}"
    try:
        r = requests.get(url, headers=UA, timeout=15); r.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(r.text, "lxml")
    vals = []
    for c in soup.select("td.oddsPoint"):
        t = c.get_text(strip=True).replace(",", "")
        try:
            vals.append(float(t))
        except ValueError:
            vals.append(None)
    if len(vals) < 120:
        return None
    vals = vals[:120]
    # 各1着(列)の (2着,3着) を昇順で20通り。cellは20行×6列の row-major。
    pairs_by_first = {f: [(j, k) for j in range(1, 7) if j != f
                          for k in range(1, 7) if k != f and k != j] for f in range(1, 7)}
    odds = {}; idx = 0
    for row_i in range(20):
        for first in range(1, 7):
            j, k = pairs_by_first[first][row_i]
            odds[(first, j, k)] = vals[idx]; idx += 1
    return {f"{a}-{b}-{c}": v for (a, b, c), v in odds.items()}


def _tri_stars(conf):
    if conf >= 0.28: return 5
    if conf >= 0.23: return 4
    if conf >= 0.18: return 3
    if conf >= 0.13: return 2
    return 1


def _tri_reason(boats, power, top):
    bynum = {b["frame"]: b for b in boats}
    h = max(power.items(), key=lambda kv: kv[1])[0]
    hb = bynum.get(h, {})
    feats = []
    if int(hb.get("course") or h) == 1: feats.append("イン(1コース)")
    if _num(hb.get("nat"), 0) >= 6.0: feats.append(f"全国勝率{hb.get('nat')}")
    if hb.get("st") is not None and _num(hb.get("st"), 9) <= 0.16: feats.append(f"ST{hb.get('st')}")
    if hb.get("ex") is not None: feats.append(f"展示{hb.get('ex')}")
    seconds = sorted({c[1] for c, _ in top}); thirds = sorted({c[2] for c, _ in top})
    return (f"{h}号艇を本命（{'・'.join(feats) if feats else '総合評価が最上位'}）。"
            f"2着候補は{'・'.join(map(str, seconds))}号艇、3着候補は{'・'.join(map(str, thirds))}号艇。"
            f"上位の的中確率が近いため{len(top)}点に分散。自信度が基準を超えたレースのみ購入(それ以外は見送り)。")


def trifecta_pick(boats, odds_map):
    power = predict_power(boats)
    conf, gap = _conf_gap(power)
    honmei = max(power.items(), key=lambda kv: kv[1])[0]
    ranked = sorted(trifecta_probs_pl(power).items(), key=lambda kv: kv[1], reverse=True)
    if conf < TRI_CONF_GATE:
        return {"decision": "見送り", "stars": _tri_stars(conf), "conf": round(conf, 4),
                "gap": round(gap, 4), "honmei": honmei, "picks": [], "total": 0,
                "reason": f"自信度{conf*100:.1f}%が基準{TRI_CONF_GATE*100:.0f}%未満。的中率重視は本命が堅いレースだけ買う設計のため見送り。"}
    each = max(100, round(TRI_TOTAL / TRI_N / 100) * 100)   # 200円
    top = ranked[:TRI_N]
    picks = []; hit_sum = 0.0
    for (combo, p) in top:
        cs = f"{combo[0]}-{combo[1]}-{combo[2]}"
        o = odds_map.get(cs) if odds_map else None
        picks.append({"combo": cs, "P": round(p, 4), "hit_pct": round(p * 100, 1),
                      "odds": o, "stake": each, "payout": (int(round(each * o)) if o else None)})
        hit_sum += p
    return {"decision": "買い", "stars": _tri_stars(conf), "conf": round(conf, 4), "gap": round(gap, 4),
            "honmei": honmei, "picks": picks, "total": each * len(picks),
            "hit_sum_pct": round(hit_sum * 100, 1), "reason": _tri_reason(boats, power, top)}


# ---- 中オッズEV（記録のみの影バケット・2026-09-07 追加）----
# ユーザ着眼「20〜40倍くらいの中オッズなら結構当たるのでは」を前向きに検証する枠。
# 過去249日(38,512R)の実測では、中オッズ狙いそのものには妙味が無い:
#   モデルP帯別の回収率(全120組を1点買い) = P<0.5%(大穴):55.6% / 1-2%(≒70倍):73.4%
#   / 2-3%(≒40倍):74.1% / 3-5%(≒26倍):77.3% / 5-8%(≒16倍):76.2% / 12%+(≒7倍):86.8%
#   ＝控除率25%の壁(75%)がほぼそのまま出る。的中率が上がるぶん配当が下がるだけで回収率は動かない。
#   人気順の帯で買っても 1位81.5% / 3-8位76.5% / 11-20位72.5% と全部100%割れ。
#   唯一はっきりした歪みは「大穴が割高・本命が割安」で、方向はむしろ中オッズより本命寄り。
# ただし上記は全て「オッズを見ずに選ぶ」検証。過去に3連単オッズ盤が存在しないため、
# 「市場オッズがモデルより高い組だけ買う(EV)」は歴史的に検証不能だった＝唯一の未検証軸。
# ライブ10日の断片(的中率重視5点のオッズ記録3,361組)では
#   10〜50倍かつEV≥1.5 → 回収率102.7%(n=814・的中38) と出たが、的中38本では完全にノイズ域。
# よってここでは実弾に入れず、影バケットとして毎日記録して母数を貯める。
_MIDEV_LO, _MIDEV_HI = 10.0, 50.0   # 買う市場オッズの帯
_MIDEV_EV = 1.5                     # 期待値フロア(較正後P × オッズ)
_MIDEV_N = 5                        # 最大点数
# v91(2026-09-20 藤田指示「1点千円」): 200 → 1000。
#   実際に賭けている額は1点1,000円で、コードの200円は実態と5倍ずれていた。
#   画面の投資/払戻/収支が実際の1/5で表示されていたのを是正する(回収率・的中率は比率なので不変)。
#   ※v89として一度用意したが GitHub へ未反映だったため、v90(オッズ板保存)の上に再適用。
#   ⚠ _MIDEV_DAILY_MAX(1日10レース上限)は _MIDEV_DISPLAY 経路では効いていない
#     (_midev_today_block_disp は「上限なし・締切順」、_cpk も live を見ない)。
#     現行の gap>=0.5 は1日9〜25レース(平均12.8)なので、1日9,000〜25,000円になる。
_MIDEV_UNIT = 1000                  # 1点あたりの実際の賭け金
# 1日に実弾で買う上限レース数。これが無いと危険:
#   ライブ実測では EV≥1.5 の条件は「判定レースの約半分」で成立する。120通り全部から探す本番では
#   1日50〜70レース＝1日3〜5万円規模になり、まだ勝ちが証明されていない賭けに月100万円級を投じることになる。
#   そこで「その日の早い者勝ちで上限レースまでを実弾、それ以降は記録のみ」にして、
#   1日の上限を _MIDEV_DAILY_MAX × _MIDEV_N × _MIDEV_UNIT = 10,000円 に固定する。
#   上限を超えた分も台帳には残るので、検証用の母数は減らない。
_MIDEV_DAILY_MAX = 10

# モデルP の較正テーブル（2026-09-07・249日/4,621,440組で実測）。
#   trifecta_probs_pl が出す確率は実際の的中率より 11〜18% 高く出る（過大申告）:
#     P 1-2%帯 実測/予測=0.885 / 2-3%:0.872 / 3-4%:0.926 / 4-5%:0.912 / 5-6%:0.881
#     / 6-8%:0.883 / 8-10%:0.890 / 10-15%:0.864 / 15%+:0.816
#   EV = P×オッズ なので、較正しないと期待値も同じ倍率で水増しされる（EV1.5 は実質1.3程度）。
#   ここでは較正後のPで判定し、生のEVも併記して後から比較できるようにする。
_P_CALIB = ((0.005, 1.00), (0.01, 1.00), (0.02, 0.885), (0.03, 0.872), (0.04, 0.926),
            (0.05, 0.912), (0.06, 0.881), (0.08, 0.883), (0.10, 0.890), (0.15, 0.864))
_P_CALIB_TAIL = 0.816


def calib_p(p):
    """モデルの3連単確率を実測ベースに引き戻す。"""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 0.0
    for hi, mul in _P_CALIB:
        if p <= hi:
            return p * mul
    return p * _P_CALIB_TAIL


def midev_pick(boats, odds_map):
    """中オッズ(10〜50倍)かつモデル期待値の高い組を最大5点。記録専用・実弾ではない。"""
    if not odds_map:
        return {"decision": "見送り", "picks": [], "total": 0, "reason": "オッズ未取得"}
    power = predict_power(boats)
    tri = trifecta_probs_pl(power)
    cands = []
    for ijk, p in tri.items():
        cs = f"{ijk[0]}-{ijk[1]}-{ijk[2]}"
        o = odds_map.get(cs)
        if not o:
            continue
        try:
            o = float(o)
        except (TypeError, ValueError):
            continue
        if not (_MIDEV_LO <= o <= _MIDEV_HI):
            continue
        pc = calib_p(p)
        ev = pc * o                 # 較正後の期待値（判定に使う）
        ev_raw = p * o              # 生のモデルEV（比較用に記録）
        if ev < _MIDEV_EV:
            continue
        cands.append({"combo": cs, "P": round(p, 4), "Pc": round(pc, 4), "odds": o,
                      "ev": round(ev, 2), "ev_raw": round(ev_raw, 2), "stake": _MIDEV_UNIT})
    cands.sort(key=lambda x: x["ev"], reverse=True)
    picks = cands[:_MIDEV_N]
    if not picks:
        return {"decision": "見送り", "picks": [], "total": 0,
                "reason": f"{int(_MIDEV_LO)}〜{int(_MIDEV_HI)}倍で較正後EV≥{_MIDEV_EV}の組が無いため見送り。"}
    return {"decision": "買い", "picks": picks, "total": _MIDEV_UNIT * len(picks),
            "reason": (f"{int(_MIDEV_LO)}〜{int(_MIDEV_HI)}倍かつ較正後EV≥{_MIDEV_EV}の組を"
                       f"EV上位{len(picks)}点。記録のみ(実弾ではない)。")}


# v78(2026-09-18 藤田指示「並行して検証したい」): D案＝回収率が一番良い帯として見つけた
#   「市場オッズ10〜15倍 × 較正EV≥1.2 の較正EV上位3点」を C案と並行で記録・精算する。1点200円。
#   検証(2026 1-9月・ポセイドン最終オッズ 29日): ROI約103%・大穴3本抜き93.6%・的中率約9%・最大連敗約41。
#     C案midev(10-50倍)より回収率・変動・連敗すべて上。ただし大穴3本抜き93.6%＝まだ100%未満＝利益は未実証。
#     詳細は project doc [[boat-race-bestroi-lowmid-0918]]。前向きに実配当で確認する枠。
_DROI_LO, _DROI_HI = 10.0, 15.0
_DROI_EV = 1.2
_DROI_N = 3
_DROI_UNIT = 200


def bestroi_pick(boats, odds_map):
    """D案(並行検証): 市場オッズ10〜15倍 かつ 較正EV(=calib_p×オッズ)≥1.2 の較正EV上位3点。1点200円。"""
    if not odds_map:
        return {"decision": "見送り", "picks": [], "total": 0, "reason": "オッズ未取得"}
    power = predict_power(boats)
    tri = trifecta_probs_pl(power)
    cands = []
    for ijk, p in tri.items():
        cs = f"{ijk[0]}-{ijk[1]}-{ijk[2]}"
        o = odds_map.get(cs)
        if not o:
            continue
        try:
            o = float(o)
        except (TypeError, ValueError):
            continue
        if not (_DROI_LO <= o <= _DROI_HI):
            continue
        pc = calib_p(p)
        ev = pc * o
        if ev < _DROI_EV:
            continue
        cands.append({"combo": cs, "P": round(p, 4), "Pc": round(pc, 4), "odds": o,
                      "ev": round(ev, 2), "stake": _DROI_UNIT})
    cands.sort(key=lambda x: x["ev"], reverse=True)
    picks = cands[:_DROI_N]
    if not picks:
        return {"decision": "見送り", "picks": [], "total": 0,
                "reason": f"{int(_DROI_LO)}〜{int(_DROI_HI)}倍で較正EV≥{_DROI_EV}の組が無いため見送り。"}
    return {"decision": "買い", "picks": picks, "total": _DROI_UNIT * len(picks),
            "reason": (f"{int(_DROI_LO)}〜{int(_DROI_HI)}倍で較正EV≥{_DROI_EV}の較正EV上位{len(picks)}点(D案・1点{_DROI_UNIT}円)。")}


# v69(2026-09-16 ユーザ指示): C案(中オッズEV)の「厳選」候補を影バケットとして記録する。お金は入れない。
#   ライブ9/08〜9/15(721R・1,328点)の後付け分析で良く見えた2条件。ただし8月の別5日(535R・本番ロジック再現)
#   では 「EV1位＋EV≥2.0」174.5%→81.8% / 「候補2〜3点」113.0%→104.0% と崩れる or 弱まる＝まだ偽物と区別できない。
#   よって実弾C案(早い者順10R・全点)は変更せず、同じ候補から派生させた集計だけを並走させる。
#   派生のみ＝新しい記録項目は不要で、midev が記録された 9/08 以降の全レースに遡って集計できる。
#   昇格の目安: 100レース超 かつ 大穴3本抜き90%超 かつ 過去オッズ(ポセイドン最終オッズ)の別期間でも100%超。
_MIDEV_SHADOWS = (
    ("midev_ev1ev2", "EV1位の1点だけ＋EV≥2.0"),
    ("midev_n23", "候補が2〜3点のレースだけ（全点）"),
    ("midev_sel", "EV1位＋EV≥2.0＋オッズ≥30倍＋候補2組以上"),
    # v82(2026-09-19): 現行selのゲートを1つずつ外して測り直した結果の3本。記録のみ。
    #   ライブ990R(9/08〜9/18)で「オッズ≥30」は切る側が負けておらず、外したほうが
    #   レース数279(+67)・的中率6.1%(+0.4pt)・95%下限137%(+8pt)・9/16〜18も163%(vs146%)と全項目で上。
    #   代わりに効いたのは「EV1位と2位の差」で、差<0.5を切る側は大穴3本抜き63.0%と明確に負け。
    ("midev_ev2n2", "EV1位＋EV≥2.0＋候補2組以上（オッズ条件なし）"),
    ("midev_ev2h1", "同＋頭が1号艇"),
    ("midev_ev2gap", "EV1位＋EV≥2.0＋候補2組以上＋EV1位と2位の差≥0.5"),
)
_MIDEV_SH_GAP = 0.5

# v84(2026-09-19 藤田指示「A案の5連敗ストップを外して」):
#   A案の「その日5連敗したらその日は終了」を無効化し、安定型フラットを実弾にする。
#   根拠(ライブ台帳 8/28〜9/18・安定型118本を4点均等1,000円で再精算):
#     各日の中で「結果の並び順だけ」をシャッフルして同じルールを当てる検定(各2,000回)で、
#     3連敗 p=0.55 / 4連敗 p=0.84 / 5連敗 p=0.64 / 6連敗 p=0.71 ＝ どれも偶然と区別できない。
#     実収支も ストップ無し +176,000円 → 5連敗ストップ +150,300円 と減っていた。
#     そもそも「1日の的中本数のばらつき」も偶然と区別できず(p=0.15)、連敗がかたまる証拠が無い。
#   過去BT(7か月)では効いて見えたルールだが、ライブでは再現しなかった。
#   None=ストップ無し。数字(5など)を入れれば元に戻る。集計側(uv_stable_stop)は記録として残す。
_A_STOP_N = None
_MIDEV_SH_EV = 2.0
# v72(2026-09-16 ユーザ指示「今日から買いで表示」): EV1位＋EV≥2.0 をさらに「レース前に分かる特徴」で絞る。
#   ライブ9/08〜15(278R)と別期間8/20〜9/05の17日(658R・ポセイドン最終オッズで本番 midev_pick 再生成)の両方で
#   切る側が一貫して負けていた2条件を採用:
#     オッズ<30倍: ライブ65.5% / 別期間68.3%   候補(EV≥1.5)が1組だけ: 108.5% / 49.2%
#   採用後: ライブ153R 238.0%(大穴3本抜き123.2%) / 別期間368R 124.8%(88.0%)・別期間1日抜き17/17日改善。
#   ただし的中は各10〜16本、別期間の大穴3本抜きは90%未満＝勝ち証明ではない。24通り試した中から選んだ点にも注意。
#   ポセイドン本命との一致は外部データ(前夜GitHub保存)に依存するので買い条件には入れず「◎印＋記録」に留める。
_MIDEV_SEL_ODDS = 30.0
# v71(2026-09-16 ユーザ指示): 画面のC案は「EV1位＋EV≥2.0」の1点だけを表示する。
#   記録(uv_json.midev の全候補・1日10R上限の live フラグ)は従来どおり全部残す＝元条件との比較検証は継続。
#   None に戻せば v70 以前の表示(全点・早い者順10R)に戻る。
# v83(2026-09-19 藤田指示「切り替えて」): "midev_sel" から「オッズ≥30倍」条件を外した
#   "midev_ev2n2" に変更。ライブ990R(9/08〜9/18)で測り直したところ、オッズ≥30で切っている側が
#   負けておらず、外すと レース数 212→279 / 的中率 5.7→6.1% / 大穴3本抜き 164.4→165.2% /
#   日単位ブートストラップ95%下限 129→137% / v72デプロイ後の9/16〜18も 145.6→163.1% と全項目で上。
#   条件を「緩める」方向＝選択バイアスの向きが逆なので、絞り込み案(ev2h1/ev2gap)より安全と判断。
#   絞り込み案2本は記録のみで並走中。戻すときは "midev_sel" に書き戻すだけ。
# v88(2026-09-20 藤田指示「今すぐ切り替えて」): "midev_ev2n2" → "midev_ev2gap"。
#   EV1位と2位の差(gap)≥0.5 を追加。ライブ midev 1,077R(9/08〜9/19)で測定:
#     買い 307R→147R / 的中率 5.5%→6.8% / 大穴3本抜き 150.2%→168.0% /
#     上位3日除外 121.5%→142.6% / 日ブロック95%下限 129%→157%
#   切られる側(gap<0.5, 160R・的中7)は 大穴3本抜き55.9%・上位3日除外59.0% と明確に負け＝
#   「良いものを選ぶ」ではなく「悪いものを捨てる」側に効果が出ている。
#   閾値を0.2〜0.6で動かしても的中率・ROI・抜3が単調に改善(0.8以上は抜3が崩れるので採らない)。
#   前半6日300% / 後半6日241% と両期間で一貫。ただし的中10本・12日のみ＝96条件試した中の1つでもある。
#   1日の買いは約26R→約12R、投資は約5,200円→約2,400円。戻すときは "midev_ev2n2" に書き戻すだけ。
_MIDEV_DISPLAY = "midev_ev2gap"


def midev_shadow_picks(mv, name):
    """中オッズEVの買い目(mv)から影バケットの買い目を派生させる。対象外なら[]。"""
    pk = (mv or {}).get("picks") or []
    if not pk:
        return []
    if name == "midev_ev1ev2":
        top = max(pk, key=lambda p: _fnum(p.get("ev")))   # picks はEV降順だが念のためmax
        return [top] if _fnum(top.get("ev")) >= _MIDEV_SH_EV else []
    if name == "midev_n23":
        return list(pk) if len(pk) in (2, 3) else []
    if name == "midev_sel":
        top = max(pk, key=lambda p: _fnum(p.get("ev")))
        ok = (_fnum(top.get("ev")) >= _MIDEV_SH_EV and _fnum(top.get("odds")) >= _MIDEV_SEL_ODDS and len(pk) >= 2)
        return [top] if ok else []
    if name in ("midev_ev2n2", "midev_ev2h1", "midev_ev2gap"):
        if len(pk) < 2:
            return []
        ev_sorted = sorted(pk, key=lambda p: _fnum(p.get("ev")), reverse=True)
        top = ev_sorted[0]
        if _fnum(top.get("ev")) < _MIDEV_SH_EV:
            return []
        if name == "midev_ev2h1":
            head = str(top.get("combo") or "-").split("-")[0]
            if head != "1":
                return []
        if name == "midev_ev2gap":
            if _fnum(top.get("ev")) - _fnum(ev_sorted[1].get("ev")) < _MIDEV_SH_GAP:
                return []
        return [top]
    return []


_POS_HEAD_CACHE = {}


def poseidon_head(hd, jcd, rno):
    """ポセイドン(前夜取得)のAI予想で確率1位の組の1着艇。取れなければ None。
    過去日は1回取れたら保持、当日分は取れなければ10分後に再取得。"""
    hd = str(hd)
    ent = _POS_HEAD_CACHE.get(hd)
    now_t = time.time()
    if ent is None or (ent[0] is None and now_t - ent[1] > 600):
        heads = None
        js = _gh_json("poseidon/%s.json" % hd)
        if isinstance(js, dict) and js:
            heads = {}
            for k, v in js.items():
                if not isinstance(v, dict) or not v:
                    continue
                best = max(v.items(), key=lambda kv: _fnum(kv[1][0] if isinstance(kv[1], list) else kv[1]))
                heads[k] = str(best[0])[0]
        ent = (heads, now_t)
        _POS_HEAD_CACHE[hd] = ent
    heads = ent[0]
    if not heads:
        return None
    try:
        return heads.get("%d_%d" % (int(float(jcd)), int(float(rno))))
    except (TypeError, ValueError):
        return None


def midev_pos_agree(r, pick):
    """厳選の1点の1着艇が、ポセイドン本命(確率1位の組)の1着艇と同じか。True/False/None(データ無し)。"""
    h = poseidon_head(r.get("date"), r.get("jcd"), r.get("rno"))
    if h is None or not pick:
        return None
    return str(pick.get("combo", ""))[:1] == h


def midev_shadow_result(r, mv, name):
    """影バケットの (買い目, 投資, 払戻, 精算済か)。払戻は本体midevの精算結果から派生(的中は最大1点)。"""
    sub = midev_shadow_picks(mv, name)
    if not sub:
        return [], 0.0, 0.0, False
    stake = sum(_fnum(p.get("stake")) for p in sub)
    settled = (r.get("status") == "settled" and str(mv.get("hit", "")) != "" and bool(r.get("win_combo")))
    ret = 0.0
    if settled and str(mv.get("hit")) == "1":
        wc = str(r.get("win_combo"))
        if any(p.get("combo") == wc for p in sub):
            ret = _fnum(mv.get("ret"))
    return sub, stake, ret, settled


@app.get("/api/trifecta")
def api_trifecta(jcd: int = Query(...), rno: int = Query(...), hd: str = Query(...)):
    try:
        data = fetch_openapi(hd)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"openapi取得失敗: {e}"}, status_code=502)
    race = find_race(data, jcd, rno)
    if race is None:
        return JSONResponse({"ok": False, "error": "該当レースが見つかりません"}, status_code=404)
    boats = extract_boats(race)
    odds_map = fetch_trifecta_odds(jcd, rno, hd)
    res = trifecta_pick(boats, odds_map)
    res.update({"ok": True, "jcd": jcd, "rno": rno, "hd": hd,
                "odds_status": "ライブ（boatrace.jp）" if odds_map else "オッズ未取得（購入時に公式で最終確認）"})
    return res


_TRIFECTA_HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3連単 的中率重視</title>
<style>
:root{--bg:#f4f6f9;--card:#fff;--line:#dbe2ea;--buy:#1c7a38;--skip:#8a97a5;--ink:#1a2330;--sub:#5c6b7a;--accent:#b8860b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Hiragino Kaku Gothic ProN",sans-serif;padding:16px;max-width:820px;margin:0 auto}
h1{font-size:22px;margin:.2em 0}.sub{color:var(--sub);font-size:13px;line-height:1.6}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.bar a,.btn{display:inline-block;padding:8px 12px;border-radius:10px;background:#eef2f7;color:#22303f;text-decoration:none;font-size:13px;border:1px solid var(--line);cursor:pointer}
.btn{background:var(--accent);color:#fff;border:none;font-weight:700}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin-top:14px}
label{font-size:12px;color:var(--sub);margin-right:4px}input,select{padding:8px;border:1px solid var(--line);border-radius:8px;font-size:15px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:end}
.verdict{font-size:20px;font-weight:800;margin:2px 0}.buy{color:var(--buy)}.skip{color:var(--skip)}
.stars{color:var(--accent);font-size:18px;letter-spacing:2px}
table{width:100%;border-collapse:collapse;margin-top:8px}th,td{border-bottom:1px solid var(--line);padding:8px;font-size:14px;text-align:center}
th{color:var(--sub);font-weight:600;font-size:12px}.combo{font-weight:800;font-size:16px}
.tsub{color:var(--sub);font-size:12px}.big{font-size:17px;font-weight:800}
.note{background:#fff8e6;border:1px solid #f0e2b8;border-radius:10px;padding:10px;font-size:12.5px;color:#6b5a1f;margin-top:10px}
</style></head><body>
<h1>&#127919; 3連単 的中率重視</h1>
<div class="sub">自信度の高いレースだけを買い、それ以外は<b>見送り</b>。半年超BTの最適設定＝<b>自信度上位約5%・5点・総額1000円</b>を適用。
<br>順位付けは本サイト共通の焼き込みモデル→Plackett-Luceで120通り。BT実績は的中≒48%/回収≒83%（研究用GBDTでの数値、ライブは近似）。</div>
<div class="bar"><a href="/final">&#128337; 締切5分前 自動判定</a><a href="/undervalue">&#127775; 過小評価キー</a><a href="/today">今日の高オッズEV</a><a href="/dashboard">成績</a><a href="/highodds">2連単 手動判定</a></div>
<div class="card">
  <div class="row">
    <div><label>場</label><br><select id="jcd">
      <option value="1">桐生</option><option value="2">戸田</option><option value="3">江戸川</option><option value="4">平和島</option>
      <option value="5">多摩川</option><option value="6">浜名湖</option><option value="7">蒲郡</option><option value="8">常滑</option>
      <option value="9">津</option><option value="10">三国</option><option value="11">びわこ</option><option value="12">住之江</option>
      <option value="13">尼崎</option><option value="14">鳴門</option><option value="15">丸亀</option><option value="16">児島</option>
      <option value="17">宮島</option><option value="18">徳山</option><option value="19">下関</option><option value="20">若松</option>
      <option value="21">芦屋</option><option value="22">福岡</option><option value="23">唐津</option><option value="24">大村</option></select></div>
    <div><label>R</label><br><input id="rno" type="number" min="1" max="12" value="1" style="width:70px"></div>
    <div><label>日付</label><br><input id="hd" type="date"></div>
    <div><button class="btn" onclick="go()">判定する</button></div>
  </div>
  <div class="tsub" style="margin-top:8px">締切直前に押すほどオッズが最終値に近づきます。</div>
</div>
<div id="out"></div>
<div class="note">⚠ 控除率25%の壁は不変で、これは勝ちを保証しません（BTでも回収率は100%未満）。「当てにいく」用途で見送りを効かせ、通算成績を前向きに記録して検証する道具です。オッズは購入直前に公式で最終確認を。</div>
<script>
const $=id=>document.getElementById(id);
const jst=new Date(Date.now()+9*3600*1000);$("hd").value=jst.toISOString().slice(0,10);
function stars(n){return "★".repeat(n)+"☆".repeat(5-n);}
async function go(){
  $("out").innerHTML='<div class="card">判定中…</div>';
  const hd=$("hd").value.replaceAll("-","");
  const q=new URLSearchParams({jcd:$("jcd").value,rno:$("rno").value,hd:hd});
  try{
    const j=await (await fetch("/api/trifecta?"+q.toString())).json();
    if(!j.ok){$("out").innerHTML='<div class="card">取得できませんでした：'+(j.error||"")+'</div>';return;}
    let h='<div class="card">';
    if(j.decision==="買い"){
      h+='<div class="verdict buy">買い</div>';
    }else{
      h+='<div class="verdict skip">見送り</div>';
    }
    h+='<div class="stars">'+stars(j.stars)+' <span class="tsub">自信度'+(j.conf*100).toFixed(1)+'%</span></div>';
    h+='<div class="sub" style="margin-top:6px">本命：<b>'+j.honmei+'号艇</b>　'+j.reason+'</div>';
    if(j.decision==="買い"){
      h+='<table><tr><th>買い目</th><th>推定的中率</th><th>現在オッズ</th><th>購入金額</th><th>的中時の払戻</th></tr>';
      for(const p of j.picks){
        h+='<tr><td class="combo">'+p.combo+'</td><td>'+p.hit_pct+'%</td><td>'+(p.odds!=null?p.odds+'倍':'<span class=tsub>—</span>')+'</td><td>'+p.stake+'円</td><td>'+(p.payout!=null?p.payout.toLocaleString()+'円':'<span class=tsub>要確認</span>')+'</td></tr>';
      }
      h+='</table>';
      h+='<div class="sub" style="margin-top:8px">合計購入金額：<span class="big">'+j.total.toLocaleString()+'円</span>　'
        +'この5点で当たる推定確率：<span class="big">'+j.hit_sum_pct+'%</span>　'
        +'<span class="tsub">オッズ取得：'+j.odds_status+'</span></div>';
    }
    h+='</div>';
    $("out").innerHTML=h;
  }catch(e){$("out").innerHTML='<div class="card">エラー：'+e+'</div>';}
}
</script></body></html>"""


@app.get("/trifecta", response_class=HTMLResponse)
def trifecta_page():
    return _TRIFECTA_HTML


# ==================== 締切5分前 最終判定カード（/final）====================
@app.get("/final", response_class=HTMLResponse)
def final_page():
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    hd = now.strftime("%Y%m%d")
    rows = tri_load_ledger()   # Render常駐スケジューラのローカル台帳のみ（過小評価キー入り）
    today = [r for r in rows if r.get("date") == hd]

    def ven(j):
        try:
            return _VEN.get(int(j), j)
        except Exception:
            return j

    def fnum(x, d=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return d

    # ---- 当日成績: 実運用(買い) と 研究(見送りを買っていたら) を分離 ----
    buys = [r for r in today if r.get("decision") == "買い"]
    skips = [r for r in today if r.get("decision") == "見送り"]
    b_settled = [r for r in buys if r.get("status") == "settled"]
    b_hits = sum(1 for r in b_settled if str(r.get("hit")) == "1")
    b_stake = sum(fnum(r.get("total")) for r in b_settled)
    b_ret = sum(fnum(r.get("ret")) for r in b_settled)
    s_settled = [r for r in skips if r.get("status") == "settled"]
    s_hits = sum(1 for r in s_settled if str(r.get("shadow_hit")) == "1")

    # ---- 過小評価キー 当日成績（的中率重視とは別集計）----
    def _uvj(r):
        try:
            return json.loads(r.get("uv_json") or "{}")
        except Exception:
            return {}
    uv_today = [(r, _uvj(r)) for r in today]

    def _uv_stats(field):
        buys = [(r, u) for r, u in uv_today
                if _uv_dec_or_decision(u, field) == "買い" and fnum(u.get("total")) > 0]
        bset = [(r, u) for r, u in buys if r.get("status") == "settled"]
        hits = sum(1 for r, u in bset if str(r.get("uv_hit")) == "1")
        stake = sum(fnum(u.get("total")) for r, u in bset)
        ret = sum(fnum(r.get("uv_ret")) for r, u in bset)
        skips = sum(1 for r, u in uv_today if _uv_dec_or_decision(u, field) == "見送り")
        roi = (ret / stake * 100) if stake else 0
        return {"buy": len(buys), "hits": hits, "bset": len(bset), "stake": stake, "ret": ret, "skip": skips, "roi": roi}
    def _uv_stats_stop(field="dec_stable", N=5):
        # A案: 当日の安定型を5連敗ストップ適用で集計(時刻順)
        bl = [(r, u) for r, u in uv_today if u.get(field) == "買い" and r.get("status") == "settled"]
        bl.sort(key=lambda ru: str(ru[0].get("closed_at") or ru[0].get("ts") or ""))
        buy = hits = 0; stake = 0.0; ret = 0.0; run = 0; stopped = 0
        for r, u in bl:
            if run >= N:
                stopped += 1; continue
            buy += 1; stake += fnum(u.get("total"))
            h = 1 if str(r.get("uv_hit")) == "1" else 0
            hits += h; ret += fnum(r.get("uv_ret"))
            run = 0 if h else run + 1
        return {"buy": buy, "hits": hits, "bset": buy, "stake": stake, "ret": ret,
                "skip": stopped, "roi": (ret / stake * 100) if stake else 0}
    def _a_stopped_keys(field="dec_stable", N=5):
        """v73: A案(安定型＋5連敗ストップ)で当日ストップにより買わなかったレースの (ts,jcd,rno) 集合。"""
        # v76: 未精算(判定済み・結果待ち)のレースも、その時点で5連敗していれば「ストップ」扱いにする
        #      (カードの「買い」表示を実際に買うかどうかと一致させるため)。連敗数は精算済みだけで数える。
        bl = [(r, u) for r, u in uv_today if u.get(field) == "買い"]
        bl.sort(key=lambda ru: str(ru[0].get("closed_at") or ru[0].get("ts") or ""))
        run = 0; out = set()
        for r, u in bl:
            if run >= N:
                out.add((r.get("date"), str(r.get("jcd")), str(r.get("rno")))); continue
            if r.get("status") == "settled":
                run = 0 if str(r.get("uv_hit")) == "1" else run + 1
        return out
    _A_STOPPED = _a_stopped_keys(N=_A_STOP_N) if _A_STOP_N else set()

    def _midev_stats(rs, live_only=False, shadow=None):
        """中オッズEVの集計。live_only=True なら実弾(1日上限内)だけ。
        shadow=影バケット名 なら、その条件で派生させた買い目で集計(上限に関係なく全件)。"""
        if shadow:
            buy = bset = hits = skips = 0; stake = ret = 0.0
            for r in rs:
                try:
                    u = json.loads(r.get("uv_json") or "{}")
                except Exception:
                    continue
                mv = (u or {}).get("midev") or {}
                if not mv:
                    continue
                sub, s_, r_, st_ = midev_shadow_result(r, mv, shadow)
                if not sub:
                    skips += 1; continue
                buy += 1
                if st_:
                    bset += 1; stake += s_; ret += r_
                    hits += 1 if r_ > 0 else 0
            return {"buy": buy, "hits": hits, "bset": bset, "stake": stake,
                    "ret": ret, "skip": skips, "roi": (ret / stake * 100) if stake else 0}
        buys = []; skips = 0
        for r in rs:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                continue
            mv = (u or {}).get("midev") or {}
            if mv.get("picks"):
                if live_only and not mv.get("live"):
                    continue
                buys.append((r, mv))
            elif mv:
                skips += 1
        bset = [(r, mv) for r, mv in buys
                if r.get("status") == "settled" and str(mv.get("hit", "")) != ""]
        hits = sum(1 for _, mv in bset if str(mv.get("hit")) == "1")
        stake = sum(fnum(mv.get("total")) for _, mv in bset)
        ret = sum(fnum(mv.get("ret")) for _, mv in bset)
        return {"buy": len(buys), "hits": hits, "bset": len(bset), "stake": stake,
                "ret": ret, "skip": skips, "roi": (ret / stake * 100) if stake else 0}
    uv_midev = _midev_stats(today, live_only=True)
    uv_midev_all = _midev_stats(today)
    uv_midev_cum = _midev_stats(rows, live_only=True)
    uv_midev_cum_all = _midev_stats(rows)
    uv_midev_sh = {nm: _midev_stats(today, shadow=nm) for nm, _ in _MIDEV_SHADOWS}
    uv_midev_all = _midev_stats(today)
    uv_midev_disp_cum = _midev_stats(rows, shadow=_MIDEV_DISPLAY) if _MIDEV_DISPLAY else None

    def _midev_pos_stats(rs, agree_val=True):
        """表示中の厳選のうち、ポセイドン本命と一致(agree_val=True)した分だけの集計(記録のみ)。"""
        buy = bset = hits = 0; stake = ret = 0.0; nodata = 0
        for r in rs:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                continue
            mv = (u or {}).get("midev") or {}
            sub, s_, r_, st_ = midev_shadow_result(r, mv, _MIDEV_DISPLAY)
            if not sub:
                continue
            ag = midev_pos_agree(r, sub[0])
            if ag is None:
                nodata += 1; continue
            if ag != agree_val:
                continue
            buy += 1
            if st_:
                bset += 1; stake += s_; ret += r_; hits += 1 if r_ > 0 else 0
        return {"buy": buy, "bset": bset, "hits": hits, "stake": stake, "ret": ret, "nodata": nodata,
                "roi": (ret / stake * 100) if stake else 0}
    uv40 = _uv_stats("dec40")
    uv55 = _uv_stats("dec55")
    uv_stable = _uv_stats("dec_stable")
    uv_stable_stop = _uv_stats_stop(N=(_A_STOP_N or 5))   # v84: 実弾から外した後も比較用に集計は続ける
    uv40_ex = _uv_stats("dec40_ex")
    uv40_calm = _uv_stats("dec40_calm")
    uv40_ev12 = _uv_stats("dec40_ev12")
    uv40_ev13 = _uv_stats("dec40_ev13")
    uv40_lane = _uv_stats("dec40_lane")
    uv40_stable = _uv_stats("dec40_stable")
    uv40_in = _uv_stats("dec40_in")
    uv40_in_ev = _uv_stats("dec40_in_ev")
    uv_in_ev12 = _uv_stats("dec_in_ev12")

    def _uvp_stats(name):
        buy = hits = 0; stake = ret = 0.0
        for r, u in uv_today:
            if r.get("status") != "settled":
                continue
            k, s_, r_ = uvp_result(r, u, name)
            if not k:
                continue
            buy += 1; stake += s_; ret += r_
            hits += 1 if r_ > 0 else 0
        return {"buy": buy, "hits": hits, "bset": buy, "stake": stake, "ret": ret,
                "skip": 0, "roi": (ret / stake * 100) if stake else 0}
    uvp = {nm: _uvp_stats(nm) for nm, _, _ in _UVP_FILTERS}

    css = _DASH_CSS + (
        ".fcard{background:#fff;border:1px solid #dbe2ea;border-radius:14px;padding:12px 14px;margin-top:12px}"
        ".fhead{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}"
        ".big{font-size:20px;font-weight:800}.buy{color:#1c7a38}.skip{color:#8a97a5}"
        ".stars{color:#b8860b;font-size:16px}.tsub{color:#5c6b7a;font-size:12px}"
        ".ct{font-weight:700}.combo{font-weight:700}"
        ".kpi{display:flex;gap:14px;flex-wrap:wrap}.kpi div{background:#fff;border:1px solid #dbe2ea;border-radius:12px;padding:10px 14px;min-width:120px}"
        ".kpi b{font-size:19px}.res-hit{color:#1c7a38;font-weight:800}.res-miss{color:#b23b3b}")
    head = ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="refresh" content="60">'
            '<title>締切5分前 最終判定</title><style>%s</style></head><body>'
            '<h1>&#128337; 締切5分前 自動判定（A案・B案・C案）</h1>' % css)
    bar = ('<div class="bar"><a href="/final">&#8635; 最新に更新</a>'
           '<a class="g" href="/trifecta">手動で1レース判定</a>'
           '<a class="g" href="/today">今日の高オッズEV</a>'
           '<a class="g" href="/dashboard">成績</a></div>')

    b_roi = (b_ret / b_stake * 100) if b_stake else 0
    kpi = ('<div class="kpi" style="margin-top:12px">'
           '<div class="tsub">別モデル：的中率重視（本命型・5点）<br><b>%d</b> レース購入</div>'
           '<div class="tsub">的中<br><b>%d</b> 本（的中率 %.0f%%）</div>'
           '<div class="tsub">投資 / 払戻<br><b>%s</b> / %s 円</div>'
           '<div class="tsub">回収率<br><b>%.1f%%</b></div>'
           '<div class="tsub">見送り<br><b>%d</b> レース</div>'
           '<div class="tsub">研究: 見送りを買っていたら<br><b>%d</b> 本的中</div>'
           '</div>') % (len(buys), b_hits, (b_hits/len(b_settled)*100 if b_settled else 0),
                        f"{int(b_stake):,}", f"{int(b_ret):,}", b_roi, len(skips), s_hits)

    # v63: 「どれを実際に買うのか」が一目で分かるように、実弾2つと記録専用を分離表示する。
    def _uv_row(label, s, live=False):
        hr = (s["hits"] / s["bset"] * 100) if s["bset"] else 0
        badge = ('<span style="background:#1c7a38;color:#fff;border-radius:6px;padding:1px 7px;'
                 'font-size:11px;font-weight:800;margin-right:6px">実際に買う</span>' if live else
                 '<span style="background:#e8ecf1;color:#5c6b7a;border-radius:6px;padding:1px 7px;'
                 'font-size:11px;font-weight:700;margin-right:6px">記録のみ</span>')
        col = "#1c7a38" if live else "#8a97a5"
        return ('<div class="tsub" style="margin-top:10px;font-weight:700;color:%s">%s%s 当日成績</div>'
                '<div class="kpi" style="margin-top:6px">'
                '<div class="tsub">買い<br><b>%d</b> レース</div>'
                '<div class="tsub">的中<br><b>%d</b> 本（%.0f%%）</div>'
                '<div class="tsub">投資 / 払戻<br><b>%s</b> / %s 円</div>'
                '<div class="tsub">回収率<br><b>%.1f%%</b></div>'
                '<div class="tsub">見送り<br><b>%d</b> レース</div>'
                '</div>') % (col, badge, label, s["buy"], s["hits"], hr,
                             f"{int(s['stake']):,}", f"{int(s['ret']):,}", s["roi"], s["skip"])
    uv_live = ('<div class="fcard" style="border-color:#1c7a38;background:#f4fbf6">'
               '<div style="font-weight:800;color:#1c7a38;font-size:15px">'
               '&#9989; 実際に買う枠（下の各レースのカードで「&#128994; 買い」になるもの）</div>'
               + _uv_row("B案：≥40混戦（過小評価キー）", uv40, live=True)
               + _uv_row("A案：安定型 keyp&ge;10%（フラット・連敗ストップなし）", uv_stable, live=True)
               + (_uv_row("C案：中オッズEV 厳選（%s の1点×%d円・検証中）"
                          % (dict(_MIDEV_SHADOWS)[_MIDEV_DISPLAY], _MIDEV_UNIT), uv_midev_sh[_MIDEV_DISPLAY], live=True)
                  if _MIDEV_DISPLAY else
                  _uv_row("C案：中オッズEV（少額・検証中／5点×%d円・1日%dレースまで）"
                          % (_MIDEV_UNIT, _MIDEV_DAILY_MAX), uv_midev, live=True))
               + '<div class="tsub" style="margin-top:8px">'
                 'A案・B案（過小評価キー）とC案（中オッズEV）を並行してライブ検証中。'
                 'A案とB案は<b>同じ買い目</b>なので、両方に「買い」が出ても買うのは1回分。'
                 '<br>&#9888;&#65039; <b>カードで「&#128994; 買い」になる枠は上の3つだけではありません。</b>'
                 '<b>D案</b>（10〜15倍×較正EV≥1.2の上位3点・1点' + str(_DROI_UNIT) + '円＝最大'
                 + str(_DROI_UNIT * _DROI_N) + '円）と'
                 '<b>波乱≤10%／≤20%</b>（16点×1,000円＝<b>1回16,000円</b>・展示で1号艇3位以下のときのみ）も'
                 '実際に買う枠です。とくに波乱≤20%は1日10件以上出ることがあり、'
                 '<b>金額ではここが最大</b>になります（実測 9/18は17件・179,600円）。'
                 '通算はそれぞれ下の「中オッズEV」「波乱の目」ブロックを見てください。'
                 '<br>下の「記録のみ」は<b>買いません</b>。</div></div>')
    uv_shadow = ('<div class="fcard" style="border-color:#dbe2ea">'
                 '<div style="font-weight:700;color:#5c6b7a;font-size:15px">'
                 '&#128203; 記録のみのバケット（買わない・検証用）　%d件</div>' % (18 if _MIDEV_DISPLAY else 15 + len(_MIDEV_SHADOWS))
                 + _uv_row("≥40混戦＋1号艇展示良好（B案改良案）", uv40_ex)
                 + _uv_row("≥40混戦＋穏やか水面 波&lt;4cm（B案改良案2）", uv40_calm)
                 + _uv_row("≥40混戦＋期待値ev&ge;1.2", uv40_ev12)
                 + _uv_row("≥40混戦＋期待値ev&ge;1.3", uv40_ev13)
                 + _uv_row("≥40混戦＋進入クリーン（キーが外に回されず本命も動かない）", uv40_lane)
                 + _uv_row("A案・B案が重なったレースだけ（≥40混戦＋keyp&ge;10%・買い目はA/Bと同一）", uv40_stable)
                 + _uv_row("B案＋キーが内枠2・3（外枠4-6のキーは的中率が半分）", uv40_in)
                 + _uv_row("B案＋キー内枠＋実弾4点のEV&ge;1.0", uv40_in_ev)
                 + _uv_row("別系統：キー内枠＋実弾4点のEV&ge;1.2（混戦ゲートを使わない）", uv_in_ev12)
                 + "".join(_uv_row("買い目の組み方：%s（A案・B案の買いレースが対象）" % lab, uvp[nm])
                           for nm, lab, _ in _UVP_FILTERS)
                 + _uv_row("≥55（厳選）", uv55)
                 + _uv_row("旧A案：安定型＋5連敗ストップ（v84で外した・比較用）", uv_stable_stop)
                 + ("".join(_uv_row("C案の比較案：%s（1点%d円）" % (lab, _MIDEV_UNIT), uv_midev_sh[nm])
                             for nm, lab in _MIDEV_SHADOWS
                             if nm in ("midev_sel", "midev_ev2n2", "midev_ev2h1", "midev_ev2gap")
                             and nm != _MIDEV_DISPLAY)
                    if _MIDEV_DISPLAY else "")
                 + (_uv_row("C案の元の条件（10〜50倍×EV&ge;1.5・最大5点×%d円・上限なし全件）" % _MIDEV_UNIT,
                            uv_midev_all) if _MIDEV_DISPLAY else
                    _uv_row("中オッズEV（10〜50倍×EV&ge;1.5・最大5点×100円）", uv_midev)
                    + "".join(_uv_row("C案の厳選候補：%s（1点%d円）" % (lab, _MIDEV_UNIT), uv_midev_sh[nm])
                              for nm, lab in _MIDEV_SHADOWS))
                 + '<div class="tsub" style="margin-top:8px">'
                   'これらは「もしこの条件で買っていたら」を記録しているだけで、'
                   '<b>実際にはお金を入れていません</b>。実弾に昇格させるのは、'
                   '上の「通算と95%信頼区間」で下限が100%を超えてからです。</div></div>')
    uv_kpi = uv_live + uv_shadow

    # v79 (2026-09-18 藤田指示「展示タイムも全レース表示するようにして」):
    #   これまで展示タイムは波乱の目(≤10%/≤20%)の表で「1号艇の順位」だけ出していた。
    #   /final の全レースカードに 6艇ぶんの展示タイム(順位つき)＋展示進入 を表示する。
    #   取得元は当日 openapi の直前情報(1回だけ取得してカードと波乱ブロックで共用)。
    #   openapi に未反映で締切25分前〜2分後のレースは公式 beforeinfo を直読み(60秒キャッシュ)＝v75と同じ経路。
    try:
        _rare_data = fetch_openapi(hd)
    except Exception:
        _rare_data = None

    _ex_cache = {}

    def _ex_line(r):
        """カード1枚ぶんの「展示タイム(6艇・順位つき)＋展示進入」の行。未公開なら『展示待ち』。"""
        try:
            jcd = int(r.get("jcd")); rno = int(r.get("rno"))
        except (TypeError, ValueError):
            return ''
        ck = (jcd, rno)
        if ck in _ex_cache:
            return _ex_cache[ck]
        rc = find_race(_rare_data, jcd, rno) if _rare_data is not None else None
        bi = None
        if rc is not None and r.get("status") != "settled":
            closed = _tri_parse_closed(r.get("closed_at"), now.date())
            ml = ((closed - now).total_seconds() / 60.0) if closed else None
            if ml is not None and -2 <= ml <= 25 and (_ex_rank1(rc) is None or _lane_desc(rc) is None):
                bi = fetch_beforeinfo(hd, jcd, rno)
        ex = {}
        if bi and len(bi.get("ex") or {}) >= 4:
            for f, v in (bi.get("ex") or {}).items():
                try:
                    ex[int(f)] = float(v)
                except (TypeError, ValueError):
                    pass
        elif rc is not None:
            try:
                for b in extract_boats(rc):
                    v = fnum(b.get("ex"))
                    if v > 0:
                        ex[int(b["frame"])] = v
            except Exception:
                ex = {}
        if len(ex) < 4:
            out = ('<div style="margin-top:4px;font-size:13px"><span class="tsub">展示タイム</span>　'
                   '<b style="color:#8a6d1f">&#9203; 展示待ち</b>'
                   '<span class="tsub" style="font-size:11px">　締切10分前ごろ判明（この画面は1分ごとに自動更新）</span></div>')
            _ex_cache[ck] = out
            return out
        best = min(ex.values()); worst = max(ex.values())
        cells = []
        for f in range(1, 7):
            v = ex.get(f)
            if v is None:
                cells.append('<span class="tsub">%d号 —</span>' % f)
                continue
            rk = 1 + sum(1 for g, w in ex.items() if g != f and w < v)
            if v <= best + 1e-9:
                col = 'color:#1c7a38;font-weight:800'
            elif v >= worst - 1e-9:
                col = 'color:#b23b3b'
            else:
                col = 'color:#2c3742'
            box = 'background:#fff3c4;border-radius:4px;padding:0 4px;' if f == 1 else ''
            cells.append('<span style="%s%s">%d号 <b>%.2f</b><span style="font-size:11px">(%d位)</span></span>'
                         % (box, col, f, v, rk))
        ld = _lane_desc(rc, bi) if rc is not None else None
        lane_s = ''
        if ld:
            if ld["wakunari"]:
                lane_s = '　<span style="font-size:12px">進入 <b style="color:#1c7a38">%s</b> <span style="color:#1c7a38">枠なり</span></span>' % ld["text"]
            else:
                lane_s = ('　<span style="font-size:12px">進入 <b style="color:#b8860b">%s</b> '
                          '<span style="color:#b8860b">枠なり崩れ</span>%s</span>'
                          % (ld["text"], ('（1号艇は1コース）' if ld["c1"] == 1
                                          else '<b style="color:#b23b3b">（1号艇が%dコース）</b>' % ld["c1"])))
        out = ('<div style="margin-top:4px;font-size:13px"><span class="tsub">展示タイム</span>　%s%s</div>'
               % ("　".join(cells), lane_s))
        _ex_cache[ck] = out
        return out

    def card(r):
        try:
            picks = json.loads(r.get("picks_json") or "[]")
        except Exception:
            picks = []
        # v76(2026-09-16 藤田指示「ややこしいから、実際に買うものだけ『買い』にして」):
        #   カード右上の大きい「買い/見送り」は、実際にお金を入れる A案・B案・C案 だけで決める。
        #   的中率重視モデル(本命型)の判定・星・5点の表はカードに出さない（台帳への記録は従来どおり継続）。
        try:
            uv = json.loads(r.get("uv_json") or "{}")
        except Exception:
            uv = {}
        _rk = (r.get("date"), str(r.get("jcd")), str(r.get("rno")))
        # v85: 4点のうちオッズ上限/EVフロアを通る点が無いレースは、判定は「買い」でも実際には買わない
        _legs_ok = bool(uv.get("picks"))
        _planB = (uv.get("dec40") == "買い" and _legs_ok)
        _planA = (uv.get("dec_stable") == "買い" and _legs_ok and _rk not in _A_STOPPED)
        _planA_stopped = (uv.get("dec_stable") == "買い" and _rk in _A_STOPPED)
        _cpk = midev_shadow_picks(uv.get("midev") or {}, _MIDEV_DISPLAY) if (uv and _MIDEV_DISPLAY) else []
        # v77: 波乱の目(≤10%/≤20%)フォーメーション。展示で1号艇が3位以下(ex_skip False)と確認できた時だけ買い。
        #   ≤10%該当は≤20%にも該当し買い目は同一なので、二重買いにせず ≤10%(より希少)を優先ラベルにする。
        def _rare_buy(field):
            rr = uv.get(field) or {}
            return rr if (rr.get("flag") and rr.get("picks") and rr.get("ex_skip") is False) else None
        _rare10 = _rare_buy("rare"); _rare20 = _rare_buy("rare20")
        _rarebuy = _rare10 or _rare20
        _rarelab = "波乱≤10%" if _rare10 else ("波乱≤20%" if _rare20 else None)
        _dpk = (uv.get("droi") or {}).get("picks") or []   # v78: D案(並行検証)
        _plans = [lab for ok, lab in ((_planA, "A案"), (_planB, "B案"), (bool(_cpk), "C案"), (bool(_dpk), "D案")) if ok]
        if _rarebuy:
            _plans.append(_rarelab)
        isbuy = bool(_plans)
        badge = (('<span class="big buy">&#128994; 買い</span> <span class="tsub" style="font-weight:700;color:#1c7a38">%s</span>'
                  % "・".join(_plans)) if isbuy
                 else '<span class="big skip">&#128308; 見送り</span>')
        h = '<div class="fcard"><div class="fhead">'
        h += '<div><span class="ct">%s %sR</span> <span class="tsub">締切%s・判定%s</span></div>' % (
            ven(r.get("jcd")), r.get("rno"), r.get("closed_at", ""), str(r.get("ts", ""))[11:16])
        h += '<div>%s</div></div>' % badge
        h += _ex_line(r)   # v79: 展示タイム(6艇)＋展示進入を全レースのカードに表示
        if isbuy:
            _tot = 0.0
            h += '<table><tr><th>案</th><th>買い目</th><th>オッズ</th><th>金額</th><th>的中時払戻（判定時オッズ）</th></tr>'
            if _planA or _planB:
                _lab = "・".join(x for x, ok in (("A案", _planA), ("B案", _planB)) if ok)
                for p in (uv.get("picks") or []):
                    o = p.get("odds"); stk = fnum(p.get("stake")); _tot += stk
                    pay = (o * stk) if (o is not None and stk) else None
                    h += '<tr><td>%s</td><td class="combo">%s</td><td>%s</td><td>%s円</td><td>%s</td></tr>' % (
                        _lab, p.get("combo"), (str(o) + "倍" if o is not None else "—"),
                        f"{int(stk):,}", (f"{int(pay):,}円" if pay else "—"))
            for p in _cpk:
                o = p.get("odds"); stk = fnum(p.get("stake")); _tot += stk
                pay = (fnum(o) * stk) if (o is not None and stk) else None
                h += '<tr><td>C案</td><td class="combo">%s</td><td>%s</td><td>%s円</td><td>%s</td></tr>' % (
                    p.get("combo"), (str(o) + "倍" if o is not None else "—"),
                    f"{int(stk):,}", (f"{int(pay):,}円" if pay else "—"))
            for p in _dpk:
                o = p.get("odds"); stk = fnum(p.get("stake")); _tot += stk
                pay = (fnum(o) * stk) if (o is not None and stk) else None
                h += '<tr><td>D案</td><td class="combo">%s</td><td>%s</td><td>%s円</td><td>%s</td></tr>' % (
                    p.get("combo"), (str(o) + "倍" if o is not None else "—"),
                    f"{int(stk):,}", (f"{int(pay):,}円" if pay else "—"))
            if _rarebuy:
                _rpk = _rarebuy.get("picks", [])
                _rt = sum(fnum(p.get("stake")) for p in _rpk); _tot += _rt
                _fmt = "%s-[%s]-[全]" % (_rarebuy.get("head"),
                                        "".join(str(x) for x in _rarebuy.get("second", [])))
                _win = r.get("win_combo")
                _rsettled = (r.get("status") == "settled" and _win)
                _paycell = "結果で確定"
                if _rsettled:
                    _paycell = (('<span class="res-hit">%s 的中 払戻%s円</span>'
                                 % (_win, f"{int(fnum(_rarebuy.get('ret'))):,}"))
                                if str(_rarebuy.get("hit")) == "1" else "不的中")
                h += ('<tr><td>%s</td><td class="combo">%s (%d点)</td><td>—</td><td>%s円</td><td>%s</td></tr>'
                      % (_rarelab, _fmt, len(_rpk), f"{int(_rt):,}", _paycell))
            h += '</table><div class="tsub" style="margin:2px 0 4px">このレースで買う合計 <b>%s円</b></div>' % f"{int(_tot):,}"
        else:
            h += '<div class="tsub" style="margin:4px 0">A案・B案・C案とも条件に当てはまらないため買いません。%s</div>' % (
                ('（過小評価キーは立っているが、4点ともオッズ150倍以下かつ1点EV1.5以上を満たさないため買わない）'
                 if (uv and (uv.get("dec40") == "買い" or uv.get("dec_stable") == "買い") and not uv.get("picks"))
                 else ('（安定型は該当したが、当日連敗ストップ中のため買わない）' if _planA_stopped else '')))
        # 過小評価キー（毎レース 買い/見送り を表示）
        if uv:
            d40 = uv.get("dec40", uv.get("decision", "見送り"))
            d55 = uv.get("dec55", uv.get("decision", "見送り"))
            dst = uv.get("dec_stable", "見送り")

            def _bdg(lbl, dec):
                col = "#1c7a38" if dec == "買い" else "#8a97a5"
                ic = "🟢" if dec == "買い" else "🔴"
                return '<b style="color:%s">%s%s%s</b>' % (col, ic, lbl, dec)
            def _leg_line(uvx, only_dropped):
                """v88: 実弾4点を オッズ・EV 付きで並べる。only_dropped=True なら落とした点だけ。"""
                cand = _uv_live4(uvx)
                if not cand:
                    return ""
                keep = {q.get("combo") for q in (uvx.get("picks") or [])}
                seg = []
                for q in cand:
                    o = fnum(q.get("odds")); e = fnum(q.get("P")) * o
                    ng = []
                    if _UV_LEG_ODDS_CAP is not None and o > _UV_LEG_ODDS_CAP:
                        ng.append("150倍超")
                    if _UV_LEG_EV_FLOOR is not None and e < _UV_LEG_EV_FLOOR:
                        ng.append("EV不足")
                    ok = q.get("combo") in keep
                    if only_dropped and ok:
                        continue
                    seg.append('<span class="combo" style="color:%s">%s</span>'
                               '<span class="tsub">(%s倍・EV%.2f%s)</span>' % (
                                   "#1c7a38" if ok else "#8a97a5", q.get("combo"),
                                   ("%g" % o) if o else "—", e,
                                   ("・" + "/".join(ng)) if ng else ("・買う" if ok else "")))
                if not seg:
                    return ""
                lab = "見送った点" if only_dropped else "買い目候補4点（条件を満たす点なし＝買わない）"
                return ('<div class="tsub" style="margin-top:4px">%s　%s</div>' % (lab, " / ".join(seg)))
            _nolegs = not uv.get("picks")
            _sfx = "(買い目なし)" if _nolegs else ""
            badge = ('過小評価キー　' + _bdg("B案(≥40):", d40 if not (_nolegs and d40 == "買い") else "買い" + _sfx)
                     + ' ／ ' + _bdg("A案(安定型):", dst if not (_nolegs and dst == "買い") else "買い" + _sfx)
                     + ' ／ ' + _bdg("≥55(記録のみ):", d55))
            inner = ""
            k = uv.get("key")
            if k:
                kp = k.get("keyp")
                kps = ('・キー1着%s%%' % round(fnum(kp) * 100)) if kp is not None else ''
                inner += '　<span class="tsub">キー<b>%s号艇</b>(モ%s%%/全国2連率%s%%%s)</span>' % (
                    k.get("frame"), round(fnum(k.get("motor2"))), round(fnum(k.get("nat2"))), kps)
            upicks = uv.get("picks", [])
            win = r.get("win_combo")
            settled = (r.get("status") == "settled" and win)
            if upicks:
                inner += '　' + " / ".join(
                    (('<span class="combo res-hit" style="background:#e3f5e8;border-radius:4px;padding:0 3px">%s</span>(%s倍)'
                      if (settled and p.get("combo") == win) else '<span class="combo">%s</span>(%s倍)') % (
                        p.get("combo"), p.get("odds") if p.get("odds") is not None else "—")) for p in upicks)
                _bought = (d40 == "買い" or (dst == "買い" and (r.get("date"), str(r.get("jcd")), str(r.get("rno"))) not in _A_STOPPED))
                if settled and _bought:
                    hp = next((p for p in upicks if p.get("combo") == win), None)
                    if hp:
                        # v73: どの組が当たったか分かるように「1-2-3 的中(45.5倍) 払戻34,000円」
                        ur = int(fnum(r.get("uv_ret")))
                        inner += '　<span class="res-hit">%s 的中(%s倍)%s</span>' % (
                            hp.get("combo"), hp.get("odds") if hp.get("odds") is not None else "—",
                            (f" 払戻{ur:,}円" if ur else ""))
                    else:
                        inner += '　<span class="res-miss">不的中</span>'
                inner += _leg_line(uv, True)
            elif (d40 == "買い" or dst == "買い") and uv.get("picks_all"):
                # v88: ゲートは通ったが4点とも オッズ≤150 / 1点EV≥1.5 を満たさなかったレース。
                #      「なぜ買わないのか」を、候補4点のオッズとEVを出して見せる。
                inner += _leg_line(uv, False)
            elif d40 == "見送り" and d55 == "見送り":
                inner += '　<span class="tsub">%s</span>' % (uv.get("reason", "")[:60])
            if d40 == "買い" and uv.get("dec40_lane") in ("買い", "見送り"):
                badge += ' ／ ' + _bdg("進入クリーン:", uv["dec40_lane"])
            h += ('<div style="background:#fff8e6;border:1px solid #f0e2b8;border-radius:10px;padding:8px 10px;margin-top:8px;font-size:13px">'
                  '%s%s%s%s%s</div>') % (badge, inner, _lane_line(uv), _lane_real_line(uv),
                                        _midev_line(uv))
        # 結果（v73: 別モデル(的中率重視)の的中/研究は出さず、実際に買いになった案だけ）
        if r.get("status") == "settled" and r.get("win_combo"):
            win = r.get("win_combo")
            parts = []
            if uv:
                plans = []
                if uv.get("picks"):   # v88: 買い目0点(条件外)のレースは買っていないので精算しない
                    if uv.get("dec40") == "買い":
                        plans.append("B案")
                    if (uv.get("dec_stable") == "買い"
                            and (r.get("date"), str(r.get("jcd")), str(r.get("rno"))) not in _A_STOPPED):
                        plans.append("A案")
                if plans:
                    hp = next((p for p in (uv.get("picks") or []) if p.get("combo") == win), None)
                    lab = "・".join(plans)
                    if hp:
                        ur = int(fnum(r.get("uv_ret")))
                        parts.append('<span class="res-hit">%s %s 的中(%s倍) 払戻%s円</span>' % (
                            lab, win, hp.get("odds") if hp.get("odds") is not None else "—", f"{ur:,}"))
                    else:
                        parts.append('<span class="res-miss">%s 不的中</span>' % lab)
                mvd = uv.get("midev") or {}
                if _MIDEV_DISPLAY:
                    sub, s_, r_, st_ = midev_shadow_result(r, mvd, _MIDEV_DISPLAY)
                    if sub and st_:
                        p0 = sub[0]
                        parts.append(('<span class="res-hit">C案 %s 的中(%s倍) 払戻%s円</span>' % (
                            win, p0.get("odds"), f"{int(r_):,}")) if r_ > 0 else
                            '<span class="res-miss">C案 不的中</span>')
                # v78: D案(並行検証)
                dvd = uv.get("droi") or {}
                if dvd.get("picks"):
                    hpd = next((p for p in dvd["picks"] if p.get("combo") == win), None)
                    if str(dvd.get("hit")) == "1" and hpd:
                        parts.append('<span class="res-hit">D案 %s 的中(%s倍) 払戻%s円</span>' % (
                            win, hpd.get("odds"), f"{int(fnum(dvd.get('ret'))):,}"))
                    else:
                        parts.append('<span class="res-miss">D案 不的中</span>')
                # v77: 波乱の目フォメ(展示3位以下で買った分)。≤10%と≤20%は同一買いなので1回だけ表示。
                for _f, _lab in (("rare", "波乱≤10%"), ("rare20", "波乱≤20%")):
                    _rr = uv.get(_f) or {}
                    if _rr.get("flag") and _rr.get("picks") and _rr.get("ex_skip") is False:
                        parts.append(('<span class="res-hit">%s %s 的中 払戻%s円</span>'
                                      % (_lab, win, f"{int(fnum(_rr.get('ret'))):,}"))
                                     if str(_rr.get("hit")) == "1"
                                     else '<span class="res-miss">%s 不的中</span>' % _lab)
                        break
            res = '<span class="tsub">結果 %s</span>' % win
            if parts:
                res += '　' + ' ／ '.join(parts)
            h += '<div style="margin-top:6px">%s</div>' % res
            if any("res-hit" in x for x in parts):
                # v73: 実際に買った案が当たったレースは、カードの背景と枠を緑にして見出しに印を付ける
                h = h.replace('<div class="fcard">',
                              '<div class="fcard" style="background:#e6f6ea;border:2px solid #1c7a38;'
                              'box-shadow:0 0 0 3px #bfe6c9">', 1)
                h = h.replace('<div class="fhead"><div>',
                              '<div class="fhead"><div><span style="background:#1c7a38;color:#fff;border-radius:6px;'
                              'padding:2px 8px;font-weight:800;margin-right:6px">&#127919; 的中</span>', 1)
        h += '</div>'
        return h

    if not today:
        body = ('<div class="fcard tsub">本日の最終判定はまだありません。'
                '開催時間帯に、各レースの締切約5〜10分前へ来ると自動で判定が追加されます'
                '（Render常駐スケジューラが数分おきに自動実行。ボタン操作は不要）。</div>')
    else:
        order = sorted(today, key=lambda r: str(r.get("closed_at") or "z"))
        body = "".join(card(r) for r in order)
    # ---- データ未取得時のプレースホルダ（「該当なし」と区別する）----
    def _pending_block(color, title):
        return ('<div class="fcard" style="border-color:%s">'
                '<div class="tsub" style="font-weight:700;color:%s">%s</div>'
                '<div class="tsub" style="margin-top:6px">'
                '本日の番組表データをまだ取得できていません（公開前、または一時的な取得失敗）。'
                'しばらくしてから再読み込みしてください。'
                '<b>「該当なし」ではありません</b>。</div></div>') % (color, color, title)

    # ---- 本日の≥４０買い候補（混戦 pgap≤0.50・未締切のみ）----
    def _uv_candidates():
        try:
            data = fetch_openapi(hd)
        except Exception:
            data = None
        if not data:
            return _pending_block("#1c7a38",
                                  "&#128994; 本日の 買い候補（≥４０混戦 / ≥55 / 安定型・未締切）")
        judged = {(str(r.get("jcd")), str(r.get("rno"))) for r in today}
        cands = []
        for jcd in range(1, 25):
            for rno in range(1, 13):
                rc = find_race(data, jcd, rno)
                if rc is None:
                    continue
                ex, tri = _payouts(rc)
                if ex is not None or tri is not None:
                    continue
                closed = _tri_parse_closed(_tri_closed_at(rc), now.date())
                if closed is None or (closed - now).total_seconds() <= 0:
                    continue
                try:
                    power = predict_power(extract_boats(rc))
                    pgap = _conf_gap(power)[1]
                    uv = undervalue_pick(rc, None)
                except Exception:
                    continue
                k = uv.get("key") if uv else None
                if not k:
                    continue
                if not (fnum(k.get("gap")) >= 3 and fnum(k.get("motor2")) >= 40
                        and fnum(k.get("nat2")) >= 5):
                    continue
                # 3タイプの朝候補: ≥40(混戦pgap≤0.50) / ≥55(モーター≥55) / 安定型(keyp≥10%)
                is40 = (pgap <= 0.50)
                is55 = (fnum(k.get("motor2")) >= 55)
                isst = (fnum(k.get("keyp")) >= 0.10)
                if not (is40 or is55 or isst):
                    continue
                tiers = []
                if is40: tiers.append("≥40")
                if is55: tiers.append("≥55")
                if isst: tiers.append("安定型")
                try:
                    sr = _shinnyu_risk(rc, k.get("frame"), uv.get("honmei"))
                except Exception:
                    sr = None
                cands.append({"shinnyu": sr, "lane": _lane_text(rc),
                              "closed": closed, "jcd": jcd, "rno": rno,
                              "frame": k.get("frame"), "motor2": k.get("motor2"),
                              "nat2": k.get("nat2"), "keyp": k.get("keyp"), "pgap": pgap,
                              "is40": is40, "tiers": tiers,
                              # v60: 朝の時点の買い目(組番・金額)を候補行に持たせる。
                              # odds_map=None で呼んでいるためオッズ/EVはNoneだが、組番と配分は確定している。
                              "picks": uv.get("picks", []), "total": uv.get("total", 0),
                              "honmei": uv.get("honmei"),
                              "judged": (str(jcd), str(rno)) in judged})
        cands.sort(key=lambda c: c["closed"])
        n = len(cands)
        # v86: 「いつ画面を見ればいいか」が一目で分かるように、次の締切までの分数と
        #      残り候補の時間帯分布を先頭に出す。
        _nxt = ""
        try:
          if cands:
              _c0 = min(cands, key=lambda c: c["closed"])
              _mins = int((_c0["closed"] - now).total_seconds() // 60)
              _byh = {}
              for c in cands:
                  _byh[c["closed"].hour] = _byh.get(c["closed"].hour, 0) + 1
              _spread = " ".join("%d時<b>%d</b>" % (k, v) for k, v in sorted(_byh.items()))
              _nxt = ('<div style="margin-top:8px;padding:10px 12px;background:#f4fbf6;'
                      'border:1px solid #cfe8d8;border-radius:8px">'
                      '<span style="font-size:13px;color:#5c6b7a">次のA案・B案候補は</span>　'
                      '<b style="font-size:22px;color:#1c7a38">%s</b>'
                      '<span style="font-size:15px;color:#1c7a38;font-weight:700">　あと%d分</span>'
                      '<span class="tsub" style="font-size:12px">　（%s の %dR）</span>'
                      '<div class="tsub" style="margin-top:6px;font-size:12px">残り候補の時間帯　%s</div>'
                      '<div class="tsub" style="margin-top:4px;font-size:12px">'
                      '&#9888;&#65039; C案はオッズ次第なので朝には分かりません。実績では'
                      '<b>1日20〜30件・次の買いまで中央値13分</b>（30分以内が80%%）。'
                      '11〜16時に画面を見ていれば全体の約75%%を拾えます。</div></div>'
                      ) % (_c0["closed"].strftime("%H:%M"), max(_mins, 0),
                           _VEN.get(int(_c0["jcd"]), _c0["jcd"]), _c0["rno"], _spread)
        except Exception:
            _nxt = ""
        h = ('<div class="fcard" style="border-color:#1c7a38">'
             '<div class="tsub" style="font-weight:700;color:#1c7a38">'
             '&#128994; 本日の 買い候補（≥４０混戦 / ≥55 / 安定型・未締切）　%d件</div>' % n + _nxt)
        if n == 0:
            h += ('<div class="tsub" style="margin-top:6px">今のところ未締切レースに買い候補はありません'
                  '（条件を満たすレースが出れば自動で表示）。</div>')
        else:
            h += ('<table style="margin-top:8px"><tr><th>締切</th><th>場R</th><th>対象</th>'
                  '<th>キー</th><th>モ/全国2連</th><th>pgap</th><th>進入</th>'
                  '<th style="text-align:left">買い目（朝の時点・最大4点×1,000円／締切前に絞る）</th><th>状態</th></tr>')
            for c in cands:
                bd = (c["is40"] and c["pgap"] > 0.45)
                st = "判定済" if c["judged"] else ("直前で見送りの可能性" if bd else "候補")
                tier_s = " / ".join(c["tiers"])
                # v60: 朝の時点の買い目(組番・金額)。オッズは締切5分前に確定するためここでは出さない。
                if c.get("picks"):
                    buy_s = ('<span class="combo">'
                             + '</span> / <span class="combo">'.join(
                                 "%s<span class=\"tsub\">(%s円)</span>" % (p.get("combo"), p.get("stake"))
                                 for p in c["picks"])
                             + '</span> <span class="tsub">計%s円</span>' % f"{int(fnum(c.get('total'))):,}")
                else:
                    buy_s = '<span class="tsub">—</span>'
                sr = c.get("shinnyu")
                lb = _shinnyu_label(sr)
                lcol = {"高": "#b23b3b", "中": "#b8860b"}.get(lb, "#5c6b7a")
                if c.get("lane"):
                    sn = '<span style="color:#1c7a38;font-weight:700">%s</span>' % c["lane"]
                elif sr:
                    sn = ('<span style="color:%s;font-weight:700" title="%s">%s</span>'
                          '<br><span class="tsub">崩れ%.0f%%</span>') % (
                        lcol, " / ".join(sr.get("names") or []) or "前づけ常習者なし", lb,
                        (sr.get("race") or 0) * 100)
                else:
                    sn = '<span class="tsub">—</span>'
                h += ('<tr><td>%s</td><td class="ct">%s%sR</td><td><b>%s</b></td>'
                      '<td>%s号(1着%s%%)</td><td>%s/%s%%</td><td>%.2f%s</td><td>%s</td>'
                      '<td style="text-align:left;font-size:13px">%s</td><td class="tsub">%s</td></tr>') % (
                    c["closed"].strftime("%H:%M"), ven(c["jcd"]), c["rno"], tier_s,
                    c["frame"], round(fnum(c["keyp"]) * 100), round(fnum(c["motor2"])),
                    round(fnum(c["nat2"])), c["pgap"], ("*" if bd else ""), sn, buy_s, st)
            h += '</table>'
        h += ('<div class="tsub" style="margin-top:6px">※「対象」=そのレースが該当する買いタイプ。'
             '≥40は混戦(pgap≤0.50)限定・≥55はモーター2連率≥55・安定型はキー1着率≥10%。複数該当ほど強い候補。'
             '最終判定は各レース締切５分前に最新オッズで確定（下の一覧に追加）。pgap0.45超(*)は≥40が直前で見送りに転ぶことあり。'
             '<b>v85以降、実際に買うのはこの4点のうち「オッズ150倍以下かつ1点EV1.5以上」だけ</b>'
             'なので、締切前に1〜2点へ減るか、0点＝見送りになることがあります（過去実測で4割強のレースが見送り）。'
             '過去半年BT ≥40混戦91.9%／≥55約97%／安定型約90%＝いずれも100%未満の検証ツール。<br>'
             '<b>「買い目」は朝の時点の予定（4点×1,000円＝1レース4,000円。キーを1着に置く2点と、'
             '3着流しのうち P-Q-X / P-R-X の2点は負けが明確なため除外済み）。</b>'
             'オッズは締切5分前に確定するためここでは未表示、'
             '直前情報(展示・進入)でパワー順位が動くと組番が入れ替わることがあります。'
             '最終確定は下の各レースカードを参照。<br>'
             '<b>「進入」列</b>=展示進入が出ていればその隊形(内側コース順の枠番)を緑で表示。'
             'まだ出ていないレースは、選手ごとの前づけ率から出した事前予想（高/中/低＋隊形が崩れる確率）。'
             '前づけ常習者名はマウスオーバーで表示。過去249日の実測では枠なりにならないレースは19.3%、'
             '進入リスク「高」の帯では約69%が崩れます。<br>'
             '<b>ただし検証結果として、朝の進入リスクで買うレースを削っても回収率は上がりません</b>'
             '（94.2%→最良94.5%・月別も改善せず）。効くのは展示進入を見てからの足切り（影バケット'
             '「進入クリーン」で並走検証中）です。この列は注意喚起のための情報表示です。</div></div>')
        return h
    # ---- 通算（全期間）の回収率と95%信頼区間 ----
    # v61: 「本数を絞れば回収率が上がる」は成り立たない、を毎日画面で見えるようにする枠。
    #   ランダムに減らしても期待値は変わらず、ブレ(信頼区間)が広がるだけ。小さいバケットの
    #   高い回収率は、その広いブレの上振れを見ているだけのことが多い。
    def _uv_pairs_all(field):
        out = []
        for r in rows:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                continue
            if not u:
                continue
            if _uv_dec_or_decision(u, field) != "買い" or fnum(u.get("total")) <= 0:
                continue   # v85: 4点とも条件外で実際には買っていないレースは集計に入れない
            if r.get("status") != "settled":
                continue
            out.append((fnum(u.get("total")), fnum(r.get("uv_ret"))))
        return out

    def _uvp_pairs_all(name):
        out = []
        for r in rows:
            if r.get("status") != "settled":
                continue
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                continue
            if not u:
                continue
            k, s_, r_ = uvp_result(r, u, name)
            if k:
                out.append((s_, r_))
        return out

    def _midev_pairs_all(shadow=None):
        """通算表用: 中オッズEV(shadow=None は全件)の精算済み (投資, 払戻) をレース単位で。"""
        out = []
        for r in rows:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                continue
            mv = (u or {}).get("midev") or {}
            if not mv.get("picks"):
                continue
            if shadow:
                sub, s_, r_, st_ = midev_shadow_result(r, mv, shadow)
                if sub and st_:
                    out.append((s_, r_))
            elif r.get("status") == "settled" and str(mv.get("hit", "")) != "":
                out.append((fnum(mv.get("total")), fnum(mv.get("ret"))))
        return out

    def _midev_today_block():
        """本日の中オッズEV 買い目一覧（実弾ぶんを上に、上限超えの記録のみを下に）。"""
        if _MIDEV_DISPLAY:
            return _midev_today_block_disp()
        live = []; rec = []
        for r in today:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                continue
            mv = (u or {}).get("midev") or {}
            if not mv.get("picks"):
                continue
            (live if mv.get("live") else rec).append((r, mv))
        n = len(live)
        h = ('<div class="fcard" style="border-color:#1c7a38">'
             '<div class="tsub" style="font-weight:700;color:#1c7a38">'
             '&#128994; 本日の 中オッズEV 買い目（C案・少額）　実弾 %d / %d レース'
             '<span style="font-weight:400">　※上限に達した分は記録のみ %d レース</span></div>') % (
            n, _MIDEV_DAILY_MAX, len(rec))
        if not live and not rec:
            h += '<div class="tsub" style="margin-top:6px">まだ該当レースがありません。</div></div>'
            return h
        h += ('<table style="margin-top:8px"><tr><th>締切</th><th>場R</th>'
              '<th style="text-align:left">買い目（組番・オッズ・EV）</th><th>金額</th>'
              '<th>扱い</th><th>結果</th></tr>')
        for r, mv in (live + rec):
            pk = mv.get("picks") or []
            body = " / ".join('<span class="combo">%s</span><span class="tsub">(%s倍・EV%s)</span>'
                              % (p.get("combo"), p.get("odds"), p.get("ev")) for p in pk)
            if str(mv.get("hit", "")) == "":
                res = '<span class="tsub">—</span>'
            elif str(mv.get("hit")) == "1":
                res = '<span class="res-hit">的中 %s円</span>' % f"{int(fnum(mv.get('ret'))):,}"
            else:
                res = '<span class="res-miss">不的中</span>'
            h += ('<tr><td>%s</td><td class="ct">%s%sR</td>'
                  '<td style="text-align:left;font-size:13px">%s</td><td>%s円</td>'
                  '<td class="tsub">%s</td><td>%s</td></tr>') % (
                r.get("closed_at", ""), ven(int(fnum(r.get("jcd")))), r.get("rno"), body,
                f"{int(fnum(mv.get('total'))):,}",
                ("<b style='color:#1c7a38'>実弾</b>" if mv.get("live") else "記録のみ"), res)
        h += '</table>'
        h += ('<div class="tsub" style="margin-top:6px">'
              '市場オッズ%d〜%d倍のうち、較正後の期待値(確率×オッズ)が%s以上の組を、EVの高い順に最大%d点。'
              '1点%d円＝1レース最大%s円、<b>1日%dレースまでを実弾</b>（それ以降は同じ条件で記録だけ続けます）。<br>'
              '<b style="color:#b23b3b">この枠はまだ勝ちが証明されていません。</b>'
              'ライブ10日の断片では回収率102.7%%ですが大穴3本を抜くと88.5%%・日別100%%超は4/9日で、'
              'EVフロアを上げるとむしろ悪化します(1.8→96.9%%・2.0→72.0%%)。'
              'オッズを見ずにモデルだけで中オッズを買う版は過去249日で74〜77%%＝控除率の壁どおりでした。'
              '少額に固定してあるのは、勝ちが確認できるまで損失を1日1万円以内に閉じ込めるためです。</div></div>') % (
            int(_MIDEV_LO), int(_MIDEV_HI), _MIDEV_EV, _MIDEV_N, _MIDEV_UNIT,
            f"{_MIDEV_N * _MIDEV_UNIT:,}", _MIDEV_DAILY_MAX)
        return h

    def _midev_today_block_disp():
        """v71/v72: 本日のC案 厳選(表示中の条件)の1点だけの一覧。上限なし・締切順。ポセイドン一致は◎印。"""
        items = []
        for r in today:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                continue
            mv = (u or {}).get("midev") or {}
            sub, s_, r_, st_ = midev_shadow_result(r, mv, _MIDEV_DISPLAY)
            if sub:
                items.append((r, mv, sub, s_, r_, st_))
        items.sort(key=lambda x: str(x[0].get("closed_at") or x[0].get("ts") or ""))
        stake = sum(x[3] for x in items if x[5]); ret = sum(x[4] for x in items if x[5])
        hits = sum(1 for x in items if x[5] and x[4] > 0)
        h = ('<div class="fcard" style="border-color:#1c7a38">'
             '<div class="tsub" style="font-weight:700;color:#1c7a38">'
             '&#128994; 本日の C案 買い（%s の1点×%d円）　%d レース'
             '<span style="font-weight:400">　精算済 %d／的中 %d／投資 %s円・払戻 %s円</span></div>') % (
            dict(_MIDEV_SHADOWS)[_MIDEV_DISPLAY], _MIDEV_UNIT, len(items), sum(1 for x in items if x[5]), hits,
            f"{int(stake):,}", f"{int(ret):,}")
        if not items:
            h += '<div class="tsub" style="margin-top:6px">まだ該当レースがありません。</div></div>'
            return h
        h += ('<table style="margin-top:8px"><tr><th>締切</th><th>場R</th>'
              '<th style="text-align:left">買い目（組番・オッズ・EV）</th><th>金額</th>'
              '<th>ポセイドン本命と</th><th>結果</th></tr>')
        for r, mv, sub, s_, r_, st_ in items:
            p = sub[0]
            ag = midev_pos_agree(r, p)
            agc = ('<b style="color:#1c7a38">◎一致</b>' if ag is True else
                   ('<span class="tsub">不一致</span>' if ag is False else '<span class="tsub">—</span>'))
            if not st_:
                res = '<span class="tsub">—</span>'
            elif r_ > 0:
                res = '<span class="res-hit">的中 %s円</span>' % f"{int(r_):,}"
            else:
                res = '<span class="res-miss">不的中（%s）</span>' % (r.get("win_combo") or "")
            h += ('<tr><td>%s</td><td class="ct">%s%sR</td>'
                  '<td style="text-align:left;font-size:13px"><span class="combo">%s</span>'
                  '<span class="tsub">(%s倍・EV%s)</span></td><td>%s円</td><td>%s</td><td>%s</td></tr>') % (
                r.get("closed_at", ""), ven(int(fnum(r.get("jcd")))), r.get("rno"),
                p.get("combo"), p.get("odds"), p.get("ev"), f"{int(s_):,}", agc, res)
        h += '</table>'
        sc = uv_midev_disp_cum or {"bset": 0, "hits": 0, "roi": 0}
        pa = _midev_pos_stats(rows, True); pn = _midev_pos_stats(rows, False)
        h += ('<div class="tsub" style="margin-top:6px">'
              '市場オッズ%d〜%d倍・較正後EV≥%sの候補が<b>2組以上</b>あり、<b>EVが一番高い1点が EV≥%s かつ オッズ%d倍以上</b>の時だけ、その1点を%d円。'
              '1日の上限なし（過去の該当は1日20レース前後＝約4,000円）。<br>'
              '通算（9/08〜・精算済 %d レース）：的中 %d 本・回収率 %.1f%%。'
              '　うちポセイドン本命と◎一致：%d レース・的中 %d・回収率 %.1f%%／不一致：%d レース・的中 %d・回収率 %.1f%%（記録のみ）。<br>'
              '根拠：EV1位＋EV≥2.0 のうち「オッズ30倍未満」(ライブ65.5%%・別期間68.3%%)と「候補が1組だけ」(108.5%%・49.2%%)が'
              '両期間とも負けていたので外した。絞った後はライブ153レース238.0%%（大穴3本抜き123.2%%）／別期間17日368レース124.8%%（88.0%%）。<br>'
              '<b style="color:#b23b3b">まだ勝ちは証明されていません。</b>的中は各10〜16本で、別期間の大穴3本抜きは90%%未満。'
              '24通りの特徴を試した中から選んだ条件なので、今日からの前向き成績で確かめます。'
              '元の条件（全点・早い者順10レース）の記録は裏で続けています。</div></div>') % (
            int(_MIDEV_LO), int(_MIDEV_HI), _MIDEV_EV, _MIDEV_SH_EV, int(_MIDEV_SEL_ODDS), _MIDEV_UNIT,
            sc["bset"], sc["hits"], sc["roi"],
            pa["bset"], pa["hits"], pa["roi"], pn["bset"], pn["hits"], pn["roi"])
        return h

    def _midev_cum_block():
        st = uv_midev_cum; sa = uv_midev_cum_all
        return ('<div class="fcard" style="border-color:#8a97a5">'
                '<div class="tsub" style="font-weight:700;color:#3b4652">'
                '&#128200; 中オッズEV（C案）の通算</div>'
                '<div class="tsub" style="margin-top:6px"><b>実弾ぶん</b>：買い <b>%d</b> レース（精算済 %d）／的中 <b>%d</b> 本'
                '／投資 %s 円・払戻 %s 円／<b>回収率 %.1f%%</b></div>'
                '<div class="tsub" style="margin-top:6px">'
                'あなたの着眼「20〜40倍くらいの中オッズなら結構当たるのでは」を前向きに検証する枠です。'
                '過去249日・38,512レースで測った結論は<b>「中オッズ狙いそのものには妙味なし」</b>：'
                'モデル確率帯ごとの回収率は 大穴(P&lt;0.5%%)55.6%% ／ 約40倍74.1%% ／ 約26倍77.3%% ／ '
                '約16倍76.2%% ／ 約7倍86.8%% で、<b>控除率25%%の壁(=75%%)がほぼそのまま出ます</b>。'
                '的中率が上がるぶんちょうど配当が下がるだけで、回収率は動きません。'
                'はっきりした歪みは「大穴が割高・本命が割安」だけで、向きはむしろ中オッズより本命寄りです。<br>'
                'ただしこれは全部「オッズを見ずに選ぶ」検証で、過去に3連単オッズ盤が無いため'
                '<b>「市場オッズがモデルより高い組だけ買う(EV)」は歴史的に検証不能</b>＝唯一の未検証軸です。'
                'ここはその1点だけを前向きに測る枠で、実弾には入れていません。<br>'
                '<b>ライブ10日の断片（的中率重視5点ぶんのオッズ記録のみ＝この枠の下位集合）では '
                '回収率102.7%%（814点・的中38本）ですが、大穴3本を抜くと88.5%%・日別100%%超は4/9日</b>＝'
                '鉄則でいう「数本の高配当に乗っているだけ」の典型パターンで、まだ判断材料になりません。'
                'また過去249日の実測でモデルの確率は実際より11〜18%%高く出る（過大申告）と分かったため、'
                'ここでは較正後の確率でEVを計算しています（較正しても102.7%%→104.6%%・3本抜き85.1%%で結論は変わらず）。'
                '<b>的中100本を目安に貯めてから判断します。</b><br>'
                '<b>上限超えも含めた全件（判断用の母数）</b>：買い %d レース（精算済 %d）／的中 %d 本／回収率 %.1f%%'
                '</div></div>') % (
            st["buy"], st["bset"], st["hits"], f"{int(st['stake']):,}", f"{int(st['ret']):,}", st["roi"],
            sa["buy"], sa["bset"], sa["hits"], sa["roi"])

    def _cum_block():
        specs = [("≥40混戦（B案・実弾）", "dec40"),
                 ("≥55（厳選）", "dec55"),
                 ("安定型 keyp≥10%（A案・実弾）", "dec_stable"),
                 ("影 keyp≥6%", "dec_kp6"), ("影 keyp≥7%", "dec_kp7"), ("影 keyp≥8%", "dec_kp8"),
                 ("影 モーター≥50", "dec_m50"), ("影 モーター≥60", "dec_m60"),
                 ("影 本命自信度≥20%", "dec_conf20"),
                 ("影 1号艇展示良好", "dec40_ex"), ("影 穏やか水面", "dec40_calm"),
                 ("影 期待値≥1.2", "dec40_ev12"), ("影 期待値≥1.3", "dec40_ev13"),
                 ("影 進入クリーン", "dec40_lane"),
                 ("影 A案・B案の重なり", "dec40_stable"),
                 ("影 B案＋キー内枠", "dec40_in"),
                 ("影 B案＋キー内枠＋4点EV≥1.0", "dec40_in_ev"),
                 ("影 キー内枠＋4点EV≥1.2", "dec_in_ev12"),
                 ("C案 中オッズEV（上限超え含む全件）", lambda: _midev_pairs_all(None))]
        specs += [("買い目 " + lab, (lambda nm=nm: _uvp_pairs_all(nm))) for nm, lab, _ in _UVP_FILTERS]
        specs += [(("C案 買い(表示中) " if nm == _MIDEV_DISPLAY else "C案 比較 ") + lab, (lambda nm=nm: _midev_pairs_all(nm)))
                  for nm, lab in _MIDEV_SHADOWS
                  if (not _MIDEV_DISPLAY or nm in (_MIDEV_DISPLAY, "midev_ev1ev2", "midev_sel",
                                                   "midev_ev2n2", "midev_ev2h1", "midev_ev2gap"))]
        h = ('<div class="fcard" style="border-color:#8a97a5">'
             '<div class="tsub" style="font-weight:700;color:#3b4652">'
             '&#128202; 通算（全期間）の回収率と95%信頼区間　'
             '<span style="font-weight:400">— 数字を信じてよい幅を必ず一緒に見る</span></div>'
             '<table style="margin-top:8px"><tr><th style="text-align:left">バケット</th>'
             '<th>買い(精算済)</th><th>的中</th><th>回収率</th>'
             '<th>95%信頼区間(目安)</th><th>大穴上位3本を除くと</th></tr>')
        any_row = False
        for label, field in specs:
            pairs = field() if callable(field) else _uv_pairs_all(field)
            if not pairs:
                continue
            any_row = True
            n = len(pairs)
            S = sum(x[0] for x in pairs)
            R = sum(x[1] for x in pairs)
            hits = sum(1 for x in pairs if x[1] > 0)
            roi = (R / S * 100) if S else 0
            lo, hi, _se = _roi_ci95(pairs)
            top3 = sum(sorted((x[1] for x in pairs), reverse=True)[:3])
            roi_n3 = ((R - top3) / S * 100) if S else 0
            ci = ("%.0f〜%.0f%%" % (lo, hi)) if lo is not None else "—"
            wide = (lo is not None and (hi - lo) > 60)
            h += ('<tr><td style="text-align:left">%s</td><td class="ct">%d</td><td>%d</td>'
                  '<td><b>%.1f%%</b></td><td%s>%s</td><td class="tsub">%.1f%%</td></tr>') % (
                label, n, hits, roi,
                (' style="color:#b23b3b;font-weight:700"' if wide else ''), ci, roi_n3)
        h += '</table>'
        if not any_row:
            h += '<div class="tsub" style="margin-top:6px">まだ精算済みデータがありません。</div>'
        h += ('<div class="tsub" style="margin-top:8px">'
              '※ <b>「買うレースを減らせば回収率が上がる」は成り立ちません。</b>'
              'ランダムに減らしても期待値は元のまま変わらず、<b>ブレ(信頼区間)が広がるだけ</b>です。'
              '母数が小さいバケットほど区間が広く（赤字表示＝幅60ポイント超）、'
              'たまたま上振れて100%を超えて見えることが頻繁に起きます。'
              '回収率が上がるのは「切ったレースが構造的に平均より悪い」場合だけで、'
              'それを証明するにはこの区間が100%をまたがなくなるまでサンプルが要ります。<br>'
              '※「大穴上位3本を除くと」の列が本体の回収率と大きく離れているバケットは、'
              '数字が数本の高配当に乗っているだけ＝まだ判断材料になりません。<br>'
              '※ 区間は正規近似の目安。払戻の裾が重いので<b>実際はこれよりさらに広い</b>と考えてください。'
              '絞るほど1日に貯まる件数が減るため、<b>絞れば絞るほど検証は速くなるどころか何倍も遅くなります。</b>'
              '</div></div>')
        return h
    cum_block = _cum_block()
    cand_block = _uv_candidates()
    midev_block = _midev_cum_block()
    midev_today = _midev_today_block()

    # v79: 当日データ(_rare_data)はカード表示より前で1回だけ取得済み（上の _ex_line 参照）。

    def _rare_block(field="rare", thr=10.0):
        # v70: 閾値別に同じ表示を使う(≤10%=従来 / ≤20%=藤田指示の拡張版・記録のみ)。
        data = _rare_data
        title = ("&#127775; 希少・波乱の目（1号艇モーター激弱 ≤10%＝約1/180の希少・16点フォメで買い）" if field == "rare" else
                 "&#127775; 波乱の目・拡張版（1号艇モーター ≤20%＝1日約8件・16点フォメで買い）")
        color = "#b8860b" if field == "rare" else "#8a6d1f"
        if not data:
            return _pending_block(color, title)
        settled = {}
        for r in today:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                u = {}
            if (u.get(field) or {}).get("flag"):
                settled[(str(r.get("jcd")), str(r.get("rno")))] = (r, u.get(field))
        cands = []
        for jcd in range(1, 25):
            for rno in range(1, 13):
                rc = find_race(data, jcd, rno)
                if rc is None:
                    continue
                ex, tri = _payouts(rc)
                done = (ex is not None or tri is not None)
                rr = _rare_upset_pick(rc, thr=thr)
                if not rr.get("flag"):
                    continue
                key = (str(jcd), str(rno))
                closed = _tri_parse_closed(_tri_closed_at(rc), now.date())
                cands.append({"jcd": jcd, "rno": rno, "closed": closed, "rr": rr, "done": done,
                              "led": settled.get(key), "tri": tri, "rc": rc})
        cands.sort(key=lambda c: c["closed"] or now)
        n = len(cands)
        h = ('<div class="fcard" style="border-color:%s">'
             '<div class="tsub" style="font-weight:700;color:%s">%s　%d件</div>' % (color, color, title, n))
        if n == 0:
            h += ('<div class="tsub" style="margin-top:6px">本日は該当なし（1号艇の全国モーター2連率が%d%%以下のレースが出れば表示）。</div>'
                  % int(thr))
        else:
            h += ('<table style="margin-top:8px"><tr><th>締切</th><th>場R</th><th>1号艇モ</th>'
                  '<th>1号艇の展示</th><th>判定</th>'
                  '<th>買い目(頭-2着4艇-3着全=16点=16,000円)</th><th>結果</th></tr>')
            for c in cands:
                rr = c["rr"]
                combos = "%s-[%s]-[全] <span class=\"tsub\">(%d点=%s円)</span>" % (
                    rr.get("head"), "".join(str(x) for x in rr.get("second", [])),
                    len(rr.get("picks", [])), f"{int(rr.get('total', 0)):,}")
                # v74: 展示タイムで見送り判定(公開前は「展示待ち」)
                bi = None
                mins_left = ((c["closed"] - now).total_seconds() / 60.0) if c["closed"] else None
                if (not c["done"]) and mins_left is not None and -2 <= mins_left <= 25:
                    if _ex_rank1(c["rc"]) is None or _lane_desc(c["rc"]) is None:
                        bi = fetch_beforeinfo(hd, c["jcd"], c["rno"])
                exr = _ex_rank1(c["rc"], bi)
                ld = _lane_desc(c["rc"], bi)
                if ld is None:
                    lane_s = ''
                elif ld["wakunari"]:
                    lane_s = '<br><span style="font-size:11px">進入 <b style="color:#1c7a38">%s</b> <span style="color:#1c7a38">枠なり</span></span>' % ld["text"]
                else:
                    lane_s = ('<br><span style="font-size:11px">進入 <b style="color:#b8860b">%s</b> <span style="color:#b8860b">枠なり崩れ</span>%s</span>'
                              % (ld["text"], ('（1号艇は1コース）' if ld["c1"] == 1 else
                                              '<b style="color:#b23b3b">（1号艇が%dコース）</b>' % ld["c1"])))
                if exr is None and c["led"]:
                    lrr0 = c["led"][1] or {}
                    if lrr0.get("ex_rank1"):
                        exr = {"rank": lrr0["ex_rank1"], "ex1": lrr0.get("ex1")}
                if exr is None:
                    ex_s = '<span class="tsub">未公開</span>' + lane_s
                    jd = '<b style="color:#8a6d1f">&#9203; 展示待ち</b><br><span class="tsub" style="font-size:11px">締切10分前ごろ判明（1分ごとに確認）</span>'
                    row_style = ''
                elif exr["rank"] <= _RARE_EX_SKIP_RANK:
                    ex_s = '<b>%s位</b>（%.2f）' % (exr["rank"], exr["ex1"]) + lane_s
                    jd = ('<b style="color:#fff;background:#b23b3b;border-radius:6px;padding:2px 8px">&#128308; 見送り</b>'
                          '<br><span style="font-size:11px;color:#b23b3b">展示%s位＝エンジン好調</span>' % exr["rank"])
                    row_style = ' style="background:#f1f1f1;color:#8a97a5"'
                else:
                    ex_s = '<b>%s位</b>（%.2f）' % (exr["rank"], exr["ex1"]) + lane_s
                    jd = ('<b style="color:#fff;background:#1c7a38;border-radius:6px;padding:2px 8px">&#128994; 買い候補</b>'
                          '<br><span style="font-size:11px;color:#1c7a38">展示%s位＝弱さ確認</span>' % exr["rank"])
                    row_style = ' style="background:#eef8f0"'
                res = "—"
                if c["led"]:
                    lr, lrr = c["led"]
                    if lr.get("status") == "settled":
                        res = ('<span class="res-hit">的中</span>' if lrr and lrr.get("hit")
                               else '<span class="res-miss">不的中(%s)</span>' % lr.get("win_combo", ""))
                elif c["done"] and c.get("tri"):
                    # v70: 台帳に無いレース(デプロイ前・判定窓外)も公式結果で当否だけ表示(集計には入れない)
                    wc, amt = c["tri"]
                    hit_ = any(p["combo"] == wc for p in rr["picks"])
                    res = (('<span class="res-hit">的中 %s（%s円/100円）</span>' % (wc, f"{int(amt):,}")) if hit_
                           else '<span class="res-miss">不的中(%s)</span>' % wc)
                elif c["done"]:
                    res = "締切後"
                both = (' <span style="color:#b8860b;font-size:11px">≤10%も該当</span>'
                        if (field != "rare" and rr["mot1"] <= 10.0) else '')
                h += ('<tr%s><td>%s</td><td class="ct">%s%sR</td><td>%.1f%%%s</td><td>%s</td><td>%s</td>'
                      '<td class="combo" style="font-size:12px">%s</td><td>%s</td></tr>') % (
                    row_style, (c["closed"].strftime("%H:%M") if c["closed"] else "-"), ven(c["jcd"]), c["rno"],
                    rr["mot1"], both, ex_s, jd, combos, res)
            h += '</table>'
        if field == "rare":
            h += ('<div class="tsub" style="margin-top:6px"><b>見方：</b>朝から該当レースと買い目を出し、'
                  '<b>展示タイム（公式の直前情報を1分ごとに読み直し・この画面は1分ごとに自動更新）で1号艇が2位以内なら「🔴見送り」</b>、'
                  '3位以下なら「🟢買い（頭-2着4艇-3着全＝16点×1,000円）」に切り替わる。買い目は頭＝2-6号艇の最強機力、'
                  '2着＝1号艇を除く残り4艇、3着＝全艇（1号艇可）。</div>')
            h += ('<div class="tsub" style="margin-top:6px">※v77(9/18)フォーメーションで実際に買う枠（藤田指示）。'
                  '<b>過去2026 1-9月・展示3位以下限定の検証: 回収率36.7%（大穴3本抜き18.4%）＝まだ100%未満・利益は未実証。</b>'
                  '前向きに実配当で確かめる。展示で1号艇の遅さは確認できても公開情報のため配当が縮み、頭(強機力)が1着に来るのは約15%。'
                  '通算は /api/tri/summary の rare_upset で自動集計。</div></div>')
        else:
            h += ('<div class="tsub" style="margin-top:6px"><b>見方：</b>≤10%版と同じく、'
                  '<b>展示タイムで1号艇が2位以内なら「🔴見送り」、3位以下なら「🟢買い（16点×1,000円）」</b>（公開前は「展示待ち」）。'
                  '買い目は頭＝2-6号艇の最強機力、2着＝1号艇を除く残り4艇、3着＝全艇。</div>')
            h += ('<div class="tsub" style="margin-top:6px">※v77(9/18)フォーメーションで実際に買う枠（藤田指示）。'
                  '<b>過去2026 1-9月・展示3位以下限定の検証: 回収率65.4%（大穴3本抜き48.5%）＝まだ100%未満・利益は未実証。</b>'
                  '1日約8件×16,000円＝1日最大約12.8万円。前向きに実配当で確かめる枠。'
                  '通算は /api/tri/summary の rare_upset20 で自動集計。</div></div>')
        return h
    rare10_block = _rare_block("rare", 10.0)
    rare_block = _rare_block("rare20", 20.0)
    note = ('<div class="small" style="margin-top:14px">締切5分前(実際は約5〜10分前)に、その時点の最新データ＋3連単オッズで'
            '120通りを再計算し、買い/見送りを確定してそのまま保存します（結果を見てから予測は変えません）。'
            '「実運用」は買いと判定したレースのみ。「研究」は見送りを仮に買っていた場合で、実運用成績には含めません。'
            '控除率25%の壁は不変で、これは勝ちを保証しない検証ツールです。</div>')
    # v76: 的中率重視(本命型)は実際には買わないので、ページ上部から外して最下部に「参考・記録のみ」として置く
    kpi_ref = ('<div class="fcard" style="border-color:#dbe2ea"><div style="font-weight:700;color:#5c6b7a;font-size:14px">'
               '&#128203; 参考：的中率重視モデル（本命型・記録のみ／買いません）</div>' + kpi + '</div>')
    # v86: 「いつ買うのか」が最優先の情報なので、候補(=今日の予定)を実弾KPIの直後に置く。
    # v86: 並び順 = 実弾KPI → 今日の予定(買い候補) → 本日のC案/波乱 → 各レースのカード
    #      → 通算 → 記録のみバケット。「いつ買うのか」を最優先で上に出す。
    return HTMLResponse(head + bar + uv_live + cand_block + midev_today + rare10_block
                        + body + cum_block + midev_block + rare_block + uv_shadow
                        + kpi_ref + note + "</body></html>")


# ================= 常時稼働スケジューラ（Render Starter=常時ON）=================
# 締切約5〜10分前に自動で3連単を最終判定→固定保存し、確定レースを自動精算する。
# GitHubの不安定な定時実行に頼らず、常時稼働のRender内で回す。台帳は永続ディスクへ保存。
import threading

_TRI_DIR = os.environ.get("TRI_DATA_DIR") or ("/var/data" if os.path.isdir("/var/data") else "/tmp")
_TRI_LEDGER = os.path.join(_TRI_DIR, "tri_hit_ledger.csv")

# v90(2026-09-20 藤田指示): 3連単オッズ板(120点)を毎レース保存する。
#   背景: 台帳にはモデルが選んだ4〜8点のオッズしか残っていない。残り112点が見えないため
#   「モデルが見ていない買い目に歪みがあるか」「オッズ板の形そのものに歪みがあるか」を
#   検証できなかった。fetch_trifecta_odds() は judge のたびに120点を取得済みなので、
#   それを捨てずに書き出すだけで済む(新たなスクレイプは発生しない)。
#   ⚠ boatrace.jp は締切後のオッズを保持しないので過去には遡れない。今日からの蓄積のみ。
#   量: 1レース120行 × 1日約150レース = 1日約18,000行(約1MB)。1か月で約30MB。
_ODDS_LEDGER = os.path.join(_TRI_DIR, "odds3t_{hd}.csv")
_ODDS_COLS = ["ts", "date", "jcd", "rno", "combo", "odds"]
_odds_seen = set()          # (date,jcd,rno) 当プロセスで保存済み


def _odds_sanity(od):
    """オッズ板の健全性。3連単の控除率25%なら Σ(1/odds) ≈ 1/0.75 ≈ 1.333 になるはず。
    パースがずれていれば この値が大きく外れるので、壊れたデータの蓄積を自動で防げる。"""
    if not od or len(od) < 120:
        return False, 0.0, len(od or {})
    inv = 0.0
    for v in od.values():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v > 0:
            inv += 1.0 / v
    return (1.10 <= inv <= 1.70), round(inv, 4), len(od)


def _odds_save(hd, jcd, rno, od):
    """オッズ板をCSVへ追記。健全性チェックを通らなければ保存しない(ログだけ残す)。"""
    key = (str(hd), str(jcd), str(rno))
    if key in _odds_seen:
        return False
    ok, inv, n = _odds_sanity(od)
    if not ok:
        print(f"[odds3t] 異常のため保存せず date={hd} jcd={jcd} rno={rno} n={n} sum(1/odds)={inv}", flush=True)
        return False
    path = _ODDS_LEDGER.format(hd=hd)
    new = not os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            if new:
                w.writerow(_ODDS_COLS)
            ts = _tri_now().strftime("%Y-%m-%d %H:%M:%S")
            for combo, v in sorted(od.items()):
                w.writerow([ts, hd, jcd, rno, combo, v])
    except Exception as e:
        print(f"[odds3t] 書き込み失敗 {e}", flush=True)
        return False
    _odds_seen.add(key)
    return True
_TRI_COLS = ["ts", "date", "jcd", "rno", "closed_at", "decision", "stars", "conf", "honmei",
             "picks_json", "total", "reason", "status",
             "win_combo", "hit", "payout", "ret", "shadow_hit", "shadow_ret",
             "uv_json", "uv_hit", "uv_ret"]
_TRI_WIN_LO, _TRI_WIN_HI = 3, 13
# v65: 展示進入(直前情報)は締切のおよそ10〜13分前に出る。判定ウィンドウの入口(13分前)は
#   それより早いことが多く、ライブ台帳の実測では判定の約28%が「進入が出る前」に行われていた。
#   進入なし＝枠なり前提でモデルを回すことになり、過去BTでは dec40混戦 94.2% → 87.9% と
#   6pt落ちる(=バックテストと同じ土俵に立てていない)。
#   そこで「進入が出るまでは判定を先送りし、_TRI_LANE_FALLBACK 分前になったら諦めて判定する」。
#   スケジューラは150秒間隔なので13分前〜6分前の間に3回チャンスがある。
_TRI_LANE_FALLBACK = 7
_tri_lock = threading.Lock()


def _tri_now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))


def _tri_parse_closed(s, day0):
    if not s:
        return None
    s = str(s)
    try:
        if "T" in s or (":" in s and len(s) > 6):
            t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=dt.timezone(dt.timedelta(hours=9)))
            return t.astimezone(dt.timezone(dt.timedelta(hours=9)))
    except Exception:
        pass
    try:
        hh, mm = s.strip()[:5].split(":")
        return dt.datetime.combine(day0, dt.time(int(hh), int(mm)),
                                   tzinfo=dt.timezone(dt.timedelta(hours=9)))
    except Exception:
        return None


def _tri_closed_at(race):
    for k in ("race_closed_at", "closed_at", "race_close_time"):
        if isinstance(race, dict) and race.get(k):
            return str(race.get(k))
    return None


def tri_load_ledger():
    rows = []
    try:
        if os.path.exists(_TRI_LEDGER):
            with open(_TRI_LEDGER, newline="", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
    except Exception:
        rows = []
    return rows


def _tri_save_ledger(rows):
    os.makedirs(_TRI_DIR, exist_ok=True)
    tmp = _TRI_LEDGER + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=_TRI_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _TRI_COLS})
    os.replace(tmp, _TRI_LEDGER)


def _tri_top_combos(boats, n=3, odds_map=None):
    power = predict_power(boats)
    ranked = sorted(trifecta_probs_pl(power).items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for c, p in ranked[:n]:
        cs = f"{c[0]}-{c[1]}-{c[2]}"
        o = odds_map.get(cs) if odds_map else None
        out.append({"combo": cs, "P": round(p, 4), "hit_pct": round(p * 100, 1), "odds": o})
    return out


# v74(2026-09-16 藤田指示): 希少・波乱の目は「1号艇のモーターが数字上は激弱でも、展示タイムが2位以内なら見送り」。
#   過去258日の1号艇モーター≤10%(211R): 1号艇1着率 展示1〜2位 57.3%(96R・平均55%並み) / 3〜4位 42.2% / 5〜6位 37.3%。
#   ライブ(蒲郡・同一モーター7.1%): 9/05 展示2位→1号艇1着 / 9/07 3位→負け / 9/14 6位→負け / 9/15 2位→1号艇1着。
#   ※6点買いの的中は211Rで6本しかなく、展示で分けた回収率は判断不能(各2本以下)。見送り基準は1号艇1着率の差に基づく。
_RARE_EX_SKIP_RANK = 2
# v77(2026-09-18 藤田指示): 波乱の目(≤10%/≤20%)を「頭-2着4艇-3着全=16点」フォーメーションで実際に買う。1点1,000円。
_RARE_UNIT = 1000


# v75(2026-09-16 藤田指示「早くして」): 波乱の目の対象レースだけ、公式サイトの直前情報ページを直接読む。
#   openapi(約3分ごとに公式を写す)＋このアプリの3分キャッシュ＋画面更新2分で、9/16住之江1Rは
#   openapi反映15:08(締切9分前)→画面は最大15:13(4分前)だった。公式ページを60秒キャッシュで読めば数分早くなる。
#   取れなければ従来どおり openapi の値を使う(best-effort)。
_BI_CACHE = {}


def _parse_beforeinfo(html):
    """boatrace.jp 直前情報ページから {"ex": {枠: 展示タイム}, "lane": [コース1〜6の艇番]}。"""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None
    ex = {}
    for tb in soup.find_all("tbody"):
        tr = tb.find("tr")
        if not tr:
            continue
        tds = tr.find_all("td", recursive=False)
        if not tds:
            continue
        t0 = tds[0].get_text(strip=True)
        if not (t0.isdigit() and 1 <= int(t0) <= 6):
            continue
        if "is-boatColor" not in " ".join(tds[0].get("class") or []):
            continue
        for td in tds[1:]:
            s = td.get_text(strip=True)
            if re.fullmatch(r"\d\.\d{2}", s):
                v = float(s)
                if 6.0 <= v <= 7.6:
                    ex[int(t0)] = v
                break
    lane = None
    nums = [el.get_text(strip=True) for el in soup.select('[class*="table1_boatImage1Number"]')]
    nums = [int(n) for n in nums if n.isdigit()]
    if len(nums) >= 6 and sorted(nums[:6]) == [1, 2, 3, 4, 5, 6] and len(ex) >= 4:
        lane = nums[:6]     # 展示タイムが出ている時だけ信用する(公開前の既定並びを「枠なり」と誤表示しないため)
    if not ex and not lane:
        return None
    return {"ex": ex, "lane": lane}


def fetch_beforeinfo(hd, jcd, rno, ttl=60):
    """公式の直前情報(展示タイム・スタート展示の進入)を60秒キャッシュで取得。失敗は None。"""
    key = (str(hd), int(jcd), int(rno))
    now_t = time.time()
    ent = _BI_CACHE.get(key)
    if ent and now_t - ent[0] < ttl:
        return ent[1]
    res = None
    try:
        url = "https://boatrace.jp/owpc/pc/race/beforeinfo?rno=%d&jcd=%02d&hd=%s" % (int(rno), int(jcd), hd)
        rq = requests.get(url, headers=UA, timeout=6)
        if rq.status_code == 200:
            res = _parse_beforeinfo(rq.text)
    except Exception:
        res = None
    _BI_CACHE[key] = (now_t, res)
    return res


def _lane_desc(rc, bi=None):
    """展示進入の説明。{"text": "1-2-3-4-6-5", "wakunari": bool, "c1": 1号艇のコース} / 未公開 None。"""
    order = None
    if bi and bi.get("lane"):
        order = list(bi["lane"])
    else:
        try:
            lm = _lane_map(rc)
        except Exception:
            lm = {}
        if all(lm.get(f) for f in range(1, 7)):
            inv = {c: f for f, c in lm.items()}
            if len(inv) == 6:
                order = [inv[c] for c in range(1, 7)]
    if not order:
        return None
    return {"text": "-".join(str(x) for x in order), "wakunari": order == [1, 2, 3, 4, 5, 6],
            "c1": order.index(1) + 1}


def _ex_rank1(rc, bi=None):
    """直前情報の展示タイムで1号艇が何位か(1=最速・同タイムは1号艇を上位扱い)。展示未公開なら None。
    bi(公式直前情報)があればそれを優先。"""
    ex = {}
    if bi and len(bi.get("ex") or {}) >= 4 and 1 in bi["ex"]:
        ex = dict(bi["ex"])
    else:
        try:
            boats = extract_boats(rc)
        except Exception:
            return None
        for b in boats:
            try:
                v = float(b.get("ex"))
            except (TypeError, ValueError):
                continue
            if v > 0:
                ex[b["frame"]] = v
    if 1 not in ex or len(ex) < 4:
        return None
    rank = 1 + sum(1 for f, v in ex.items() if f != 1 and v < ex[1])
    return {"rank": rank, "ex1": ex[1], "n": len(ex), "best": min(ex.values())}


def _rare_upset_pick(rc, thr=10.0):
    # 🌟波乱の目: 1号艇の全国モーター2連率が極端に低い(≤thr%)時、最強機力(2-6号艇)を頭に、
    # 2着=1号艇を除く残り4艇、3着=全6艇(1号艇可)に流すフォーメーション=16点×1,000円=16,000円。
    # v77(2026-09-18 藤田指示): 展示で1号艇が3位以下(=ex_skip False)と確認できた時だけ実際に買う(cardの買い表示)。
    #   前提=「1号艇はモーター激弱＋展示も遅い時は1・2着に来ない」。検証(2026 1-9月・展示3位以下限定)は
    #   ≤10% 36.7% / ≤20% 65.4%(大穴3本抜き18〜49%)＝紙トレ検証。利益は未実証で前向きに実配当で確認する枠。
    try:
        mm = _uv_extract(rc)
    except Exception:
        return {"flag": False}
    def m(f):
        v = mm.get(f, {}).get("motor2")
        return 0.0 if (v is None or (isinstance(v, float) and v != v)) else float(v)
    mot1 = m(1)
    if mot1 <= 0 or mot1 > thr:
        return {"flag": False}
    others = sorted(range(2, 7), key=lambda f: m(f), reverse=True)
    head = others[0]                          # 2-6号艇で最強機力=頭(1着固定)
    second = [f for f in others if f != head]  # 1号艇を除く残り4艇=2着
    combos = []; seen = set()
    for b in second:
        for c in range(1, 7):                  # 3着は全艇(1号艇を含む)
            if c != head and c != b:
                cs = f"{head}-{b}-{c}"
                if cs not in seen:
                    seen.add(cs); combos.append({"combo": cs, "stake": _RARE_UNIT})
    exr = _ex_rank1(rc)
    return {"flag": True, "mot1": round(mot1, 1), "head": head, "second": second,
            "motors": {f: round(m(f), 1) for f in range(1, 7)}, "picks": combos, "total": len(combos) * _RARE_UNIT,
            "ex_rank1": (exr["rank"] if exr else None), "ex1": (exr["ex1"] if exr else None),
            "ex_skip": (bool(exr["rank"] <= _RARE_EX_SKIP_RANK) if exr else None)}


def _midev_today_live_count(rows, hd):
    """当日すでに実弾扱いにした中オッズEVのレース数。"""
    n = 0
    for r in rows:
        if str(r.get("date")) != str(hd):
            continue
        try:
            u = json.loads(r.get("uv_json") or "{}")
        except Exception:
            continue
        mv = (u or {}).get("midev") or {}
        if mv.get("picks") and mv.get("live"):
            n += 1
    return n


def _midev_with_cap(rows, hd, boats, odds_map):
    """中オッズEVの買い目を作り、1日の実弾上限内なら live=True を立てる。"""
    mv = midev_pick(boats, odds_map)
    if mv.get("picks"):
        mv["live"] = (_midev_today_live_count(rows, hd) < _MIDEV_DAILY_MAX)
        if not mv["live"]:
            mv["reason"] = (mv.get("reason", "") +
                            f" ※本日の実弾上限{_MIDEV_DAILY_MAX}レースに達したため、この分は記録のみ。")
    return mv


def tri_judge_cycle():
    now = _tri_now()
    hd = now.strftime("%Y%m%d")
    day0 = now.date()
    try:
        data = fetch_openapi(hd)
    except Exception:
        return 0
    if not data:
        return 0
    with _tri_lock:
        rows = tri_load_ledger()
        seen = {(r["date"], r["jcd"], r["rno"]) for r in rows}
        made = 0
        for jcd in range(1, 25):
            for rno in range(1, 13):
                key = (hd, str(jcd), str(rno))
                if key in seen:
                    continue
                rc = find_race(data, jcd, rno)
                if rc is None:
                    continue
                _, tri = _payouts(rc)
                if tri is not None:
                    continue
                closed = _tri_parse_closed(_tri_closed_at(rc), day0)
                if closed is None:
                    continue
                mins = (closed - now).total_seconds() / 60.0
                if not (_TRI_WIN_LO <= mins <= _TRI_WIN_HI):
                    continue
                # v65: 展示進入(直前情報)がまだ出ていないうちは判定を先送りする。
                #   7分前になっても出なければ、そのとき進入なしで判定する(記録に lane_known=False)。
                if (not _lane_known(rc)) and mins > _TRI_LANE_FALLBACK:
                    continue
                boats = extract_boats(rc)
                if sum(1 for b in boats if b.get("nat") is not None) < 4:
                    rows.append({"ts": now.strftime("%Y-%m-%d %H:%M:%S"), "date": hd, "jcd": jcd, "rno": rno,
                                 "closed_at": closed.strftime("%H:%M"), "decision": "見送り", "stars": 1,
                                 "conf": 0, "honmei": "", "picks_json": "[]", "total": 0,
                                 "reason": "直前データ不足のため見送り。", "status": "pending",
                                 "win_combo": "", "hit": "", "payout": "", "ret": "", "shadow_hit": "", "shadow_ret": ""})
                    seen.add(key); made += 1
                    continue
                odds_map = fetch_trifecta_odds(jcd, rno, hd)
                # v90: 取得済みの120点をそのまま保存(追加のスクレイプなし)
                if odds_map:
                    try:
                        _odds_save(hd, jcd, rno, odds_map)
                    except Exception as _e:
                        print("[odds3t]", _e, flush=True)
                res = trifecta_pick(boats, odds_map)
                picks = res["picks"] if res["decision"] == "買い" else _tri_top_combos(boats, 3, odds_map)
                # 過小評価キー判定も同時に実施（同じオッズを再利用）
                try:
                    uv = undervalue_pick(rc, odds_map)
                    uv_json = json.dumps({"decision": uv["decision"],
                                          "dec40": uv.get("dec40"), "dec55": uv.get("dec55"),
                                          "dec_stable": uv.get("dec_stable"),
                                          "dec_kp6": uv.get("dec_kp6"), "dec_kp7": uv.get("dec_kp7"),
                                          "dec_m50": uv.get("dec_m50"), "dec_m60": uv.get("dec_m60"),
                                          "dec_kp8": uv.get("dec_kp8"), "dec_conf20": uv.get("dec_conf20"),
                                          "dec40_ex": uv.get("dec40_ex"),
                                          "dec40_calm": uv.get("dec40_calm"),
                                          "dec40_ev12": uv.get("dec40_ev12"),
                                          "dec40_ev13": uv.get("dec40_ev13"),
                                          "dec40_lane": uv.get("dec40_lane"),
                                          "lane": uv.get("lane"),
                                          "stars": uv["stars"],
                                          "honmei": uv["honmei"], "key": uv["key"], "picks": uv["picks"],
                                          "picks_all": uv.get("picks_all"),
                                          "total": uv["total"], "basket_ev": uv["basket_ev"],
                                          "basket_ev8": uv.get("basket_ev8"),
                                          "reason": uv["reason"], "rare": _rare_upset_pick(rc),
                                          "rare20": _rare_upset_pick(rc, thr=20.0),
                                          "midev": _midev_with_cap(rows, hd, boats, odds_map),
                                          "droi": bestroi_pick(boats, odds_map)},
                                         ensure_ascii=False)
                except Exception:
                    uv_json = ""
                rows.append({"ts": now.strftime("%Y-%m-%d %H:%M:%S"), "date": hd, "jcd": jcd, "rno": rno,
                             "closed_at": closed.strftime("%H:%M"), "decision": res["decision"],
                             "stars": res["stars"], "conf": res["conf"], "honmei": res["honmei"],
                             "picks_json": json.dumps(picks, ensure_ascii=False),
                             "total": res.get("total", 0), "reason": res["reason"], "status": "pending",
                             "win_combo": "", "hit": "", "payout": "", "ret": "", "shadow_hit": "", "shadow_ret": "",
                             "uv_json": uv_json, "uv_hit": "", "uv_ret": ""})
                seen.add(key); made += 1
        # 精算（確定レース）
        for r in rows:
            if r.get("status") == "settled" or r.get("date") != hd:
                continue
            rc = find_race(data, int(r["jcd"]), int(r["rno"]))
            if rc is None:
                continue
            _, tri = _payouts(rc)
            if tri is None:
                continue
            win_combo, pay = tri[0], float(tri[1])
            try:
                picks = json.loads(r.get("picks_json") or "[]")
            except Exception:
                picks = []
            r["win_combo"] = win_combo
            if r["decision"] == "買い":
                ret = 0.0; hit = 0
                for p in picks:
                    if p.get("combo") == win_combo:
                        hit = 1; ret += (p.get("stake", 0) or 0) * (pay / 100.0)
                r["hit"] = hit; r["payout"] = int(pay) if hit else 0; r["ret"] = int(ret)
            else:
                sret = 0.0; shit = 0
                for p in picks:
                    if p.get("combo") == win_combo:
                        shit = 1; sret += 100 * (pay / 100.0)
                r["shadow_hit"] = shit; r["shadow_ret"] = int(sret)
            # 過小評価キーの精算: 買い目があれば常に計算(≥40/≥55 両モデルで共用。組・金額は同一)
            try:
                uv = json.loads(r.get("uv_json") or "{}")
            except Exception:
                uv = {}
            if uv.get("picks"):
                uret = 0.0; uhit = 0
                for p in uv.get("picks", []):
                    if p.get("combo") == win_combo:
                        uhit = 1; uret += (p.get("stake", 0) or 0) * (pay / 100.0)
                r["uv_hit"] = uhit; r["uv_ret"] = int(uret)
            # 🌟希少・波乱の目の精算(記録のみ・実運用に影響なし)
            for _rf in ("rare", "rare20"):      # v70: ≤20%拡張版も同じ方法で精算
                rr = uv.get(_rf) or {}
                if rr.get("flag") and rr.get("picks"):
                    rhit = 0; rret = 0.0
                    for p in rr["picks"]:
                        if p.get("combo") == win_combo:
                            rhit = 1; rret += (p.get("stake", 0) or 0) * (pay / 100.0)
                    rr["hit"] = rhit; rr["ret"] = int(rret); uv[_rf] = rr
            # 中オッズEV(記録のみ)の精算
            mv = uv.get("midev") or {}
            if mv.get("picks"):
                mhit = 0; mret = 0.0
                for p in mv["picks"]:
                    if p.get("combo") == win_combo:
                        mhit = 1; mret += (p.get("stake", 0) or 0) * (pay / 100.0)
                mv["hit"] = mhit; mv["ret"] = int(mret); uv["midev"] = mv
            # D案(並行検証)の精算
            dv = uv.get("droi") or {}
            if dv.get("picks"):
                dhit = 0; dret = 0.0
                for p in dv["picks"]:
                    if p.get("combo") == win_combo:
                        dhit = 1; dret += (p.get("stake", 0) or 0) * (pay / 100.0)
                dv["hit"] = dhit; dv["ret"] = int(dret); uv["droi"] = dv
            # 本番進入(スタート直前に確定)を記録。展示との食い違いが後から検証できる。
            if uv:
                try:
                    lr = _lane_real(rc, (uv.get("key") or {}).get("frame"), uv.get("honmei"))
                except Exception:
                    lr = None
                if lr:
                    ln = uv.get("lane") or {}
                    lr["changed"] = bool(ln.get("known") and ln.get("text") and ln["text"] != lr["text"])
                    uv["lane_real"] = lr
                r["uv_json"] = json.dumps(uv, ensure_ascii=False)
            r["status"] = "settled"
        _tri_save_ledger(rows)
        return made


# ================= 高オッズEV(2連単) 常駐スケジューラ版（durable台帳）=================
# tri と同じ設計: 締切直前に生オッズでEV判定→/var/data の台帳へ記録→確定後に自動精算。
# GitHubのcron(daily-autorun)に依存せず毎日勝手に貯まる。GitHub Actionで毎日コミットして永続化。
_HO_LEDGER2 = os.path.join(_TRI_DIR, "ho_live_ledger.csv")
_HO_COLS = ["ts", "date", "jcd", "rno", "closed_at", "strat", "combo", "P", "odds",
            "market_p", "EV", "stake", "status", "payout", "ret", "hit", "model_buy"]
_ho_lock = threading.Lock()


def ho_load_ledger():
    rows = []
    try:
        if os.path.exists(_HO_LEDGER2):
            with open(_HO_LEDGER2, newline="", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
    except Exception:
        rows = []
    return rows


def _ho_save_ledger(rows):
    os.makedirs(_TRI_DIR, exist_ok=True)
    tmp = _HO_LEDGER2 + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=_HO_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _HO_COLS})
    os.replace(tmp, _HO_LEDGER2)


def ho_judge_cycle():
    """高オッズEV(2連単)を常駐で判定→durable台帳へ記録→確定後に精算。"""
    now = _tri_now()
    hd = now.strftime("%Y%m%d")
    day0 = now.date()
    try:
        data = fetch_openapi(hd)
    except Exception:
        return 0
    if not data:
        return 0
    with _ho_lock:
        rows = ho_load_ledger()
        judged = {(r.get("date"), r.get("jcd"), r.get("rno")) for r in rows}
        made = 0
        for jcd in range(1, 25):
            for rno in range(1, 13):
                key = (hd, str(jcd), str(rno))
                if key in judged:
                    continue
                rc = find_race(data, jcd, rno)
                if rc is None:
                    continue
                ex, tri = _payouts(rc)
                if ex is not None:      # 既に確定=締切時オッズが取れない→判定不可でスキップ
                    continue
                closed = _tri_parse_closed(_tri_closed_at(rc), day0)
                if closed is None:
                    continue
                mins = (closed - now).total_seconds() / 60.0
                if not (1.0 <= mins <= 18.0):   # 締切1〜18分前の広めの窓(取りこぼし低減)
                    continue
                boats = extract_boats(rc)
                # tri判定直後はboatrace.jpのレート制限で2連単オッズが取れないことがある。
                # オッズが無くても uvpx50(オッズ非依存) は記録し、pos(オッズ必須)だけ空振りにする。
                odds_map = fetch_exacta_odds(jcd, rno, hd) or {}
                power = predict_power(boats)
                conf, gap = _conf_gap(power)
                exq = _exacta_probs(power)
                ca = closed.strftime("%H:%M")
                base = {"ts": now.strftime("%Y-%m-%d %H:%M:%S"), "date": hd, "jcd": jcd, "rno": rno, "closed_at": ca}
                any_rec = False
                # --- 戦略1(pos): 位置決め打ち 1-4/1-5(自信度ゲート)。1-4/1-5を両方記録し閾値検証可。
                #     model_buy=1 = 現行モデルが実際に買う条件(odds>=20 & EV>=1.2 & P>=0.05)。0=影。
                if conf >= 0.12 and gap >= 0.30:
                    for (i, j) in ((1, 4), (1, 5)):
                        cs = f"{i}-{j}"
                        o = odds_map.get(cs)
                        if o is None:
                            continue
                        p = exq.get((i, j), 0.0)
                        ev = p * o
                        mb = 1 if (o >= 20.0 and ev >= 1.20 and p >= 0.05) else 0
                        rows.append({**base, "strat": "pos", "combo": cs, "P": round(p, 4), "odds": o,
                                     "market_p": round(1.0 / o, 4) if o else "", "EV": round(ev, 3),
                                     "stake": 100, "status": "pending", "payout": "", "ret": "", "hit": "", "model_buy": mb})
                        any_rec = True
                # --- 戦略2(uvpx50): 過小評価キー2連単 P-X 1点(キーmotor2連率>=50・gap>=3・全国2連率>=5)。
                #     過去124日BTで ROI87.8%(最良・依然100%未満)。オッズ無視の機械買い=model_buy=1。
                res_uv = undervalue_pick(rc, None)
                ku = res_uv.get("key")
                if ku and float(ku.get("gap") or 0) >= 3 and float(ku.get("nat2") or 0) >= 5 and float(ku.get("motor2") or 0) >= 50:
                    Pp = res_uv["honmei"]; Xx = ku["frame"]; cs = f"{Pp}-{Xx}"
                    o = odds_map.get(cs)
                    p = exq.get((Pp, Xx), 0.0)
                    rows.append({**base, "strat": "uvpx50", "combo": cs, "P": round(p, 4),
                                 "odds": (o if o is not None else ""),
                                 "market_p": (round(1.0 / o, 4) if o else ""),
                                 "EV": (round(p * o, 3) if o else ""), "stake": 100,
                                 "status": "pending", "payout": "", "ret": "", "hit": "", "model_buy": 1})
                    any_rec = True
                if not any_rec:
                    rows.append({**base, "strat": "", "combo": "SKIP", "P": "", "odds": "", "market_p": "",
                                 "EV": "", "stake": 0, "status": "skip", "payout": "", "ret": "", "hit": "", "model_buy": 0})
                judged.add(key); made += 1
        # 精算(2連単の結果が出たレース)
        for r in rows:
            if r.get("status") != "pending" or r.get("date") != hd:
                continue
            rc = find_race(data, int(r["jcd"]), int(r["rno"]))
            if rc is None:
                continue
            ex, tri = _payouts(rc)
            if ex is None:
                continue
            wc, amt = ex
            hit = (r["combo"] == wc)
            r["payout"] = int(amt) if hit else 0
            r["ret"] = round(float(r["stake"]) * amt / 100.0, 1) if hit else 0.0
            r["hit"] = int(hit)
            r["status"] = "settled"
        _ho_save_ledger(rows)
        return made


def _tri_scheduler_loop():
    while True:
        try:
            if 8 <= _tri_now().hour <= 23:
                tri_judge_cycle()
        except Exception as e:
            print("[tri-sched]", e, flush=True)
        try:
            if 8 <= _tri_now().hour <= 23:
                ho_judge_cycle()
        except Exception as e:
            print("[ho-sched]", e, flush=True)
        time.sleep(150)


_tri_started = False


@app.on_event("startup")
def _tri_start():
    global _tri_started
    if _tri_started:
        return
    _tri_started = True
    threading.Thread(target=_tri_scheduler_loop, daemon=True).start()
    threading.Thread(target=_today_loop, daemon=True).start()
    print("[tri-sched] 常駐スケジューラ起動 ledger=" + _TRI_LEDGER, flush=True)


@app.get("/api/tri/run")
def api_tri_run():
    """手動でも今すぐ1周判定できる確認用エンドポイント。"""
    n = tri_judge_cycle()
    return {"ok": True, "made": n, "ledger": _TRI_LEDGER}


@app.get("/api/highodds/live_ledger.csv", response_class=PlainTextResponse)
def api_ho_live_ledger_csv():
    """高オッズ常駐台帳を生CSVで返す(GitHub Actionが毎日コミット→永続化・分析用)。"""
    try:
        if os.path.exists(_HO_LEDGER2):
            with open(_HO_LEDGER2, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return ",".join(_HO_COLS) + "\n"


@app.get("/api/highodds/live_run")
def api_ho_live_run():
    """手動で高オッズ常駐サイクルを1周回す確認用。"""
    n = ho_judge_cycle()
    return {"ok": True, "made": n, "ledger": _HO_LEDGER2}


_UVPX_CSS = """
*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#12263a;
font-family:-apple-system,BlinkMacSystemFont,"Hiragino Kaku Gothic ProN",sans-serif;padding:16px;max-width:900px;margin:0 auto}
h1{font-size:21px;margin:.2em 0}.sub{color:#5c6b7a;font-size:12.5px;line-height:1.6}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
.bar a{display:inline-block;padding:8px 12px;border-radius:10px;background:#eef2f7;color:#22303f;text-decoration:none;font-size:13px;border:1px solid #dbe2ea}
.kpi{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}
.kpi div{background:#fff;border:1px solid #dbe2ea;border-radius:12px;padding:10px 14px;min-width:96px}
.kpi .lab{font-size:11px;color:#5c6b7a}.kpi .val{font-size:19px;font-weight:800;margin-top:2px}
.good{color:#1c7a38}.bad{color:#c0392b}
.card{background:#fff;border:1px solid #dbe2ea;border-radius:14px;padding:14px;margin-top:12px}
table{width:100%;border-collapse:collapse;margin-top:6px}
th,td{border-bottom:1px solid #dbe2ea;padding:8px 6px;font-size:13px;text-align:center}
th{color:#5c6b7a;font-weight:600;font-size:11px}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
.combo{font-weight:800;font-size:15px}
.tsub{color:#8a97a5}
.note{background:#fff8e6;border:1px solid #f0e2b8;border-radius:10px;padding:10px;font-size:12px;color:#6b5a1f;margin-top:12px}
"""


@app.get("/uvpx", response_class=HTMLResponse)
def uvpx_page():
    """過小評価キー 2連単(P-X・キーmotor>=50)の当日買い目と通算成績。strat=uvpx50。"""
    rows = ho_load_ledger()
    uv = [r for r in rows if r.get("strat") == "uvpx50"]
    now = _tri_now()
    hd = now.strftime("%Y%m%d")

    def fn(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    def stats(rs):
        s = [r for r in rs if r.get("status") == "settled"]
        stake = sum(fn(r.get("stake")) for r in s)
        ret = sum(fn(r.get("ret")) for r in s)
        h = sum(1 for r in s if str(r.get("hit")) == "1")
        roi = (ret / stake * 100) if stake else 0
        return len(s), h, stake, ret, roi

    def ven(j):
        try:
            return _VEN.get(int(j), j)
        except Exception:
            return j

    today = [r for r in uv if r.get("date") == hd]
    tn, th, tstk, tret, troi = stats(today)
    cn, ch, cstk, cret, croi = stats(uv)

    order = sorted(today, key=lambda r: str(r.get("closed_at") or "z"))
    body_rows = []
    for r in order:
        o = r.get("odds")
        ev = r.get("EV")
        if r.get("status") == "settled":
            if str(r.get("hit")) == "1":
                res = '<span class="good">的中 &yen;' + "{:,}".format(int(fn(r.get("ret")))) + '</span>'
            else:
                res = '<span class="tsub">外れ</span>'
        else:
            res = '<span class="tsub">締切待ち</span>'
        odds_s = (str(o) + "倍") if (o not in (None, "", "0", 0)) else "&mdash;"
        ev_s = str(ev) if (ev not in (None, "")) else "&mdash;"
        body_rows.append(
            "<tr><td>" + str(r.get("closed_at") or "") + "</td><td>" + str(ven(r.get("jcd")))
            + "</td><td>" + str(r.get("rno")) + "R</td><td class=\"combo\">" + str(r.get("combo"))
            + "</td><td>" + odds_s + "</td><td>" + ev_s + "</td><td>" + res + "</td></tr>")
    if not body_rows:
        body_rows = ['<tr><td colspan="7" class="tsub">本日の対象レースはまだありません（キーがmotor&ge;50を満たすレースが締切前後に来ると自動で追加されます）。</td></tr>']

    troi_c = "good" if troi >= 100 else "bad"
    croi_c = "good" if croi >= 100 else "bad"
    nows = now.strftime("%Y-%m-%d %H:%M")

    html = (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="refresh" content="180">'
        '<title>過小評価キー 2連単</title><style>' + _UVPX_CSS + '</style></head><body>'
        '<h1>&#127775; 過小評価キー 2連単（P-X）</h1>'
        '<div class="sub">本命(P) &rarr; 過小評価キー(X) の 2連単1点。条件＝キーのモーター2連率&ge;50・gap&ge;3・全国2連率&ge;5。'
        '過去124日BTでROI87.8%（最良だが依然100%未満＝勝ち保証なし）。実オッズ込みで前向き検証中。1点100円。</div>'
        '<div class="bar"><a href="/uvpx">&#8635; 更新</a>'
        '<a href="/today">今日の高オッズEV</a><a href="/final">締切5分前 自動判定</a>'
        '<a href="/undervalue">過小評価キー(3連単)</a><a href="/dashboard">成績</a></div>'
        '<div class="kpi">'
        '<div><div class="lab">当日 記録</div><div class="val">' + str(tn) + '</div></div>'
        '<div><div class="lab">当日 的中</div><div class="val">' + str(th) + '</div></div>'
        '<div><div class="lab">当日 回収率</div><div class="val ' + troi_c + '">' + ("%.0f%%" % troi) + '</div></div>'
        '<div><div class="lab">通算 記録</div><div class="val">' + str(cn) + '</div></div>'
        '<div><div class="lab">通算 的中</div><div class="val">' + str(ch) + '</div></div>'
        '<div><div class="lab">通算 回収率</div><div class="val ' + croi_c + '">' + ("%.1f%%" % croi) + '</div></div>'
        '</div>'
        '<div class="card"><b>本日の買い目（P-X）</b>'
        '<table><tr><th>締切</th><th>場</th><th>R</th><th>買い目</th><th>オッズ</th><th>EV</th><th>結果</th></tr>'
        + "".join(body_rows) + '</table></div>'
        '<div class="note">&#9888; これは紙トレ検証です。過去124日でROI87.8%（最良）でも100%未満＝控除率25%の壁は越えていません。'
        '賭ける直前はオッズを公式で再確認を。金額は最小・据え置きで。通算は数百件貯まってから判断します。'
        '<br>データ更新: ' + nows + '（3分ごと自動更新）</div>'
        '</body></html>')
    return HTMLResponse(html)


@app.get("/api/tri/odds3t.csv", response_class=PlainTextResponse)
def tri_odds3t_csv(date: str = ""):
    """v90: その日の3連単オッズ板CSVを返す。GitHub Actions から取得して data/ へコミットする用。
    例: /api/tri/odds3t.csv?date=20260921"""
    hd = date or _tri_now().strftime("%Y%m%d")
    path = _ODDS_LEDGER.format(hd=hd)
    if not os.path.exists(path):
        return PlainTextResponse(",".join(_ODDS_COLS) + "\n")
    with open(path, encoding="utf-8") as f:
        return PlainTextResponse(f.read())


@app.get("/api/tri/odds3t/status")
def tri_odds3t_status(date: str = ""):
    """v90: 当日のオッズ板収集の状況(レース数・行数・最終更新)。"""
    hd = date or _tri_now().strftime("%Y%m%d")
    path = _ODDS_LEDGER.format(hd=hd)
    if not os.path.exists(path):
        return {"date": hd, "races": 0, "rows": 0, "exists": False}
    races = set(); rows = 0
    with open(path, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows += 1; races.add((r.get("jcd"), r.get("rno")))
    return {"date": hd, "races": len(races), "rows": rows, "exists": True,
            "bytes": os.path.getsize(path),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))}


@app.get("/api/tri/ledger.csv", response_class=PlainTextResponse)
def api_tri_ledger_csv():
    """tri台帳(uv_json入り)を生CSVで返す。GitHub Actionが毎日取得→data/へコミットし、
    レース単位の詳細分析を回せるようにする。秘密情報は含まない(紙トレ記録のみ)。"""
    try:
        if os.path.exists(_TRI_LEDGER):
            with open(_TRI_LEDGER, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return ",".join(_TRI_COLS) + "\n"


@app.get("/api/tri/summary")
def api_tri_summary():
    """成績サマリ(的中率重視 と 過小評価キー)。毎日の自動検証用に集計済みの数字を返す。"""
    rows = tri_load_ledger()
    jst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    hd = jst.strftime("%Y%m%d")

    def fnum(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    def agg_hit(rs):
        b = [r for r in rs if r.get("decision") == "買い" and r.get("status") == "settled"]
        hits = sum(1 for r in b if str(r.get("hit")) == "1")
        stake = sum(fnum(r.get("total")) for r in b)
        ret = sum(fnum(r.get("ret")) for r in b)
        skip = sum(1 for r in rs if r.get("decision") == "見送り")
        return {"buy": len(b), "hits": hits, "skip": skip, "stake": int(stake), "ret": int(ret),
                "roi": round(ret / stake * 100, 1) if stake else 0}

    def agg_uv(rs, field):
        bset = []
        skip = 0
        for r in rs:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                u = {}
            dec = _uv_dec(u, field)  # v50: 該当フィールドが無い行は除外(旧行がdecisionで混入しないように)
            if dec == "買い" and r.get("status") == "settled" and fnum(u.get("total")) > 0:
                bset.append((r, u))
            elif dec == "見送り":
                skip += 1
        hits = sum(1 for r, u in bset if str(r.get("uv_hit")) == "1")
        stake = sum(fnum(u.get("total")) for r, u in bset)
        ret = sum(fnum(r.get("uv_ret")) for r, u in bset)
        _pairs = [(fnum(u.get("total")), fnum(r.get("uv_ret"))) for r, u in bset]
        _lo, _hi, _se = _roi_ci95(_pairs)
        return {"buy": len(bset), "hits": hits, "skip": skip, "stake": int(stake), "ret": int(ret),
                "roi": round(ret / stake * 100, 1) if stake else 0,
                "roi_lo": (round(_lo, 1) if _lo is not None else None),
                "roi_hi": (round(_hi, 1) if _hi is not None else None)}

    def agg_uvp(rs, name):
        """v84: 実弾4点のうち条件を満たす点だけ買った場合の集計(記録のみ)。"""
        buy = hits = 0; stake = ret = 0.0; pts = 0
        for r in rs:
            if r.get("status") != "settled":
                continue
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                continue
            if not u:
                continue
            k, s_, r_ = uvp_result(r, u, name)
            if not k:
                continue
            buy += 1; pts += k; stake += s_; ret += r_
            hits += 1 if r_ > 0 else 0
        return {"buy": buy, "points": pts, "hits": hits, "stake": int(stake), "ret": int(ret),
                "roi": round(ret / stake * 100, 1) if stake else 0}

    def agg_midev(rs, shadow=None):
        """中オッズEV(記録のみ)の集計。shadow=影バケット名 なら派生買い目で集計。"""
        buy = hits = bset = 0; stake = ret = 0.0
        for r in rs:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                continue
            mv = (u or {}).get("midev") or {}
            if shadow:
                sub, s_, r_, st_ = midev_shadow_result(r, mv, shadow)
                if not sub:
                    continue
                buy += 1
                if st_:
                    bset += 1; stake += s_; ret += r_
                    hits += 1 if r_ > 0 else 0
                continue
            if not mv.get("picks"):
                continue
            buy += 1
            if r.get("status") == "settled" and str(mv.get("hit", "")) != "":
                bset += 1
                stake += _fnum(mv.get("total")); ret += _fnum(mv.get("ret"))
                if str(mv.get("hit")) == "1":
                    hits += 1
        return {"buy": buy, "settled": bset, "hits": hits, "stake": int(stake),
                "ret": int(ret), "roi": round(ret / stake * 100, 1) if stake else 0}

    def agg_uv_stop(rs, field="dec_stable", N=5):
        # A案: 安定型を「その日N連敗したらその日以降は買わない」ルールで集計(検証済み99.4%)。
        # settledの買いを日付ごと・時刻順に並べ、連敗がNに達したらその日の残りを除外。
        by_date = {}
        for r in rs:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                u = {}
            if u.get(field) == "買い" and r.get("status") == "settled":
                by_date.setdefault(r.get("date"), []).append((r, u))
        buy = hits = 0; stake = 0.0; ret = 0.0; stopped = 0
        for d, lst in by_date.items():
            lst.sort(key=lambda ru: str(ru[0].get("closed_at") or ru[0].get("ts") or ""))
            run = 0
            for r, u in lst:
                if run >= N:
                    stopped += 1
                    continue
                buy += 1; stake += fnum(u.get("total"))
                h = 1 if str(r.get("uv_hit")) == "1" else 0
                hits += h; ret += fnum(r.get("uv_ret"))
                run = 0 if h else run + 1
        return {"buy": buy, "hits": hits, "stopped": stopped, "stake": int(stake), "ret": int(ret),
                "roi": round(ret / stake * 100, 1) if stake else 0}

    def agg_rare(rs, field="rare", ex_ok_only=False):
        buy = hits = 0; stake = 0.0; ret = 0.0
        for r in rs:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                u = {}
            rr = u.get(field) or {}
            if ex_ok_only and rr.get("ex_skip") is not False:   # v74: 展示3位以下(=見送りでない)と確認できた分だけ
                continue
            if rr.get("flag") and rr.get("picks") and r.get("status") == "settled":
                buy += 1
                stake += sum((p.get("stake", 0) or 0) for p in rr["picks"])
                hits += 1 if rr.get("hit") else 0
                ret += fnum(rr.get("ret"))
        return {"buy": buy, "hits": hits, "stake": int(stake), "ret": int(ret),
                "roi": round(ret / stake * 100, 1) if stake else 0}

    def agg_droi(rs):
        buy = hits = 0; stake = 0.0; ret = 0.0
        for r in rs:
            try:
                u = json.loads(r.get("uv_json") or "{}")
            except Exception:
                u = {}
            d = u.get("droi") or {}
            if d.get("picks") and r.get("status") == "settled":
                buy += 1
                stake += sum((p.get("stake", 0) or 0) for p in d["picks"])
                hits += 1 if d.get("hit") else 0
                ret += fnum(d.get("ret"))
        return {"buy": buy, "hits": hits, "stake": int(stake), "ret": int(ret),
                "roi": round(ret / stake * 100, 1) if stake else 0}

    today = [r for r in rows if r.get("date") == hd]
    dates = sorted(set(r.get("date") for r in rows if r.get("date")))
    return {"ok": True, "hd": hd, "dates": dates,
            "today": {"hit": agg_hit(today), "uv40": agg_uv(today, "dec40"), "uv55": agg_uv(today, "dec55"),
                      "uv_stable": agg_uv(today, "dec_stable"), "uv_stable_stop": agg_uv_stop(today),
                      "rare_upset": agg_rare(today),
                      "rare_upset20": agg_rare(today, "rare20"),
                      "rare_upset_exok": agg_rare(today, "rare", True),
                      "uv40_kp6": agg_uv(today, "dec_kp6"), "uv40_kp7": agg_uv(today, "dec_kp7"),
                      "uv40_m50": agg_uv(today, "dec_m50"), "uv40_m60": agg_uv(today, "dec_m60"),
                      "uv40_kp8": agg_uv(today, "dec_kp8"), "uv40_conf20": agg_uv(today, "dec_conf20"),
                      "uv40_ex": agg_uv(today, "dec40_ex"),
                      "uv40_calm": agg_uv(today, "dec40_calm"),
                      "uv40_ev12": agg_uv(today, "dec40_ev12"),
                      "uv40_ev13": agg_uv(today, "dec40_ev13"),
                      "uv40_lane": agg_uv(today, "dec40_lane"),
                      "uv40_stable": agg_uv(today, "dec40_stable"),
                      "uv40_in": agg_uv(today, "dec40_in"),
                      "uv40_in_ev": agg_uv(today, "dec40_in_ev"),
                      "uv_in_ev12": agg_uv(today, "dec_in_ev12"),
                      "uvp": {nm: agg_uvp(today, nm) for nm, _, _ in _UVP_FILTERS},
                      "midev": agg_midev(today),
                      "midev_ev1ev2": agg_midev(today, "midev_ev1ev2"),
                      "midev_n23": agg_midev(today, "midev_n23"),
                      "midev_sel": agg_midev(today, "midev_sel"),
                      "midev_ev2n2": agg_midev(today, "midev_ev2n2"),
                      "midev_ev2h1": agg_midev(today, "midev_ev2h1"),
                      "midev_ev2gap": agg_midev(today, "midev_ev2gap"),
                      "droi": agg_droi(today)},
            "cumulative": {"hit": agg_hit(rows), "uv40": agg_uv(rows, "dec40"), "uv55": agg_uv(rows, "dec55"),
                           "uv_stable": agg_uv(rows, "dec_stable"), "uv_stable_stop": agg_uv_stop(rows),
                           "rare_upset": agg_rare(rows),
                           "rare_upset20": agg_rare(rows, "rare20"),
                           "rare_upset_exok": agg_rare(rows, "rare", True),
                           "uv40_kp6": agg_uv(rows, "dec_kp6"), "uv40_kp7": agg_uv(rows, "dec_kp7"),
                           "uv40_m50": agg_uv(rows, "dec_m50"), "uv40_m60": agg_uv(rows, "dec_m60"),
                           "uv40_kp8": agg_uv(rows, "dec_kp8"), "uv40_conf20": agg_uv(rows, "dec_conf20"),
                           "uv40_ex": agg_uv(rows, "dec40_ex"),
                           "uv40_calm": agg_uv(rows, "dec40_calm"),
                           "uv40_ev12": agg_uv(rows, "dec40_ev12"),
                           "uv40_ev13": agg_uv(rows, "dec40_ev13"),
                           "uv40_lane": agg_uv(rows, "dec40_lane"),
                           "uv40_stable": agg_uv(rows, "dec40_stable"),
                           "uv40_in": agg_uv(rows, "dec40_in"),
                           "uv40_in_ev": agg_uv(rows, "dec40_in_ev"),
                           "uv_in_ev12": agg_uv(rows, "dec_in_ev12"),
                           "uvp": {nm: agg_uvp(rows, nm) for nm, _, _ in _UVP_FILTERS},
                           "midev": agg_midev(rows),
                           "midev_ev1ev2": agg_midev(rows, "midev_ev1ev2"),
                           "midev_n23": agg_midev(rows, "midev_n23"),
                           "midev_sel": agg_midev(rows, "midev_sel"),
                           "midev_ev2n2": agg_midev(rows, "midev_ev2n2"),
                           "midev_ev2h1": agg_midev(rows, "midev_ev2h1"),
                           "midev_ev2gap": agg_midev(rows, "midev_ev2gap"),
                           "droi": agg_droi(rows)}}


# ================= 過小評価モーターキー・3連単EVモデル =================
# 戸田7R(5-1-3)の勝ち筋をモデル化。検証: 不人気(選手2連率低)×モーター良の艇は3着以内+4pt(31→35.5%)。
# ボート2連率は3着以内を予測しない(相関0.002)ので参考のみ。妙味はライブオッズでのみ測れる=前向き検証。
def _uv_extract(race):
    """各枠の 全国2連率/モータ2連率/ボート2連率 を取り出す。"""
    ents = []
    _collect_with_key(race, "national_win_rate", ents)
    by = {}
    for e in ents:
        n = e.get("entry_number")
        if not n:
            continue
        by[int(n)] = {
            "nat2": _num(e.get("national_top_2_percent"), np.nan),
            "motor2": _num(e.get("motor_top_2_percent"), np.nan),
            "boat2": _num(e.get("boat_top_2_percent"), np.nan),
        }
    return by


def _marginal_top3(power):
    """各枠が3着以内(1-3着のどこか)に入るモデル確率。"""
    tri = trifecta_probs_pl(power)
    out = {f: 0.0 for f in range(1, 7)}
    for (i, j, k), p in tri.items():
        out[i] += p; out[j] += p; out[k] += p
    return out


def _roi_ci95(pairs):
    """回収率(=Σ払戻/Σ投資)の95%信頼区間。pairs=[(投資,払戻),...]（精算済みのみ）。
    比推定量の分散 Var ≈ n/(n-1) * Σ(r_i - roi*s_i)^2 / S^2 を正規近似したもの。
    払戻の裾が重い(数本の大穴が支配)ため、これは「最低でもこれくらい広い」目安。
    戻り値: (下限%, 上限%, 標準誤差%) / 計算不能なら (None, None, None)。"""
    n = len(pairs)
    S = sum(x[0] for x in pairs)
    R = sum(x[1] for x in pairs)
    if n < 2 or S <= 0:
        return (None, None, None)
    roi = R / S
    v = sum((r - roi * st) ** 2 for st, r in pairs) * n / (n - 1) / (S * S)
    se = math.sqrt(v) if v > 0 else 0.0
    return (max(0.0, 100 * (roi - 1.96 * se)), 100 * (roi + 1.96 * se), 100 * se)


def _uv_stars(gap, basket_ev):
    s = 1
    if gap >= 2: s += 1
    if gap >= 3: s += 1
    if basket_ev >= 1.0: s += 1
    if basket_ev >= 1.3: s += 1
    return min(5, s)


def undervalue_pick(race, odds_map):
    boats = extract_boats(race)
    power = predict_power(boats)                 # モデルP(1着)
    top3 = _marginal_top3(power)                 # モデルP(3着以内)
    mm = _uv_extract(race)
    frames = list(range(1, 7))
    # 人気ランク(全国2連率が高い=人気, rank1=最人気) / 機力ランク(モータ2連率が高い, rank1=最良)
    nat = {f: (mm.get(f, {}).get("nat2") if not np.isnan(mm.get(f, {}).get("nat2", np.nan)) else 0.0) for f in frames}
    mot = {f: (mm.get(f, {}).get("motor2") if not np.isnan(mm.get(f, {}).get("motor2", np.nan)) else 0.0) for f in frames}
    nat_rank = {f: r for r, f in enumerate(sorted(frames, key=lambda x: nat[x], reverse=True), 1)}
    mot_rank = {f: r for r, f in enumerate(sorted(frames, key=lambda x: mot[x], reverse=True), 1)}
    # 過小評価ギャップ = 人気ランク - 機力ランク (正=機力の割に不人気=妙味)
    gap = {f: nat_rank[f] - mot_rank[f] for f in frames}
    # 候補: モーター2連率>=25 かつ ギャップ>=2 かつ 本命級でない(モデルP(1着)が最上位ではない)
    honmei = max(frames, key=lambda f: power.get(f, 0))
    cand = [f for f in frames if mot[f] >= 25 and gap[f] >= 2 and f != honmei]
    if not cand:
        # v65: dec_stable 等が抜けていて画面に「安定型:None」と出ていたため、全バケットを埋める。
        #      進入は買い/見送りに関係なく毎レース見たいので lane もここで返す。
        _lane0 = _lane_map(race)
        _known0 = all(_lane0.get(f) for f in range(1, 7))
        _inv0 = {c: f for f, c in _lane0.items()} if _known0 else {}
        return {"decision": "見送り", "dec40": "見送り", "dec55": "見送り", "stars": 1,
                "dec_stable": "見送り",
                "dec_kp6": "見送り", "dec_kp7": "見送り",
                "dec_m50": "見送り", "dec_m60": "見送り",
                "dec_kp8": "見送り", "dec_conf20": "見送り",
                "dec40_ex": "見送り", "dec40_calm": "見送り",
                "dec40_ev12": "見送り", "dec40_ev13": "見送り", "dec40_lane": "見送り",
                "lane": {"known": _known0,
                         "wakunari": bool(_known0 and all(_lane0[f] == f for f in range(1, 7))),
                         "text": ("-".join(str(_inv0[c]) for c in range(1, 7)) if len(_inv0) == 6 else ""),
                         "key_out": False, "key_in": False, "honmei_moved": False},
                "honmei": honmei, "key": None, "picks": [], "total": 0, "basket_ev": 0,
                "reason": f"過小評価キー(機力良×不人気)が見当たらないため見送り。本命{honmei}号艇。"}
    X = max(cand, key=lambda f: (gap[f], mot[f]))    # ギャップ最大→モーター最良
    # 本命勢P,Q,R = モデルP(1着)上位(Xを除く)
    order = [f for f in sorted(frames, key=lambda f: power.get(f, 0), reverse=True) if f != X]
    P, Q, R = order[0], order[1], order[2]
    # 買い目: Xを3着中心・2着少し・1着1点で絡める(戸田7Rの型)
    combos = [((P, Q, X), 200, "3着(本命順)"), ((Q, P, X), 200, "3着(本命逆)"), ((P, R, X), 200, "3着(3番手)"),
              ((P, X, Q), 150, "2着"), ((Q, X, P), 150, "2着(逆)"), ((P, X, R), 100, "2着(裏)"),
              ((X, P, Q), 100, "1着(上振れ)"), ((X, Q, P), 100, "1着(裏)")]
    # v59(2026-09-06 ユーザ承認): キーを1着に置く2点は実弾から外す(8点1200円→6点1000円)。
    #   ライブ9日(≥40買い286R)の役割別収支: 3着3点77.9% / 2着2点86.7% / 2着(裏)97.6% に対し
    #   1着(上振れ)26.5% / 1着(裏)11.5% ＝572点に57,200円投じて的中2本・払戻10,060円の死に金。
    #   この2点を外すだけで ≥40 は 64.8%→74.3% に改善。ただし常滑370倍(キー1着)の型は今後拾えない。
    # 重要: ゲート計算(odds_ok の max_odds、basket_ev)は従来どおり8点ベースのまま維持する。
    #   6点ベースに変えると odds_ok の意味が変わり、買うレースの集合そのものが過去の検証とズレるため。
    # v62(2026-09-06 ユーザ承認): さらに「3着(本命順)=P-Q-X」「3着(3番手)=P-R-X」も実弾から外す(6点1,000円→4点600円)。
    #   ライブ10日(≥40買い286R・8点記録)の点別収支:
    #     P-Q-X 30.1%(95%区間 6〜54) / P-R-X 24.5%(0〜55) ← 区間上限が100%を大きく下回る=構造的に負け
    #     Q-P-X 114.4% / P-X-Q 102.4% / Q-X-P 106.7% / P-X-R 90.8% ← 残す4点
    #   P-Q-X は8点中いちばん人気(平均41倍・中央値29倍)で的中7本と多いのに1本あたり払戻2,463円=薄い。
    #   ＝「本命1位→2位→キー3着」という最もありがちな並びは群衆が買いすぎ、控除率25%を超えられない。
    #   1日ずつ抜くjackknifeで9日すべて 6点(61〜82%) < 2点除外(83〜121%)。特定日の1発依存ではない。
    #   ただし的中率は13.3%→9.8%に低下（連敗は長くなる）。4点セット自体の95%区間は48〜163%で、
    #   「勝ちが証明された」わけではない。証明できたのは「外した2点が明確に負けている」ことだけ。
    # 重要: ゲート計算(odds_ok の max_odds、basket_ev8)は従来どおり8点ベースのまま維持する。
    _UV_DROP = ("1着(上振れ)", "1着(裏)", "3着(本命順)", "3着(3番手)")
    # v64(2026-09-06 ユーザ指示): 実際に賭ける金額を「1点1,000円の均等」に変更(4点=4,000円/レース)。
    #   ・均等フラットは運用ルールv54の検証どおり(ダッチング=オッズ配分は期待値不変・均等が最善)。
    #   ・重み配分が変わるため過去BTの数字も微妙に変わる: 4点は傾斜600円で105.6% → 均等で103.6%。
    #   ・combos に書いた 200/150/100 は【ゲート計算専用の従来ウェイト】として据え置く。
    #     odds_ok(max_odds) と basket_ev8 を従来8点ベースのまま保つことで、
    #     「買うレースの集合」を過去の検証と地続きにする(ここを動かすと全部の数字が繋がらなくなる)。
    _UV_UNIT = 1000   # 1点あたりの実際の賭け金
    tri = trifecta_probs_pl(power)
    picks = []
    picks_all = []                      # v62: 外した点も含む全8点(後から点別に検証できるようにするため)
    exp_ret = 0.0; total = 0            # 実際に賭ける点(v62=4点600円)
    exp_ret8 = 0.0; total8 = 0          # 判定用の従来8点(据え置き)
    odds_all = []
    for (ijk, stake, pos) in combos:
        cs = f"{ijk[0]}-{ijk[1]}-{ijk[2]}"
        p = tri.get(ijk, 0.0)
        o = odds_map.get(cs) if odds_map else None
        ev = round(p * o, 2) if o else None
        payout = int(round(stake * o)) if o else None
        total8 += stake
        if o:
            exp_ret8 += stake * p * o
            odds_all.append(o)
        bet = _UV_UNIT               # v64: 実際の賭け金は1点1,000円の均等
        rec = {"combo": cs, "pos": pos, "P": round(p, 4), "hit_pct": round(p * 100, 1),
               "odds": o, "ev": ev, "stake": bet,
               "payout": (int(round(bet * o)) if o else None)}
        # v85: 形で外す4点に加えて、点ごとのオッズ上限とEVフロアでも落とす。
        # 朝の候補リストはオッズ未取得(odds_map が空)で呼ばれる。そこでは絞り込めないので
        # 従来どおり4点を暫定表示し、締切前の判定(オッズあり)でだけ絞る。
        _leg_ok = True
        if odds_map and pos not in _UV_DROP:
            if _UV_LEG_ODDS_CAP is not None and (not o or o > _UV_LEG_ODDS_CAP):
                _leg_ok = False
            if _leg_ok and _UV_LEG_EV_FLOOR is not None and (not o or p * o < _UV_LEG_EV_FLOOR):
                _leg_ok = False
        picks_all.append(dict(rec, bet=(pos not in _UV_DROP and _leg_ok)))
        if pos in _UV_DROP:
            continue                     # 賭けない(ゲート計算には従来どおり参加させる)
        if not _leg_ok:
            continue                     # v85: オッズ上限/EVフロアで落とした点(ゲート計算には参加済み)
        picks.append(rec)
        total += bet
        if o:
            exp_ret += bet * p * o
    basket_ev = round(exp_ret / total, 3) if total else 0
    basket_ev8 = round(exp_ret8 / total8, 3) if total8 else 0   # 影バケットのEVフロア判定用
    max_odds = max(odds_all, default=0)
    odds_ok = (max_odds >= 15)
    # 全国2連率フロア: キーの全国2連率0〜4%は実力的に着に絡めず回収率35%と大負け(過去182日で検証)。
    # 5%以上に限定すると≥40:71→77%・≥55:90→97%へ改善し、100倍以上の大穴もほぼ全て残る(112/121・21/22本)。
    nat_ok = (nat[X] >= 5)
    # キー自身の1着確率(モデルP(1着))。過去182日検証: keyp>=10%は回収率90.5%で全7か月中6か月>keyp<10%=構造的に安定。
    # keyp<10%(買いの85%)は全月100%割れ。大穴の87%はkeyp<10%帯から出るため、安定型では大穴はほぼ消える(トレードオフ)。
    keyp = float(power.get(X, 0.0))
    keyp_ok = (keyp >= 0.10)
    keyp_ok6 = (keyp >= 0.06)   # 追加(v50): ライブ候補フロア keyp>=6%
    keyp_ok7 = (keyp >= 0.07)   # 追加(v50): ライブ候補フロア keyp>=7%
    keyp_ok8 = (keyp >= 0.08)   # 追加(v51): 見送り検証 keyp>=8%
    conf_uv = _conf_gap(power)[0]   # 本命自信度(v51: conf>=0.2 見送り検証用)
    pgap_uv = _conf_gap(power)[1]   # 本命の抜け=パワー1位と2位の差/1位。小さい=本命が飛び抜けてない混戦
    # 3つを並走: ≥40(大穴型) / ≥55(厳選) / 安定型(≥40 かつ keyp≥10%=キーに地力あり)。全て全国2連率≥5%フロア適用。
    # v52(2026-09-02 ユーザ承認): ≥40 を「本命が抜けてない混戦(pgap<=0.50)」だけに限定。
    #   半年BT 76.1%->91.9%・現行超え5/6月・大穴3本抜き85.5%(頑健)。的中率19->15%だが1発が高配当。依然100%未満(約-8%)。
    dec40 = "買い" if (odds_map and gap[X] >= 3 and mot[X] >= 40 and odds_ok and nat_ok and pgap_uv <= 0.50) else "見送り"
    dec55 = "買い" if (odds_map and gap[X] >= 3 and mot[X] >= 55 and odds_ok and nat_ok) else "見送り"
    dec_stable = "買い" if (odds_map and gap[X] >= 3 and mot[X] >= 40 and odds_ok and nat_ok and keyp_ok) else "見送り"
    # 追加(v50): ≥40条件 に keyp フロア6%/7% を掛けた影バケット(実際の買いは変えない・記録のみ)
    dec_kp6 = "買い" if (odds_map and gap[X] >= 3 and mot[X] >= 40 and odds_ok and nat_ok and keyp_ok6) else "見送り"
    dec_kp7 = "買い" if (odds_map and gap[X] >= 3 and mot[X] >= 40 and odds_ok and nat_ok and keyp_ok7) else "見送り"
    # 追加(v50-2): motorフロアの影バケット(≥40条件のmotを50/60に上げたもの・記録のみ)
    dec_m50 = "買い" if (odds_map and gap[X] >= 3 and mot[X] >= 50 and odds_ok and nat_ok) else "見送り"
    dec_m60 = "買い" if (odds_map and gap[X] >= 3 and mot[X] >= 60 and odds_ok and nat_ok) else "見送り"
    # 追加(v51): 見送り検証の対決 — keyp>=8% と 本命自信度conf>=0.2 を並走(記録のみ)
    dec_kp8 = "買い" if (odds_map and gap[X] >= 3 and mot[X] >= 40 and odds_ok and nat_ok and keyp_ok8) else "見送り"
    dec_conf20 = "買い" if (odds_map and gap[X] >= 3 and mot[X] >= 40 and odds_ok and nat_ok and conf_uv >= 0.20) else "見送り"
    # 追加(v56): ≥40混戦 から「1号艇の展示タイムが場内下位(5-6位=不調)」を除外する影バケット。
    # 過去BT: 混戦94.8%→展示不調除外で100.4%(月別5/7改善だが6-7月は悪化)。本命(≒1号艇)が仕上がってない=軸が不安、を外す。
    bd_ex = {b.get("frame"): b for b in boats}
    ex1 = bd_ex.get(1, {}).get("ex")
    exs_all = [bd_ex[f].get("ex") for f in range(1, 7) if f in bd_ex and bd_ex[f].get("ex") is not None]
    ex1_rank = (1 + sum(1 for v in exs_all if v < ex1)) if (ex1 is not None and len(exs_all) >= 5) else None
    ex_bad = (ex1_rank is not None and ex1_rank >= 5)   # 1号艇展示5-6位=不調(データ欠損は除外しない=買い側)
    dec40_ex = "買い" if (dec40 == "買い" and not ex_bad) else "見送り"
    # 追加(v57): ≥40混戦 から「荒れ水面(波高≥4cm)」を除外する影バケット。
    # 過去BT(2143R): 混戦94.8%→波<4cmで101.0%(買い数77%残る)/波≥4cmは73.8%と明確に負け。
    # キーは機力(実力)がクリーンなスピード勝負で顕在化する賭け=荒れると結果がランダム化して壊れる。データ欠損は除外しない(=買い側)。
    _prev_uv = (race.get("preview") or {})
    _wave_uv = _prev_uv.get("wave_height")
    wave_rough = False
    if _wave_uv is not None:
        try: wave_rough = (float(_wave_uv) >= 4)
        except (TypeError, ValueError): wave_rough = False
    dec40_calm = "買い" if (dec40 == "買い" and not wave_rough) else "見送り"
    # 追加(v59): ≥40混戦 に「モデル期待値(basket_ev)フロア」を掛けた影バケット。記録のみ・実弾は変えない。
    # ライブ9日(≥40買い286R)で初めて測れた軸(過去データにオッズが無く従来は検証不能だった)。
    #   ev>=1.2=80.5%(148R) / ev>=1.3=85.7%(131R) に対し ev<1.2=48.0%(138R)＝捨てる側が明確に負け。
    #   ただし大穴3本抜きで39%、100%超は9日中3日＝まだ勝ち確定ではないので影バケットで並走観測する。
    # 判定は従来どおり8点ベースの basket_ev8 を使う(6点化前の数値と地続きにして比較可能にするため)。
    dec40_ev12 = "買い" if (dec40 == "買い" and basket_ev8 >= 1.2) else "見送り"
    dec40_ev13 = "買い" if (dec40 == "買い" and basket_ev8 >= 1.3) else "見送り"
    # 追加(v65): 展示進入ゲート。記録のみ(実弾は変えない)。
    #   過去170日(dec40混戦の買い2,052件)の実測:
    #     キーXが展示で外に押し出された      120件 ROI 66.2%  ← 負ける
    #     本命Pが展示で枠番から動いた         77件 ROI 約50%  ← 負ける
    #     キーXが展示で内に入った(前づけ成功)  67件 ROI100.7%
    #     キーXが展示で枠なり              1,865件 ROI 95.8%
    #   この2条件を外すだけで 8点94.2%→97.3%(大穴3本抜き97.5%)、実弾4点101.7%→104.9%。
    #   170日を1日ずつ抜くjackknifeで170/170日とも改善が保たれる(特定日の一発依存ではない)。
    #   理屈: この賭けは「機力のあるキーが3着に飛び込む」に賭けている。キーが外に回されると
    #   その前提が崩れ、本命が動くとレース全体の並びが読めなくなる。
    _lane = _lane_map(race)
    lane_known = all(_lane.get(f) for f in range(1, 7))
    lane_wakunari = bool(lane_known and all(_lane[f] == f for f in range(1, 7)))
    key_pushed_out = bool(lane_known and _lane[X] > X)
    key_moved_in = bool(lane_known and _lane[X] < X)
    honmei_moved = bool(lane_known and _lane[P] != P)
    lane_bad = bool(key_pushed_out or honmei_moved)
    lane_str = _lane_text(race)
    dec40_lane = "買い" if (dec40 == "買い" and not lane_bad) else "見送り"
    if odds_map and not picks:
        # v85: レースとしては条件を満たすが、4点のいずれもオッズ上限/EVフロアを通らなかった
        return {"decision": "見送り", "dec40": dec40, "dec55": dec55, "dec_stable": dec_stable,
                "dec_kp6": dec_kp6, "dec_kp7": dec_kp7, "dec_m50": dec_m50, "dec_m60": dec_m60,
                "dec_kp8": dec_kp8, "dec_conf20": dec_conf20, "dec40_ex": dec40_ex,
                "dec40_calm": dec40_calm, "dec40_ev12": dec40_ev12, "dec40_ev13": dec40_ev13,
                "dec40_lane": dec40_lane,
                "lane": {"known": lane_known, "wakunari": lane_wakunari, "text": lane_str,
                         "key_out": key_pushed_out, "key_in": key_moved_in, "honmei_moved": honmei_moved},
                "stars": _uv_stars(gap[X], basket_ev), "honmei": P,
                "key": {"frame": X, "motor2": mot[X], "nat2": nat[X], "top3": round(top3[X], 3),
                        "gap": gap[X], "keyp": round(keyp, 3)},
                "picks": [], "picks_all": picks_all, "total": 0,
                "basket_ev": basket_ev, "basket_ev8": basket_ev8,
                "reason": (f"{X}号艇の過小評価キーは立っているが、買い目4点のいずれも"
                           f"オッズ{int(_UV_LEG_ODDS_CAP)}倍以下かつ1点EV{_UV_LEG_EV_FLOOR}以上を満たさないため買わない(v85)。")}
    reason = (f"{X}号艇＝モーター2連率{mot[X]:.0f}%(機力{mot_rank[X]}位)なのに全国2連率{nat[X]:.0f}%(人気{nat_rank[X]}位)＝"
              f"実力の割に不人気で高オッズ。本命{P}号艇を軸に、{X}を3着中心で絡めて{len(picks)}点。"
              f"モデルは{X}の3着以内を{top3[X]*100:.0f}%と評価。")
    if not nat_ok:
        reason += f" ただしキーの全国2連率{nat[X]:.0f}%は5%未満＝実力不足帯(回収率35%)のため見送り。"
    if nat_ok and pgap_uv > 0.50:
        reason += f" ただし本命の抜け(pgap{pgap_uv:.2f})が大きい=本命が飛び抜けており妙味薄。≥40は混戦(pgap≤0.50)限定のため見送り。"
    if not lane_known:
        reason += " ※展示進入(直前情報)がまだ出ていないため、枠なり前提で計算した暫定値。"
    elif lane_wakunari:
        reason += f" 進入は枠なり({lane_str})。"
    else:
        reason += f" 進入が枠なりでない({lane_str})。"
        if key_pushed_out:
            reason += f"キー{X}号艇が{_lane[X]}コースへ押し出された＝この賭けの前提が崩れる型(過去ROI66%)。"
        elif key_moved_in:
            reason += f"キー{X}号艇が{_lane[X]}コースへ内に入った＝キーには有利な型(過去ROI101%)。"
        if honmei_moved:
            reason += f"本命{P}号艇が{_lane[P]}コースへ動いた＝並びが読めない型(過去ROI約50%)。"
    if not odds_map:
        reason += " オッズ未取得のため買い判断は保留(購入直前に要確認)。"
    return {"decision": dec55, "dec40": dec40, "dec55": dec55, "dec_stable": dec_stable,
            "dec_kp6": dec_kp6, "dec_kp7": dec_kp7,
            "dec_m50": dec_m50, "dec_m60": dec_m60,
            "dec_kp8": dec_kp8, "dec_conf20": dec_conf20, "dec40_ex": dec40_ex,
            "dec40_calm": dec40_calm,
            "dec40_ev12": dec40_ev12, "dec40_ev13": dec40_ev13,
            "dec40_lane": dec40_lane,
            "lane": {"known": lane_known, "wakunari": lane_wakunari, "text": lane_str,
                     "key_out": key_pushed_out, "key_in": key_moved_in, "honmei_moved": honmei_moved},
            "stars": _uv_stars(gap[X], basket_ev), "honmei": P,
            "key": {"frame": X, "motor2": mot[X], "nat2": nat[X], "top3": round(top3[X], 3),
                    "gap": gap[X], "keyp": round(keyp, 3)}, "picks": picks,
            "picks_all": picks_all, "total": total,
            "basket_ev": basket_ev, "basket_ev8": basket_ev8, "reason": reason}


@app.get("/api/undervalue")
def api_undervalue(jcd: int = Query(...), rno: int = Query(...), hd: str = Query(...)):
    try:
        data = fetch_openapi(hd)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"openapi取得失敗: {e}"}, status_code=502)
    race = find_race(data, jcd, rno)
    if race is None:
        return JSONResponse({"ok": False, "error": "該当レースが見つかりません"}, status_code=404)
    odds_map = fetch_trifecta_odds(jcd, rno, hd)
    res = undervalue_pick(race, odds_map)
    res.update({"ok": True, "jcd": jcd, "rno": rno, "hd": hd,
                "odds_status": "ライブ（boatrace.jp）" if odds_map else "オッズ未取得"})
    return res


_UNDERVALUE_HTML = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>過小評価キー 3連単EV</title>
<style>
:root{--bg:#f4f6f9;--card:#fff;--line:#dbe2ea;--buy:#1c7a38;--skip:#8a97a5;--ink:#1a2330;--sub:#5c6b7a;--accent:#7a5b1c;--key:#b8860b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Hiragino Kaku Gothic ProN",sans-serif;padding:16px;max-width:860px;margin:0 auto}
h1{font-size:22px;margin:.2em 0}.sub{color:var(--sub);font-size:13px;line-height:1.6}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.bar a,.btn{display:inline-block;padding:8px 12px;border-radius:10px;background:#eef2f7;color:#22303f;text-decoration:none;font-size:13px;border:1px solid var(--line);cursor:pointer}
.btn{background:var(--key);color:#fff;border:none;font-weight:700}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin-top:14px}
label{font-size:12px;color:var(--sub)}select,input{padding:8px;border:1px solid var(--line);border-radius:8px;font-size:15px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:end}
.verdict{font-size:20px;font-weight:800}.buy{color:var(--buy)}.skip{color:var(--skip)}
.stars{color:var(--key);font-size:18px}.keybox{background:#fff8e6;border:1px solid #f0e2b8;border-radius:10px;padding:10px;margin:8px 0;font-size:13px}
table{width:100%;border-collapse:collapse;margin-top:8px}th,td{border-bottom:1px solid var(--line);padding:7px;font-size:14px;text-align:center}
th{color:var(--sub);font-weight:600;font-size:12px}.combo{font-weight:800}.tsub{color:var(--sub);font-size:12px}.big{font-size:17px;font-weight:800}
.note{background:#fff8e6;border:1px solid #f0e2b8;border-radius:10px;padding:10px;font-size:12px;color:#6b5a1f;margin-top:12px}
</style></head><body>
<h1>&#127775; 過小評価キー（3連単EV）</h1>
<div class="sub">戸田7R(5-1-3)の勝ち筋をモデル化。<b>選手成績は低いのにモーターが良い＝不人気で高オッズ</b>の艇を“軸ヒモ”にして本命と絡める。
検証済みの本物の優位は<b>+4pt(3着以内31→35.5%)と小さめ</b>。妙味(EV)はライブオッズでのみ測れる前向き検証ツール。</div>
<div class="bar"><a href="/final">締切5分前自動</a><a href="/trifecta">3連単的中率重視</a><a href="/dashboard">成績</a></div>
<div class="card"><div class="row">
  <div><label>場</label><br><select id="jcd">
  <option value="1">桐生</option><option value="2">戸田</option><option value="3">江戸川</option><option value="4">平和島</option>
  <option value="5">多摩川</option><option value="6">浜名湖</option><option value="7">蒲郡</option><option value="8">常滑</option>
  <option value="9">津</option><option value="10">三国</option><option value="11">びわこ</option><option value="12">住之江</option>
  <option value="13">尼崎</option><option value="14">鳴門</option><option value="15">丸亀</option><option value="16">児島</option>
  <option value="17">宮島</option><option value="18">徳山</option><option value="19">下関</option><option value="20">若松</option>
  <option value="21">芦屋</option><option value="22">福岡</option><option value="23">唐津</option><option value="24">大村</option></select></div>
  <div><label>R</label><br><input id="rno" type="number" min="1" max="12" value="7" style="width:70px"></div>
  <div><label>日付</label><br><input id="hd" type="date"></div>
  <div><button class="btn" onclick="go()">判定する</button></div>
</div></div>
<div id="out"></div>
<div class="note">⚠ 検証結果: 「不人気×モーター良」の3着以内優位は約+4pt(小さい)。ボート2連率は3着以内を予測しない(参考のみ)。控除率25%の壁は不変で勝ちは保証しません。妙味が出るのは市場がモーターを安く見たとき＝ライブのオッズ次第。前向きに通算EVを測る道具です。</div>
<script>
const $=id=>document.getElementById(id);
const jst=new Date(Date.now()+9*3600*1000);$("hd").value=jst.toISOString().slice(0,10);
function stars(n){return "★".repeat(n)+"☆".repeat(5-n);}
async function go(){
  $("out").innerHTML='<div class="card">判定中…</div>';
  const hd=$("hd").value.replaceAll("-","");
  const q=new URLSearchParams({jcd:$("jcd").value,rno:$("rno").value,hd:hd});
  try{
    const j=await (await fetch("/api/undervalue?"+q.toString())).json();
    if(!j.ok){$("out").innerHTML='<div class="card">取得できません：'+(j.error||"")+'</div>';return;}
    let h='<div class="card">';
    h+='<div class="verdict '+(j.decision==="買い"?"buy":"skip")+'">'+(j.decision==="買い"?"🟢 買い":"🔴 見送り")+'</div>';
    h+='<div class="stars">'+stars(j.stars)+'</div>';
    if(j.key){h+='<div class="keybox">⭐ 過小評価キー：<b>'+j.key.frame+'号艇</b>　モーター2連率 <b>'+j.key.motor2.toFixed(0)+'%</b>／全国2連率 '+j.key.nat2.toFixed(0)+'%　モデルの3着以内 <b>'+(j.key.top3*100).toFixed(0)+'%</b></div>';}
    h+='<div class="sub">本命 '+j.honmei+'号艇　'+j.reason+'</div>';
    if(j.picks&&j.picks.length){
      h+='<table><tr><th>買い目</th><th>役割</th><th>推定的中率</th><th>オッズ</th><th>EV</th><th>金額</th><th>的中時払戻</th></tr>';
      for(const p of j.picks){
        h+='<tr><td class="combo">'+p.combo+'</td><td class="tsub">'+p.pos+'</td><td>'+p.hit_pct+'%</td><td>'+(p.odds!=null?p.odds+'倍':'—')+'</td><td>'+(p.ev!=null?p.ev:'—')+'</td><td>'+p.stake+'円</td><td>'+(p.payout!=null?p.payout.toLocaleString()+'円':'—')+'</td></tr>';
      }
      h+='</table>';
      h+='<div class="sub" style="margin-top:8px">合計 <span class="big">'+j.total.toLocaleString()+'円</span>　<span class="tsub">モデル期待回収率(参考・保守的) '+(j.basket_ev*100).toFixed(0)+'%　オッズ：'+j.odds_status+'</span></div>';
    }
    h+='</div>';
    $("out").innerHTML=h;
  }catch(e){$("out").innerHTML='<div class="card">エラー：'+e+'</div>';}
}
</script></body></html>"""


@app.get("/undervalue", response_class=HTMLResponse)
def undervalue_page():
    return _UNDERVALUE_HTML
