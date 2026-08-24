"""
ボートレース 妙味スコア予想 バックエンド
- 出走表・直前情報（展示/ST/コース）: boatraceopenapi の JSON（安定）
- 2連単オッズ: boatrace.jp を best-effort スクレイプ（取れなければ null → フロントは推定にフォールバック）
FastAPI / Render 対応。
"""
import os
import time
import json
import datetime as dt
from typing import Any, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse
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


@app.get("/today")
def today_page():
    d = _gh_json("today_picks.json")
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
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
    if not d or not d.get("races"):
        body = ('<div class="sub">まだ今日の予想がありません。予想ワークフロー(daily-predict)が'
                '1回動くと表示されます（JST 9・12・15・18時に自動実行／締切に近いほどオッズは最終値に近づきます）。更新: %s</div>' % now)
        return HTMLResponse(head + bar + body + "</body></html>")
    gen = d.get("generated_at", "?"); hd = d.get("hd", "")
    races = d["races"]
    # 高オッズEVが「買い」のレースだけ抽出し、締切順に
    buys = [r for r in races if r.get("ho") and r["ho"].get("decision") == "買い" and r["ho"].get("picks")]
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
           '予想生成: %s（本ページは10分ごと自動更新／daily-predictがJST 9・12・15・18時に生成）。'
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
    uv_buys = [(r, u) for r, u in uv_today if u.get("decision") == "買い"]
    uv_bset = [(r, u) for r, u in uv_buys if r.get("status") == "settled"]
    uv_hits = sum(1 for r, u in uv_bset if str(r.get("uv_hit")) == "1")
    uv_stake = sum(fnum(u.get("total")) for r, u in uv_bset)
    uv_ret = sum(fnum(r.get("uv_ret")) for r, u in uv_bset)
    uv_skips = sum(1 for r, u in uv_today if u.get("decision") == "見送り")
    uv_roi = (uv_ret / uv_stake * 100) if uv_stake else 0

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
            '<meta http-equiv="refresh" content="120">'
            '<title>締切5分前 最終判定</title><style>%s</style></head><body>'
            '<h1>&#128337; 締切5分前 自動判定（的中率重視＋過小評価キー）</h1>' % css)
    bar = ('<div class="bar"><a href="/final">&#8635; 最新に更新</a>'
           '<a class="g" href="/trifecta">手動で1レース判定</a>'
           '<a class="g" href="/today">今日の高オッズEV</a>'
           '<a class="g" href="/dashboard">成績</a></div>')

    b_roi = (b_ret / b_stake * 100) if b_stake else 0
    kpi = ('<div class="kpi" style="margin-top:12px">'
           '<div class="tsub">実運用（買い）<br><b>%d</b> レース購入</div>'
           '<div class="tsub">的中<br><b>%d</b> 本（的中率 %.0f%%）</div>'
           '<div class="tsub">投資 / 払戻<br><b>%s</b> / %s 円</div>'
           '<div class="tsub">回収率<br><b>%.1f%%</b></div>'
           '<div class="tsub">見送り<br><b>%d</b> レース</div>'
           '<div class="tsub">研究: 見送りを買っていたら<br><b>%d</b> 本的中</div>'
           '</div>') % (len(buys), b_hits, (b_hits/len(b_settled)*100 if b_settled else 0),
                        f"{int(b_stake):,}", f"{int(b_ret):,}", b_roi, len(skips), s_hits)

    uv_kpi = ('<div class="tsub" style="margin-top:14px;font-weight:700;color:#b8860b">&#127775; 過小評価キー（3連単EV）当日成績</div>'
              '<div class="kpi" style="margin-top:6px">'
              '<div class="tsub">買い<br><b>%d</b> レース</div>'
              '<div class="tsub">的中<br><b>%d</b> 本（的中率 %.0f%%）</div>'
              '<div class="tsub">投資 / 払戻<br><b>%s</b> / %s 円</div>'
              '<div class="tsub">回収率<br><b>%.1f%%</b></div>'
              '<div class="tsub">見送り<br><b>%d</b> レース</div>'
              '</div>') % (len(uv_buys), uv_hits, (uv_hits/len(uv_bset)*100 if uv_bset else 0),
                           f"{int(uv_stake):,}", f"{int(uv_ret):,}", uv_roi, uv_skips)

    def card(r):
        try:
            picks = json.loads(r.get("picks_json") or "[]")
        except Exception:
            picks = []
        d = r.get("decision"); isbuy = (d == "買い")
        badge = ('<span class="big buy">&#128994; 買い</span>' if isbuy
                 else '<span class="big skip">&#128308; 見送り</span>')
        stars = int(fnum(r.get("stars"), 0))
        star_s = "★" * stars + "☆" * (5 - stars)
        h = '<div class="fcard"><div class="fhead">'
        h += '<div><span class="ct">%s %sR</span> <span class="tsub">締切%s・判定%s</span></div>' % (
            ven(r.get("jcd")), r.get("rno"), r.get("closed_at", ""), str(r.get("ts", ""))[11:16])
        h += '<div>%s <span class="stars">%s</span></div></div>' % (badge, star_s)
        if r.get("honmei"):
            h += '<div class="tsub" style="margin:4px 0">本命 %s号艇・自信度%.1f%%　%s</div>' % (
                r.get("honmei"), fnum(r.get("conf")) * 100, r.get("reason", ""))
        else:
            h += '<div class="tsub" style="margin:4px 0">%s</div>' % r.get("reason", "")
        # 買い目テーブル
        if picks:
            if isbuy:
                h += '<table><tr><th>買い目</th><th>推定的中率</th><th>オッズ</th><th>金額</th><th>的中時払戻</th></tr>'
                for p in picks:
                    o = p.get("odds"); pay = p.get("payout")
                    h += '<tr><td class="combo">%s</td><td>%s%%</td><td>%s</td><td>%s円</td><td>%s</td></tr>' % (
                        p.get("combo"), p.get("hit_pct", ""),
                        (str(o) + "倍" if o is not None else "—"), p.get("stake", ""),
                        (f"{int(pay):,}円" if pay is not None else "—"))
                h += '</table>'
            else:
                h += '<div class="tsub">参考予想： ' + " / ".join(
                    '<span class="combo">%s</span>(%s%%)' % (p.get("combo"), p.get("hit_pct", "")) for p in picks) + '</div>'
        # 過小評価キー（毎レース 買い/見送り を表示）
        try:
            uv = json.loads(r.get("uv_json") or "{}")
        except Exception:
            uv = {}
        if uv:
            uvbuy = uv.get("decision") == "買い"
            badge = ('<b style="color:#1c7a38">🟢 過小評価キー 買い</b>' if uvbuy
                     else '<b style="color:#8a97a5">🔴 過小評価キー 見送り</b>')
            inner = ""
            k = uv.get("key")
            if k:
                inner += '　<span class="tsub">キー<b>%s号艇</b>(モ%s%%/全国2連率%s%%)</span>' % (
                    k.get("frame"), round(fnum(k.get("motor2"))), round(fnum(k.get("nat2"))))
            upicks = uv.get("picks", [])
            win = r.get("win_combo")
            settled = (r.get("status") == "settled" and win)
            if upicks:
                # 買い/見送り どちらも買い目(参考予想)を表示
                inner += '　' + " / ".join('<span class="combo">%s</span>(%s倍)' % (
                    p.get("combo"), p.get("odds") if p.get("odds") is not None else "—") for p in upicks)
                if settled and uvbuy:
                    inner += ('　<span class="res-hit">的中%s円</span>' % f"{int(fnum(r.get('uv_ret'))):,}"
                              if str(r.get("uv_hit")) == "1" else '　<span class="res-miss">不的中</span>')
                elif settled and not uvbuy:
                    hit = any(p.get("combo") == win for p in upicks)
                    inner += ('　<span class="tsub">研究:買っていたら%s</span>'
                              % ('<span class="res-hit">的中</span>' if hit else '不的中'))
            elif not uvbuy:
                inner += '　<span class="tsub">%s</span>' % (uv.get("reason", "")[:60])
            h += ('<div style="background:#fff8e6;border:1px solid #f0e2b8;border-radius:10px;padding:8px 10px;margin-top:8px;font-size:13px">'
                  '%s%s</div>') % (badge, inner)
        # 結果
        if r.get("status") == "settled" and r.get("win_combo"):
            if isbuy:
                res = ('<span class="res-hit">的中 %s（払戻%s円）</span>' % (r.get("win_combo"), f"{int(fnum(r.get('payout'))):,}")
                       if str(r.get("hit")) == "1" else '<span class="res-miss">不的中（結果 %s）</span>' % r.get("win_combo"))
            else:
                res = ('<span class="tsub">結果 %s ／ 研究:買っていたら%s</span>' % (
                    r.get("win_combo"), "的中" if str(r.get("shadow_hit")) == "1" else "不的中"))
            h += '<div style="margin-top:6px">%s</div>' % res
        h += '</div>'
        return h

    if not today:
        body = ('<div class="fcard tsub">本日の最終判定はまだありません。'
                '開催時間帯に、各レースの締切約5〜10分前へ来ると自動で判定が追加されます'
                '（Render常駐スケジューラが数分おきに自動実行。ボタン操作は不要）。</div>')
    else:
        order = sorted(today, key=lambda r: str(r.get("closed_at") or "z"))
        body = "".join(card(r) for r in order)
    note = ('<div class="small" style="margin-top:14px">締切5分前(実際は約5〜10分前)に、その時点の最新データ＋3連単オッズで'
            '120通りを再計算し、買い/見送りを確定してそのまま保存します（結果を見てから予測は変えません）。'
            '「実運用」は買いと判定したレースのみ。「研究」は見送りを仮に買っていた場合で、実運用成績には含めません。'
            '控除率25%の壁は不変で、これは勝ちを保証しない検証ツールです。</div>')
    return HTMLResponse(head + bar + kpi + uv_kpi + body + note + "</body></html>")


# ================= 常時稼働スケジューラ（Render Starter=常時ON）=================
# 締切約5〜10分前に自動で3連単を最終判定→固定保存し、確定レースを自動精算する。
# GitHubの不安定な定時実行に頼らず、常時稼働のRender内で回す。台帳は永続ディスクへ保存。
import threading

_TRI_DIR = os.environ.get("TRI_DATA_DIR") or ("/var/data" if os.path.isdir("/var/data") else "/tmp")
_TRI_LEDGER = os.path.join(_TRI_DIR, "tri_hit_ledger.csv")
_TRI_COLS = ["ts", "date", "jcd", "rno", "closed_at", "decision", "stars", "conf", "honmei",
             "picks_json", "total", "reason", "status",
             "win_combo", "hit", "payout", "ret", "shadow_hit", "shadow_ret",
             "uv_json", "uv_hit", "uv_ret"]
_TRI_WIN_LO, _TRI_WIN_HI = 3, 13
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


def _tri_top_combos(boats, n=3):
    power = predict_power(boats)
    ranked = sorted(trifecta_probs_pl(power).items(), key=lambda kv: kv[1], reverse=True)
    return [{"combo": f"{c[0]}-{c[1]}-{c[2]}", "P": round(p, 4), "hit_pct": round(p * 100, 1)}
            for c, p in ranked[:n]]


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
                res = trifecta_pick(boats, odds_map)
                picks = res["picks"] if res["decision"] == "買い" else _tri_top_combos(boats, 3)
                # 過小評価キー判定も同時に実施（同じオッズを再利用）
                try:
                    uv = undervalue_pick(rc, odds_map)
                    uv_json = json.dumps({"decision": uv["decision"], "stars": uv["stars"],
                                          "honmei": uv["honmei"], "key": uv["key"], "picks": uv["picks"],
                                          "total": uv["total"], "basket_ev": uv["basket_ev"],
                                          "reason": uv["reason"]}, ensure_ascii=False)
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
            # 過小評価キー(買い時)の精算
            try:
                uv = json.loads(r.get("uv_json") or "{}")
            except Exception:
                uv = {}
            if uv.get("decision") == "買い":
                uret = 0.0; uhit = 0
                for p in uv.get("picks", []):
                    if p.get("combo") == win_combo:
                        uhit = 1; uret += (p.get("stake", 0) or 0) * (pay / 100.0)
                r["uv_hit"] = uhit; r["uv_ret"] = int(uret)
            r["status"] = "settled"
        _tri_save_ledger(rows)
        return made


def _tri_scheduler_loop():
    while True:
        try:
            if 8 <= _tri_now().hour <= 23:
                tri_judge_cycle()
        except Exception as e:
            print("[tri-sched]", e, flush=True)
        time.sleep(150)


_tri_started = False


@app.on_event("startup")
def _tri_start():
    global _tri_started
    if _tri_started:
        return
    _tri_started = True
    threading.Thread(target=_tri_scheduler_loop, daemon=True).start()
    print("[tri-sched] 常駐スケジューラ起動 ledger=" + _TRI_LEDGER, flush=True)


@app.get("/api/tri/run")
def api_tri_run():
    """手動でも今すぐ1周判定できる確認用エンドポイント。"""
    n = tri_judge_cycle()
    return {"ok": True, "made": n, "ledger": _TRI_LEDGER}


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
        return {"decision": "見送り", "stars": 1, "honmei": honmei, "key": None, "picks": [],
                "total": 0, "basket_ev": 0,
                "reason": f"過小評価キー(機力良×不人気)が見当たらないため見送り。本命{honmei}号艇。"}
    X = max(cand, key=lambda f: (gap[f], mot[f]))    # ギャップ最大→モーター最良
    # 本命勢P,Q,R = モデルP(1着)上位(Xを除く)
    order = [f for f in sorted(frames, key=lambda f: power.get(f, 0), reverse=True) if f != X]
    P, Q, R = order[0], order[1], order[2]
    # 買い目: Xを3着中心・2着少し・1着1点で絡める(戸田7Rの型)
    combos = [((P, Q, X), 200, "3着"), ((Q, P, X), 200, "3着"), ((P, R, X), 200, "3着"),
              ((P, X, Q), 150, "2着"), ((Q, X, P), 150, "2着"), ((P, X, R), 100, "2着(裏)"),
              ((X, P, Q), 100, "1着(上振れ)"), ((X, Q, P), 100, "1着(裏)")]
    tri = trifecta_probs_pl(power)
    picks = []
    exp_ret = 0.0; total = 0
    for (ijk, stake, pos) in combos:
        cs = f"{ijk[0]}-{ijk[1]}-{ijk[2]}"
        p = tri.get(ijk, 0.0)
        o = odds_map.get(cs) if odds_map else None
        ev = round(p * o, 2) if o else None
        payout = int(round(stake * o)) if o else None
        picks.append({"combo": cs, "pos": pos, "P": round(p, 4), "hit_pct": round(p * 100, 1),
                      "odds": o, "ev": ev, "stake": stake, "payout": payout})
        total += stake
        if o:
            exp_ret += stake * p * o
    basket_ev = round(exp_ret / total, 3) if total else 0
    max_odds = max([p["odds"] for p in picks if p["odds"]], default=0)
    # 妙味の買い条件: 明確な過小評価キー(gap>=3 かつ モーター>=40) かつ 高オッズ(市場が安く見ている=最大オッズ>=15)
    strong = (gap[X] >= 3 and mot[X] >= 40)
    decision = "買い" if (odds_map and strong and max_odds >= 15) else "見送り"
    reason = (f"{X}号艇＝モーター2連率{mot[X]:.0f}%(機力{mot_rank[X]}位)なのに全国2連率{nat[X]:.0f}%(人気{nat_rank[X]}位)＝"
              f"実力の割に不人気で高オッズ。本命{P}号艇を軸に、{X}を3着中心で絡めて{len(picks)}点。"
              f"モデルは{X}の3着以内を{top3[X]*100:.0f}%と評価(選手成績主導で低めに出るが、検証では不人気×モーター良は実測+4pt)。")
    if not odds_map:
        reason += " オッズ未取得のため買い判断は保留(購入直前に要確認)。"
    elif decision == "見送り":
        if not strong:
            reason += " ただし過小評価の度合いが基準未満(gap<3 か モーター<40)＝妙味不足で見送り。"
        else:
            reason += f" キーは十分だが最大オッズ{max_odds}倍が低く、既に市場が織り込み済み＝妙味不足で見送り。"
    return {"decision": decision, "stars": _uv_stars(gap[X], basket_ev), "honmei": P,
            "key": {"frame": X, "motor2": mot[X], "nat2": nat[X], "top3": round(top3[X], 3),
                    "gap": gap[X]}, "picks": picks, "total": total, "basket_ev": basket_ev,
            "reason": reason}


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
