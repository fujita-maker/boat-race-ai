"""
ボートレース 妙味スコア予想 バックエンド
- 出走表・直前情報（展示/ST/コース）: boatraceopenapi の JSON（安定）
- 2連単オッズ: boatrace.jp を best-effort スクレイプ（取れなければ null → フロントは推定にフォールバック）
FastAPI / Render 対応。
"""
import os
import time
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
         "stake", "status", "payout", "ret", "hit"]


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


def highodds_pick(boats, odds_map, odds_min, ev_min, p_floor, min_conf, min_gap, max_n):
    power = predict_power(boats)
    conf, gap = _conf_gap(power)
    if conf < min_conf or gap < min_gap:
        return {"decision": "見送り", "reason": f"レースゲート未通過（自信度{conf*100:.0f}%/抜け{gap*100:.0f}%）",
                "conf": round(conf, 4), "gap": round(gap, 4), "picks": []}
    ex = _exacta_probs(power); cand = []
    for (i, j), p in ex.items():
        o = odds_map.get(f"{i}-{j}") if odds_map else None
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


def _append_ledger(hd, jcd, rno, picks, stake):
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
                        "stake": stake, "status": "pending", "payout": "", "ret": "", "hit": ""})


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
                 stake: float = Query(100.0), log: bool = Query(False)):
    try:
        data = fetch_openapi(hd)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"openapi取得失敗: {e}"}, status_code=502)
    race = find_race(data, jcd, rno)
    if race is None:
        return JSONResponse({"ok": False, "error": "該当レースが見つかりません"}, status_code=404)
    boats = extract_boats(race)
    odds_map = fetch_exacta_odds(jcd, rno, hd)
    res = highodds_pick(boats, odds_map, odds_min, ev_min, p_floor, min_conf, min_gap, max_n)
    res.update({"ok": True, "jcd": jcd, "rno": rno, "hd": hd,
                "odds_status": "ライブ（boatrace.jp）" if odds_map else "オッズ未取得（見送り扱い推奨）"})
    if log and res["decision"] == "買い" and odds_map:
        _append_ledger(hd, jcd, rno, res["picks"], stake)
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
    <div><button onclick="judge(false)">&#9654; 判定する</button></div>
  </div>
  <details style="margin-top:10px"><summary>詳細設定（しきい値）</summary>
    <div class="row" style="margin-top:8px">
      <div><label>オッズ下限(倍)</label><input type="number" id="odds_min" value="20" style="width:80px"></div>
      <div><label>EV下限</label><input type="number" id="ev_min" value="1.20" step="0.05" style="width:80px"></div>
      <div><label>自信度下限</label><input type="number" id="min_conf" value="0.12" step="0.01" style="width:80px"></div>
      <div><label>本命抜け下限</label><input type="number" id="min_gap" value="0.30" step="0.05" style="width:80px"></div>
      <div><label>最大点数</label><input type="number" id="max_n" value="2" style="width:70px"></div>
      <div><label>1点賭け金(円)</label><input type="number" id="stake" value="100" style="width:90px"></div>
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
    min_gap:$("min_gap").value,max_n:$("max_n").value,stake:$("stake").value,log:log?"true":"false"});
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
    const rows=(j.rows||[]).slice().reverse().slice(0,30);
    if(rows.length){
      let h='<table><tr><th>日付</th><th>場R</th><th>買い目</th><th>オッズ</th><th>EV</th><th>状態</th><th>払戻</th></tr>';
      for(const r of rows) h+=`<tr><td>${r.date}</td><td>${r.jcd}-${r.rno}</td><td>${r.combo}</td><td>${r.odds}</td><td>${r.EV}</td><td>${r.status=="settled"?(r.hit=="1"?"的中":"外れ"):"未"}</td><td>${r.ret||""}</td></tr>`;
      $("ledger").innerHTML=h+'</table>';
    }else $("ledger").innerHTML='<div class="detail">まだ記録がありません。判定→記録で貯まります。</div>';
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
    return HTMLResponse(INDEX_HTML)
