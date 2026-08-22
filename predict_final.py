#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict_final.py — 締切5分前(既定: 締切まで3〜13分)の最終判定を全24場で自動実行。

- そのウィンドウに入ったレースだけ、最新の出走表/直前/3連単オッズを取り直して 3連単120通りを再計算。
- 「🎯 3連単 的中率重視」ロジック(app.trifecta_pick: 自信度上位約5%ゲート・5点)で 買い/見送り を確定。
- data/tri_hit_ledger.csv に固定保存(上書き禁止=一度出した予測は結果で書き換えない)。
- 確定済み(結果が出た)レースは自動精算。見送りは「買っていたら当たったか」を shadow 列に別記録(研究用)。
本番=GitHub Actions(締切前ウィンドウ)。app.py の共通ロジックを再利用し、本番/検証でロジックを一致させる。
"""
import sys, os, csv, json, argparse
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as A   # 共通: fetch_openapi/find_race/extract_boats/trifecta_pick/fetch_trifecta_odds/_payouts/predict_power/trifecta_probs_pl

JST = dt.timezone(dt.timedelta(hours=9))
LEDGER = "data/tri_hit_ledger.csv"
COLS = ["ts", "date", "jcd", "rno", "closed_at", "decision", "stars", "conf", "honmei",
        "picks_json", "total", "reason", "status",
        "win_combo", "hit", "payout", "ret", "shadow_hit", "shadow_ret"]
WIN_LO, WIN_HI = 3, 13   # 締切まで[3,13]分 → このウィンドウで最終判定(=約5〜10分前を狙う)


def now_jst():
    return dt.datetime.now(JST)


def parse_closed(s, day0):
    """race_closed_at文字列 → JST datetime。'HH:MM' or ISO 両対応。取れなければ None。"""
    if not s:
        return None
    s = str(s)
    try:
        if "T" in s or ":" in s and len(s) > 6:
            t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=JST)
            return t.astimezone(JST)
    except Exception:
        pass
    try:
        hh, mm = s.strip()[:5].split(":")
        return dt.datetime.combine(day0, dt.time(int(hh), int(mm)), tzinfo=JST)
    except Exception:
        return None


def race_closed_at(race):
    for k in ("race_closed_at", "closed_at", "race_close_time"):
        if isinstance(race, dict) and race.get(k):
            return str(race.get(k))
    return None


def load_ledger():
    rows = []
    if os.path.exists(LEDGER):
        with open(LEDGER, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    seen = {(r["date"], r["jcd"], r["rno"]): r for r in rows}
    return rows, seen


def save_ledger(rows):
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    with open(LEDGER, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


def top_combos(boats, n=3):
    power = A.predict_power(boats)
    ranked = sorted(A.trifecta_probs_pl(power).items(), key=lambda kv: kv[1], reverse=True)
    return [{"combo": f"{c[0]}-{c[1]}-{c[2]}", "P": round(p, 4), "hit_pct": round(p * 100, 1)}
            for c, p in ranked[:n]]


def predict_window(data, hd, day0, seen, rows):
    now = now_jst()
    made = 0
    for jcd in range(1, 25):
        for rno in range(1, 13):
            key = (hd, str(jcd), str(rno))
            if key in seen:
                continue                      # 既に最終判定済み=固定(再計算しない)
            rc = A.find_race(data, jcd, rno)
            if rc is None:
                continue
            _, tri = A._payouts(rc)
            if tri is not None:
                continue                      # 既に結果確定=ウィンドウ逃した/対象外
            closed = parse_closed(race_closed_at(rc), day0)
            if closed is None:
                continue
            mins = (closed - now).total_seconds() / 60.0
            if not (WIN_LO <= mins <= WIN_HI):
                continue                      # 5分前ウィンドウ外
            boats = A.extract_boats(rc)
            # 直前データ不足の安全処理: 公式の基礎データ(勝率等)が無ければ見送り
            if sum(1 for b in boats if b.get("nat") is not None) < 4:
                row = {"ts": now.strftime("%Y-%m-%d %H:%M:%S"), "date": hd, "jcd": jcd, "rno": rno,
                       "closed_at": closed.strftime("%H:%M"), "decision": "見送り", "stars": 1,
                       "conf": 0, "honmei": "", "picks_json": "[]", "total": 0,
                       "reason": "直前データ不足のため見送り(公式基礎データ欠損)。", "status": "pending",
                       "win_combo": "", "hit": "", "payout": "", "ret": "", "shadow_hit": "", "shadow_ret": ""}
                rows.append(row); seen[key] = row; made += 1
                continue
            odds_map = A.fetch_trifecta_odds(jcd, rno, hd)
            res = A.trifecta_pick(boats, odds_map)
            if res["decision"] == "買い":
                picks = res["picks"]
            else:
                picks = top_combos(boats, 3)      # 見送りでも参考予想(上位3点)を保存
            row = {"ts": now.strftime("%Y-%m-%d %H:%M:%S"), "date": hd, "jcd": jcd, "rno": rno,
                   "closed_at": closed.strftime("%H:%M"), "decision": res["decision"],
                   "stars": res["stars"], "conf": res["conf"], "honmei": res["honmei"],
                   "picks_json": json.dumps(picks, ensure_ascii=False),
                   "total": res.get("total", 0), "reason": res["reason"], "status": "pending",
                   "win_combo": "", "hit": "", "payout": "", "ret": "", "shadow_hit": "", "shadow_ret": ""}
            rows.append(row); seen[key] = row; made += 1
    return made


def settle(data, hd, rows):
    n = 0
    for r in rows:
        if r.get("status") == "settled" or r.get("date") != hd:
            continue
        rc = A.find_race(data, int(r["jcd"]), int(r["rno"]))
        if rc is None:
            continue
        _, tri = A._payouts(rc)
        if tri is None:
            continue                        # まだ結果無し
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
                    hit = 1
                    ret += (p.get("stake", 0) or 0) * (pay / 100.0)
            r["hit"] = hit; r["payout"] = int(pay) if hit else 0; r["ret"] = int(ret)
        else:
            # 見送り: 買っていたら当たったか(研究用shadow)。上位3点を各100円と仮定。
            sret = 0.0; shit = 0
            for p in picks:
                if p.get("combo") == win_combo:
                    shit = 1; sret += 100 * (pay / 100.0)
            r["shadow_hit"] = shit; r["shadow_ret"] = int(sret)
        r["status"] = "settled"; n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default="")
    a = ap.parse_args()
    hd = a.hd or now_jst().strftime("%Y%m%d")
    day0 = dt.datetime.strptime(hd, "%Y%m%d").date()
    data = A.fetch_openapi(hd)
    if data is None:
        print("[final] 本日データ無し"); return
    rows, seen = load_ledger()
    made = predict_window(data, hd, day0, seen, rows)
    settled = settle(data, hd, rows)
    save_ledger(rows)
    buy = sum(1 for r in rows if r["date"] == hd and r["decision"] == "買い")
    skip = sum(1 for r in rows if r["date"] == hd and r["decision"] == "見送り")
    print(f"[final] hd={hd} 新規判定{made}件(買い含む) 精算{settled}件 / 本日累計 買い{buy}・見送り{skip}", flush=True)


if __name__ == "__main__":
    main()
