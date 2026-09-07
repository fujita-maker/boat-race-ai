#!/usr/bin/env python3
"""選手別「前づけ率(内寄せ率)」マスタを作る。

出力: data/maezuke_master.json
  {"global_inside_rate":..., "racers":{"登録番号":[内寄せ率, 出走数], ...},
   "stadium_break_rate":{"場番号": 枠なりにならない率}}

内寄せ率 = P(本番の進入コース < 枠番)  ※枠2〜6の出走のみ対象
  1号艇は内に行けないので分母から除く。K=40 の縮小推定(全体平均に寄せる)で少走数の暴れを抑える。

データ: boatraceopenapi (gh-pages raw)。result.racers.course_number = 本番の進入コース。
使い方: python3 tools/build_maezuke_master.py --start 2026-01-01 --end 2026-12-31
"""
import argparse, datetime as dt, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import requests

RAW = "https://raw.githubusercontent.com/boatraceopenapi/api/gh-pages/docs/v1/{y}/{hd}.json"
K = 40.0
MIN_RACES = 20


def fetch(hd):
    try:
        r = requests.get(RAW.format(y=hd[:4], hd=hd), timeout=90)
        return json.loads(r.content) if r.status_code == 200 else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--months", type=int, default=12, help="--start 未指定時、今日から遡る月数")
    ap.add_argument("--out", default="data/maezuke_master.json")
    a = ap.parse_args()
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today()
    start = dt.date.fromisoformat(a.start) if a.start else (end - dt.timedelta(days=31 * a.months))
    days = []
    d = start
    while d <= end:
        days.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)

    inside = {}   # racer -> [内寄せ回数, 出走数(枠2-6)]
    st_tot = {}   # 場 -> [枠なりでないレース数, レース数]
    tot_in = tot_n = 0
    with ThreadPoolExecutor(8) as ex:
        for data in ex.map(fetch, days):
            if not data:
                continue
            for jcd, sd in (data.get("programs", {}).get("stadiums", {}) or {}).items():
                for _rno, rc in (sd.get("races", {}) or {}).items():
                    rr = ((rc.get("result") or {}).get("racers") or {})
                    course = {}
                    for f in range(1, 7):
                        v = (rr.get(str(f)) or {}).get("course_number")
                        try:
                            course[f] = int(v) if v else None
                        except (TypeError, ValueError):
                            course[f] = None
                    if any(course[f] is None for f in range(1, 7)):
                        continue
                    s = st_tot.setdefault(str(int(jcd)), [0, 0])
                    s[1] += 1
                    if any(course[f] != f for f in range(1, 7)):
                        s[0] += 1
                    for f in range(2, 7):
                        num = ((rc.get("racers") or {}).get(str(f)) or {}).get("number")
                        if not num:
                            continue
                        rec = inside.setdefault(str(int(num)), [0, 0])
                        rec[1] += 1
                        tot_n += 1
                        if course[f] < f:
                            rec[0] += 1
                            tot_in += 1
    if not tot_n:
        print("no data", file=sys.stderr)
        return 1
    g = tot_in / tot_n
    racers = {}
    for r, (i, n) in inside.items():
        if n < MIN_RACES:
            continue
        racers[r] = [round((i + K * g) / (n + K), 3), n]
    out = {"built": dt.date.today().isoformat(),
           "window": f"{start}..{end}",
           "global_inside_rate": round(g, 4),
           "shrink_k": K, "min_races": MIN_RACES,
           "racers": racers,
           "stadium_break_rate": {k: round(v[0] / v[1], 3) for k, v in st_tot.items() if v[1]},
           "note": "racers: 登録番号 -> [内寄せ率(枠番より内のコースに入る率・縮小推定), 出走数(枠2-6)]"}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {a.out}: racers={len(racers)} global={g:.4f} window={start}..{end}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
