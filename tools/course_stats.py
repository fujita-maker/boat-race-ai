"""v202: 選手のコース別成績（data/course_stats.json）を boatraceopenapi の日別結果から作る・足す。

形式: {"start": "20260101", "asof": "YYYYMMDD", "stats": {"登番": {"コース": [出走, 1着, 3着内, ST合計, ST件数]}}}
  コースは結果の進入コース。ST は 0〜0.5 秒のものだけ（F・出遅れは除く）。
使い方:
  python tools/course_stats.py update            # asof の翌日〜昨日(JST)を openapi から足す
  python tools/course_stats.py build DIR...      # 手元の日別JSON(.json / .json.gz)から作り直す
"""
import datetime as dt
import glob
import gzip
import json
import os
import sys
import urllib.request

OUT = "data/course_stats.json"
URL = "https://raw.githubusercontent.com/boatraceopenapi/api/gh-pages/docs/v1/{y}/{hd}.json"


def add_day(stats, data):
    """1日分の openapi JSON を stats に足す。足したレース数を返す。"""
    n = 0
    progs = (data or {}).get("programs") or {}
    for st in (progs.get("stadiums") or {}).values():
        for r in (st.get("races") or {}).values():
            res = (r.get("result") or {}).get("racers") or {}
            rc = r.get("racers") or {}
            if len(res) != 6:
                continue
            n += 1
            for i in range(1, 7):
                x = res.get(str(i)) or {}
                a = rc.get(str(i)) or {}
                tb = a.get("number") or x.get("number")
                co = x.get("course_number")
                pl = x.get("place_number")
                if not tb or not co:
                    continue
                s = stats.setdefault(str(tb), {}).setdefault(str(int(co)), [0, 0, 0, 0.0, 0])
                s[0] += 1
                s[1] += 1 if pl == 1 else 0
                s[2] += 1 if isinstance(pl, int) and pl <= 3 else 0
                t = x.get("start_timing")
                if isinstance(t, (int, float)) and 0 <= t < 0.5:
                    s[3] = round(s[3] + float(t), 2)
                    s[4] += 1
    return n


def _load(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
        return json.load(f)


def build(dirs):
    files = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json*")):
            hd = os.path.basename(p)[:8]
            if hd.isdigit():
                files[hd] = p
    stats = {}
    for hd in sorted(files):
        add_day(stats, _load(files[hd]))
    out = {"start": min(files), "asof": max(files), "stats": stats}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("built", out["start"], out["asof"], len(stats), "racers")


def update():
    with open(OUT, encoding="utf-8") as f:
        cur = json.load(f)
    jst = dt.timezone(dt.timedelta(hours=9))
    yday = dt.datetime.now(jst).date() - dt.timedelta(days=1)
    d = dt.datetime.strptime(cur["asof"], "%Y%m%d").date() + dt.timedelta(days=1)
    while d <= yday:
        hd = d.strftime("%Y%m%d")
        try:
            with urllib.request.urlopen(URL.format(y=hd[:4], hd=hd), timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print("stop at", hd, type(e).__name__)
            break
        n = add_day(cur["stats"], data)
        if n == 0:
            print("no results yet", hd)
            break
        cur["asof"] = hd
        print("added", hd, n)
        d += dt.timedelta(days=1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        build(sys.argv[2:])
    else:
        update()
