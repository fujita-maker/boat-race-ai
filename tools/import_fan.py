"""v202: 公式「ファン手帳」期別成績(fanYYMM.lzh)からコース別成績を取り出して data/fan_course.json に保存する。

レイアウト: https://boatrace.jp/owpc/pc/extra/data/layout.html（固定長・Shift_JIS・1行1選手）
保存形式は data/course_stats.json と同じ [出走, 1着, 3着内, ST合計, ST件数]（ST合計＝平均ST×進入回数）。
使い方: python tools/import_fan.py fan2510   （lzh を落として lha で展開 → 解析 → 検査に通れば保存）
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

URL = "https://boatrace.jp/static_extra/pc_static/download/data/kibetsu/{name}.lzh"
OUT = "data/fan_course.json"
# (項目, バイト数)
T1 = [("toban", 4), ("name", 16), ("kana", 15), ("branch", 4), ("cls", 2), ("era", 1), ("birth", 6), ("sex", 1),
      ("age", 2), ("height", 3), ("weight", 2), ("blood", 2), ("win_rate", 4), ("top2_rate", 4), ("n1", 3), ("n2", 3),
      ("starts", 3), ("finals", 2), ("wins", 2), ("avg_st", 3)]
T2 = []
for c in range(1, 7):
    T2 += [("c%d_in" % c, 3), ("c%d_rate" % c, 4), ("c%d_st" % c, 3), ("c%d_strank" % c, 3)]
T3 = [("cls1", 2), ("cls2", 2), ("cls3", 2), ("idx_prev", 4), ("idx_now", 4), ("year", 4), ("term", 1),
      ("from", 8), ("to", 8), ("school", 3)]
T4 = []
for c in range(1, 7):
    T4 += [("c%d_p%d" % (c, k), 3) for k in range(1, 7)]
    T4 += [("c%d_%s" % (c, k), 2) for k in ("F", "L0", "L1", "K0", "K1", "S0", "S1", "S2")]
T5 = [("n_L0", 2), ("n_L1", 2), ("n_K0", 2), ("n_K1", 2), ("home", 6)]
LAYOUT = T1 + T2 + T3 + T4 + T5
RECLEN = sum(n for _, n in LAYOUT)   # 416


def parse_line(b):
    out = {}; pos = 0
    for k, n in LAYOUT:
        out[k] = b[pos:pos + n].decode("cp932", errors="replace").strip()
        pos += n
    return out


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def convert(recs):
    stats = {}
    for r in recs:
        tb = r["toban"]
        cs = {}
        for c in range(1, 7):
            n = _i(r["c%d_in" % c])
            if n <= 0:
                continue
            p = [_i(r["c%d_p%d" % (c, k)]) for k in range(1, 7)]
            st = _i(r["c%d_st" % c]) / 100.0
            cs[str(c)] = [n, p[0], p[0] + p[1] + p[2], round(st * n, 2), n]
        if cs:
            stats[tb] = cs
    return stats


def check(recs, stats):
    """検査: 1,000人以上・登番は4桁の数字・コース別1着の合計が全体の1着回数を超えない・1コース1着率が40〜70%。"""
    if len(recs) < 1000:
        return "選手が%d人しかいない" % len(recs)
    bad = sum(1 for r in recs if not (r["toban"].isdigit() and len(r["toban"]) == 4))
    if bad > 10:
        return "登番が数字でない行が%d行" % bad
    over = sum(1 for r in recs if sum(_i(r["c%d_p1" % c]) for c in range(1, 7)) > _i(r["n1"]) + 1)
    if over > len(recs) * 0.05:
        return "コース別1着の合計が全体の1着回数を超える選手が%d人（レイアウトずれの疑い）" % over
    n1 = sum(v["1"][0] for v in stats.values() if "1" in v); w1 = sum(v["1"][1] for v in stats.values() if "1" in v)
    if not n1 or not (0.40 <= w1 / n1 <= 0.70):
        return "1コースの1着率が%.1f%%（40〜70%%の外）" % (100 * w1 / max(1, n1))
    return None


def main(name):
    tmp = tempfile.mkdtemp()
    lzh = os.path.join(tmp, name + ".lzh")
    urllib.request.urlretrieve(URL.format(name=name), lzh)
    subprocess.run(["lha", "x", lzh], check=True, cwd=tmp)
    txts = [p for p in glob.glob(os.path.join(tmp, "*")) if not p.endswith(".lzh")]
    if not txts:
        sys.exit("展開したファイルが無い")
    raw = open(txts[0], "rb").read()
    lines = [l for l in raw.replace(b"\r\n", b"\n").split(b"\n") if len(l) >= RECLEN - 6]
    recs = [parse_line(l) for l in lines]
    stats = convert(recs)
    err = check(recs, stats)
    if err:
        sys.exit("検査に通らないので保存しない: " + err)
    r0 = recs[0]
    out = {"file": name, "from": r0["from"], "to": r0["to"], "racers": len(stats), "stats": stats}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("saved", name, r0["from"], r0["to"], len(stats), "racers", "reclen", RECLEN, "line0", len(lines[0]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fan2510")
