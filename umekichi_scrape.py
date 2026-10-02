#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""umekichi_scrape.py — 梅吉AI(umepyon.com)の全券種AI予想確率を収集。

梅吉は前夜23:30〜26:00に当日分を掲載し、翌日になると「公開終了」(404)で消える。
過去にさかのぼれないので、当日のうちに保存する。静的HTML(裏API無し)なので requests + bs4。
  一覧: https://umepyon.com/              (当日開催場。onclick に jcd=NN が入る)
  各R : https://umepyon.com/predict.php?jcd=NN&racedate=YYYY-MM-DD&racenum=R
        #tab1 単勝 / #tab2 2連単 / #tab3 3連単(120) / #tab4 複勝 / #tab5 2連複 / #tab6 3連複 / #tab7 拡連複
出力: data/umekichi/{YYYYMMDD}.json
  { "jcd_R": {"conf": 自信度(星の数)|null, "fetched": "ISO",
              "win": {"1": [確率%, オッズ|null]}, "ex": {"1-3": [...]}, "tri": {"1-3-2": [...]},
              "place": {...}, "qu": {"1-3": [...]}, "trio": {"1-2-3": [...]}, "wide": {"1-2": [...]}} }
  ・確率は最初に取った値を正本として残す(--merge 既定)。2回目以降は欠けているレースだけ取りに行く。
使い方: python umekichi_scrape.py --out data/umekichi            (当日JST)
        python umekichi_scrape.py --hd 20261002 --out data/umekichi
"""
import argparse, json, os, re, time
import datetime as dt
import requests
from bs4 import BeautifulSoup

BASE = "https://umepyon.com"
UA = {"User-Agent": "Mozilla/5.0 (compatible; research/1.0)"}
TABS = {"tab1": "win", "tab2": "ex", "tab3": "tri", "tab4": "place",
        "tab5": "qu", "tab6": "trio", "tab7": "wide"}
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def jst_today():
    return (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y%m%d")


def get(url, timeout=25, retries=3):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(2 + 2 * i)
    raise last


def venues_today():
    """トップページの onclick から当日開催場の jcd を拾う。"""
    r = get(BASE + "/")
    r.encoding = r.apparent_encoding or "utf-8"
    return sorted(set(re.findall(r"jcd=(\d{2})", r.text)))


def _num(s):
    s = (s or "").replace(",", "")
    m = NUM.search(s)
    return float(m.group()) if m else None


def parse(html):
    soup = BeautifulSoup(html, "lxml")
    out = {}
    for tab_id, key in TABS.items():
        pane = soup.find(id=tab_id)
        if not pane:
            continue
        tbl = pane.find("table")
        if not tbl:
            continue
        d = {}
        for tr in tbl.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            digits = re.findall(r"[1-6]", tds[0].get_text(" ", strip=True))
            if not digits:
                continue
            combo = "-".join(digits)
            p = _num(tds[1].get_text())
            if p is None:
                continue
            odds = _num(tds[2].get_text()) if len(tds) >= 3 else None
            d.setdefault(combo, [p, odds])
        if d:
            out[key] = d
    m = re.search(r"自信度\s*([⭐★]+)", soup.get_text(" ", strip=True))
    out["conf"] = len(m.group(1)) if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default="", help="YYYYMMDD(空なら当日JST)")
    ap.add_argument("--out", default="data/umekichi")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--refetch", action="store_true", help="取得済みレースも取り直す(確率は残しオッズだけ更新)")
    a = ap.parse_args()
    hd = a.hd or jst_today()
    rd = f"{hd[:4]}-{hd[4:6]}-{hd[6:]}"
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{hd}.json")
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

    jcds = venues_today() if hd == jst_today() else [f"{i:02d}" for i in range(1, 25)]
    print(f"{hd}: 開催場 {jcds}")
    got = skip = miss = 0
    for jcd in jcds:
        for r in range(1, 13):
            key = f"{jcd}_{r}"
            if key in data and not a.refetch:
                skip += 1
                continue
            url = f"{BASE}/predict.php?jcd={jcd}&racedate={rd}&racenum={r}"
            try:
                resp = get(url)
            except requests.RequestException as e:
                print("  ERR", key, e)
                continue
            if resp.status_code != 200:
                miss += 1
                if r == 1:
                    break       # その場は非開催/公開終了
                continue
            resp.encoding = resp.apparent_encoding or "utf-8"
            rec = parse(resp.text)
            if "tri" not in rec:
                miss += 1
                continue
            rec["fetched"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
            if key in data:     # 確率は最初の値を正本として残し、オッズだけ更新
                old = data[key]
                for k in TABS.values():
                    for c, v in rec.get(k, {}).items():
                        if c in old.get(k, {}) and v[1] is not None:
                            old[k][c][1] = v[1]
                old["refetched"] = rec["fetched"]
            else:
                data[key] = rec
            got += 1
            time.sleep(a.sleep)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    # 健全性: 3連単の確率合計が100前後か
    bad = [k for k, v in data.items() if abs(sum(x[0] for x in v.get("tri", {}).values()) - 100) > 3]
    print(f"取得 {got} / 既存スキップ {skip} / 未掲載 {miss} / 合計 {len(data)}R / 3連単合計ずれ {bad[:5]}")


if __name__ == "__main__":
    main()
