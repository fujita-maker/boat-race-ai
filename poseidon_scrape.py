#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""poseidon_scrape.py — ポセイドンの3連単120通り [AI予想確率・オッズ・海神指数] を収集。

静的HTML(裏API無し)なので requests のみ。オッズ/海神指数は出走15分前〜レース後に入る(前夜は---/未算出)。
  URL: https://poseidon-boatrace.net/race/{YYYYMMDD}/{jcd}/{R}R
出力: data/poseidon/{YYYYMMDD}.json = { "jcd_R": {"i-j-k":[prob, odds|null, kaijin|null], ...}, ... }
  ・前夜に実行 → probだけ入りオッズはnull(記録用)。
  ・レース後(夜)に実行 → 最終オッズ・海神指数まで入る(EV精算用)。同じ日を上書き更新。
使い方: python poseidon_scrape.py --hd 20260821
"""
import argparse, json, os, re, time
import datetime as dt
import requests

URL = "https://poseidon-boatrace.net/race/{hd}/{jcd}/{r}R"
UA = {"User-Agent": "Mozilla/5.0 (compatible; research/1.0)"}
# 組<th> の直後の3つの<td>= AI予想確率% / オッズ / 海神指数pt
ROW = re.compile(
    r'>\s*([1-6])-([1-6])-([1-6])\s*</th>\s*'
    r'<td[^>]*>\s*([^<]*?)\s*</td>\s*'
    r'<td[^>]*>\s*([^<]*?)\s*</td>\s*'
    r'<td[^>]*>\s*([^<]*?)\s*</td>',
    re.S)

def _num(s):
    s = (s or "").strip().replace(",", "").replace("pt", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None

def parse(html):
    out = {}
    for m in ROW.finditer(html):
        combo = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if combo in out:
            continue
        p = _num(m.group(4))          # AI予想確率(%)
        odds = _num(m.group(5))        # オッズ(--- なら None)
        kaijin = _num(m.group(6))      # 海神指数(未算出 なら None)
        if p is None:
            continue
        out[combo] = [p, odds, kaijin]
    return out

def fetch_race(hd, jcd, r, timeout=20, retries=3):
    url = URL.format(hd=hd, jcd=jcd, r=r)
    for _ in range(retries):
        try:
            resp = requests.get(url, headers=UA, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return parse(resp.text)
        except Exception:
            time.sleep(1.5)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default="")
    ap.add_argument("--out", default="data/poseidon")
    ap.add_argument("--sleep", type=float, default=0.4)
    a = ap.parse_args()
    hd = a.hd or dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y%m%d")
    out = {}
    for jcd in range(1, 25):
        got = 0
        for r in range(1, 13):
            probs = fetch_race(hd, jcd, r)
            time.sleep(a.sleep)
            if probs and len(probs) >= 100:
                out[f"{jcd}_{r}"] = probs
                got += 1
            elif probs is None and r == 1:
                break
        if got:
            print(f"  会場{jcd}: {got}レース", flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{hd}.json")
    json.dump(out, open(path, "w"), ensure_ascii=False)
    # オッズが入っているレース数(=レース後スクレイプの目安)
    with_odds = sum(1 for v in out.values() if any(c[1] for c in v.values()))
    print(f"[poseidon] {hd} {len(out)}レース保存(うちオッズ有 {with_odds}) -> {path}", flush=True)

if __name__ == "__main__":
    main()
