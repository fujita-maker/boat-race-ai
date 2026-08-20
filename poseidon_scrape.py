#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""poseidon_scrape.py — 競艇予想AIポセイドンの『3連単120通りAI予想確率』を収集。

ページは静的HTML(データ直書き, 裏API無し)なので requests だけで取得可。
  URL: https://poseidon-boatrace.net/race/{YYYYMMDD}/{jcd}/{R}R   (jcd=会場番号1-24, R=1-12)
各レースの 120組の "i-j-k  p%%" を正規表現で抽出。オッズ/海神指数は出走15分前に入る(前夜は---)。

出力: data/poseidon/{YYYYMMDD}.json  = { "jcd_R": {"combo":prob, ...}, ... }
使い方: python poseidon_scrape.py --hd 20260821
"""
import argparse, json, os, re, time
import datetime as dt
import requests

URL = "https://poseidon-boatrace.net/race/{hd}/{jcd}/{r}R"
UA = {"User-Agent": "Mozilla/5.0 (compatible; research/1.0)"}
# 「4-6-3 3.33%」「4-6-3</...>3.33%」等、間のタグを許容して組と確率を拾う
ROW = re.compile(r'([1-6])\s*-\s*([1-6])\s*-\s*([1-6])\D{0,80}?([0-9]+\.[0-9]+)\s*%')
# オッズ(数値)も入っていれば拾う: 組 ... p% ... odds
ODDS = re.compile(r'([1-6])-([1-6])-([1-6])\D{0,40}?[0-9.]+%\D{0,40}?([0-9]{1,3}(?:\.[0-9])?)\s')

def parse(html):
    probs = {}
    for m in ROW.finditer(html):
        combo = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        p = float(m.group(4))
        # 同じ組が複数マッチしたら最初(=AI予想確率表)を優先
        if combo not in probs:
            probs[combo] = p
    return probs

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
        got_any = False
        for r in range(1, 13):
            probs = fetch_race(hd, jcd, r)
            time.sleep(a.sleep)
            if probs and len(probs) >= 100:   # 120組そろってる想定
                out[f"{jcd}_{r}"] = probs
                got_any = True
            elif probs is None and r == 1:
                break   # その会場は非開催
        if got_any:
            print(f"  会場{jcd}: {sum(1 for k in out if k.startswith(f'{jcd}_'))}レース取得", flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{hd}.json")
    json.dump(out, open(path, "w"), ensure_ascii=False)
    print(f"[poseidon] {hd} {len(out)}レース保存 -> {path}", flush=True)

if __name__ == "__main__":
    main()
