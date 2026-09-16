#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""poseidon_scrape.py — ポセイドンの3連単120通り [AI予想確率・オッズ・海神指数] を収集。

静的HTML(裏API無し)なので requests のみ。オッズ/海神指数は出走15分前〜レース後に入る(前夜は---/未算出)。
  URL: https://poseidon-boatrace.net/race/{YYYYMMDD}/{jcd}/{R}R
出力: data/poseidon/{YYYYMMDD}.json = { "jcd_R": {"i-j-k":[prob, odds|null, kaijin|null], ...}, ... }
  ・前夜に実行 → probだけ入りオッズはnull(記録用)。
  ・レース後(夜)に実行 → 最終オッズ・海神指数まで入る(EV精算用)。同じ日を上書き更新。
使い方: python poseidon_scrape.py --hd 20260821

v69(2026-09-16):
  --merge            既存ファイルの「AI予想確率」は前夜取得の値をそのまま残し、オッズ/海神指数だけを埋める。
                     確率はレース前に取った値が前向き検証の正本なので、レース後の再取得で上書きしない。
                     (実測: 9/10 1場1R の 1-2-3/1-2-4/1-3-2 はレース後ページでも確率が完全一致。念のため保護)
  --only-missing-odds 既にオッズが入っているレースは取りに行かない(過去日の取り直し・中断再開用)。
  --lag-hours N      --hd 省略時の対象日を「JST現在 − N時間」の日付にする。
                     GitHubの定時実行は数時間遅れて日付をまたぐことがあり、23:40予定の再取得が
                     翌日(レース前)を取ってしまい、8/30以降オッズが1日も入っていなかった原因。
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

def has_odds(race):
    return any(isinstance(c, list) and len(c) > 1 and c[1] is not None for c in race.values())

def merge(old, new):
    """確率は旧(レース前)を優先、オッズ/海神指数は新(レース後)を優先。取れなかったレースは旧を残す。"""
    out = {}; kept = changed = 0
    for key in set(old) | set(new):
        o = old.get(key) or {}; n = new.get(key)
        if not n:
            out[key] = o; kept += 1
            continue
        if not o:
            out[key] = n
            continue
        race = {}
        for combo in set(o) | set(n):
            ov = (list(o.get(combo) or []) + [None, None, None])[:3]
            nv = (list(n.get(combo) or []) + [None, None, None])[:3]
            p = ov[0] if ov[0] is not None else nv[0]
            if ov[0] is not None and nv[0] is not None and abs(ov[0] - nv[0]) > 1e-9:
                changed += 1
            race[combo] = [p, nv[1] if nv[1] is not None else ov[1], nv[2] if nv[2] is not None else ov[2]]
        out[key] = race
    return out, kept, changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd", default="")
    ap.add_argument("--out", default="data/poseidon")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--only-missing-odds", action="store_true")
    ap.add_argument("--lag-hours", type=float, default=0.0)
    a = ap.parse_args()
    hd = a.hd or (dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
                  - dt.timedelta(hours=a.lag_hours)).strftime("%Y%m%d")
    path = os.path.join(a.out, f"{hd}.json")
    old = {}
    if (a.merge or a.only_missing_odds) and os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:
            old = {}
    out = {}
    skipped = 0
    for jcd in range(1, 25):
        got = 0
        for r in range(1, 13):
            key = f"{jcd}_{r}"
            if a.only_missing_odds and key in old and has_odds(old[key]):
                out[key] = old[key]; got += 1; skipped += 1
                continue
            probs = fetch_race(hd, jcd, r)
            time.sleep(a.sleep)
            if probs and len(probs) >= 100:
                out[f"{jcd}_{r}"] = probs
                got += 1
            elif probs is None and r == 1 and not any(k.startswith(f"{jcd}_") for k in old):
                break
        if got:
            print(f"  会場{jcd}: {got}レース", flush=True)
    kept = changed = 0
    if a.merge and old:
        out, kept, changed = merge(old, out)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(path, "w"), ensure_ascii=False)
    if a.merge or a.only_missing_odds:
        print(f"[poseidon] merge: 取得済みスキップ{skipped} / 取れず旧データ維持{kept}レース / "
              f"確率がレース後に変わっていた組{changed}(旧=レース前の値を採用)", flush=True)
    # オッズが入っているレース数(=レース後スクレイプの目安)
    with_odds = sum(1 for v in out.values() if any(c[1] for c in v.values()))
    print(f"[poseidon] {hd} {len(out)}レース保存(うちオッズ有 {with_odds}) -> {path}", flush=True)

if __name__ == "__main__":
    main()
