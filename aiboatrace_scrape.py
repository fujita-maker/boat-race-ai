#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aiboatrace_scrape.py — AI BOAT RACE(aiboatrace.jp)の各艇AI1着率と推奨目のAI確率を収集。

対象は1日約80レース(サイトが選んだレースだけ)。レースページは当日中に順次出て、翌日には404で消える。
過去にさかのぼれないので、日中に何度も回って新しく出たページを保存する。
  一覧: https://aiboatrace.jp/            (race/YYYY-MM-DD/JJ_rNN へのリンク)
  各R : https://aiboatrace.jp/race/YYYY-MM-DD/JJ_rNN.html
出力: data/aiboatrace/{YYYYMMDD}.json
  { "jcd_R": {"first": ISO, "last": ISO,
              "boats": {"1": {"ai1": AI1着率%, "tenji": 展示T, "st_ex": ST展示, "grade": "A2"}, ...},
              "picks": [{"combo":"1-2-3","role":"本線","ai_p":8.83,"odds":12.6,"mkt_p":5.95,"ev":1.11,"alloc":2200}, ...],
              "snaps": [ {"t": ISO, "picks": [...]} ]  ← 推奨目/オッズが変わったときだけ追加 } }
使い方: python aiboatrace_scrape.py --out data/aiboatrace
"""
import argparse, json, os, re, time
import datetime as dt
import requests
from bs4 import BeautifulSoup

BASE = "https://aiboatrace.jp"
UA = {"User-Agent": "Mozilla/5.0 (compatible; research/1.0)"}
LINK = re.compile(r"race/(\d{4}-\d{2}-\d{2})/(\d{2})_r(\d{2})")
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def now():
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def jst_today():
    return (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y-%m-%d")


def get(url, timeout=25, retries=3):
    last = None
    for i in range(retries):
        try:
            return requests.get(url, headers=UA, timeout=timeout)
        except requests.RequestException as e:
            last = e
            time.sleep(2 + 2 * i)
    raise last


def _num(s):
    m = NUM.search((s or "").replace(",", ""))
    return float(m.group()) if m else None


def _rows(tbl):
    rows = []
    for tr in tbl.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    return rows


def parse(html):
    soup = BeautifulSoup(html, "lxml")
    boats, picks = {}, []
    for tbl in soup.find_all("table"):
        rows = _rows(tbl)
        if len(rows) < 2:
            continue
        head = rows[0]
        if "AI1着率" in head and "枠" in head:
            ix = {h: i for i, h in enumerate(head)}
            for r in rows[1:]:
                if len(r) < len(head) or not re.fullmatch(r"[1-6]", r[0]):
                    continue
                b = {"ai1": _num(r[ix["AI1着率"]])}
                if "展示T" in ix:
                    b["tenji"] = _num(r[ix["展示T"]])
                if "ST展示" in ix:
                    b["st_ex"] = r[ix["ST展示"]]
                if "級" in ix:
                    b["grade"] = r[ix["級"]]
                boats[r[0]] = b
        elif "買い目" in head and "AI確率" in head and "市場確率" in head:
            ix = {h: i for i, h in enumerate(head)}
            for r in rows[1:]:
                if len(r) < len(head):
                    continue
                combo = "-".join(re.findall(r"[1-6]", r[ix["買い目"]]))
                if not combo:
                    continue
                picks.append({
                    "combo": combo,
                    "role": r[ix["役割"]] if "役割" in ix else None,
                    "ai_p": _num(r[ix["AI確率"]]),
                    "odds": _num(r[ix["オッズ"]]) if "オッズ" in ix else None,
                    "mkt_p": _num(r[ix["市場確率"]]),
                    "ev": _num(r[ix["EV"]]) if "EV" in ix else None,
                    "alloc": _num(r[ix["配分"]]) if "配分" in ix else None,
                })
    return boats, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/aiboatrace")
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    idx = get(BASE + "/")
    idx.encoding = idx.apparent_encoding or "utf-8"
    today = jst_today()
    races = sorted({(d, j, int(r)) for d, j, r in LINK.findall(idx.text) if d == today})
    hd = today.replace("-", "")
    path = os.path.join(a.out, f"{hd}.json")
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    print(f"{today}: 一覧 {len(races)}R / 既存 {len(data)}R")

    new = upd = err = 0
    for d, jcd, r in races:
        key = f"{jcd}_{r}"
        url = f"{BASE}/race/{d}/{jcd}_r{r:02d}.html"
        try:
            resp = get(url)
        except requests.RequestException as e:
            print("  ERR", key, e)
            err += 1
            continue
        if resp.status_code != 200:
            err += 1
            continue
        resp.encoding = resp.apparent_encoding or "utf-8"
        boats, picks = parse(resp.text)
        if not boats and not picks:
            err += 1
            continue
        t = now()
        rec = data.get(key)
        if rec is None:
            data[key] = {"first": t, "last": t, "boats": boats, "picks": picks,
                         "snaps": [{"t": t, "picks": picks}]}
            new += 1
        else:
            rec["last"] = t
            if boats:
                rec["boats"] = boats       # 展示後の値で上書き(AI1着率は最終版)
            if picks and picks != rec["snaps"][-1]["picks"]:
                rec["snaps"].append({"t": t, "picks": picks})
                rec["picks"] = picks
                upd += 1
        time.sleep(a.sleep)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    print(f"新規 {new} / 推奨目変化 {upd} / 失敗 {err} / 合計 {len(data)}R")


if __name__ == "__main__":
    main()
