
from __future__ import annotations

import itertools
import json
import math
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=9))
DB = Path("boat_race.db")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151 Safari/537.36"
    )
}
VENUES = {
    "桐生":"01","戸田":"02","江戸川":"03","平和島":"04","多摩川":"05","浜名湖":"06",
    "蒲郡":"07","常滑":"08","津":"09","三国":"10","びわこ":"11","住之江":"12",
    "尼崎":"13","鳴門":"14","丸亀":"15","児島":"16","宮島":"17","徳山":"18",
    "下関":"19","若松":"20","芦屋":"21","福岡":"22","唐津":"23","大村":"24"
}

st.set_page_config(page_title="WDJ Boat Race AI v7", layout="wide")


# ---------- DB ----------
def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      race_date TEXT,
      venue TEXT,
      race_no INTEGER,
      mode TEXT,
      fixed_at TEXT,
      deadline TEXT,
      classification TEXT,
      decision TEXT,
      confidence REAL,
      investment INTEGER,
      expected_value REAL,
      bets_json TEXT,
      weather_json TEXT,
      source_json TEXT,
      result_combo TEXT,
      official_payout INTEGER DEFAULT 0,
      actual_return INTEGER DEFAULT 0,
      profit INTEGER DEFAULT 0,
      settled INTEGER DEFAULT 0,
      season TEXT,
      wind_dir TEXT
    )""")
    con.commit()
    return con


# ---------- HTTP ----------
def fetch(url: str, retries: int = 2) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response.text
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(str(last_error))


def page_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def read_tables(html: str) -> list[pd.DataFrame]:
    """
    Renderのstatus 139対策。
    このアプリでは取得したHTMLテーブルを予想計算に使っていないため、
    pandas.read_html()を実行せず空配列を返す。
    """
    return []


# ---------- Sources ----------
def official_urls(date8: str, code: str, race_no: int) -> dict[str, str]:
    query = f"rno={race_no}&jcd={code}&hd={date8}"
    base = "https://www.boatrace.jp/owpc/pc/race"
    return {
        "出走表": f"{base}/racelist?{query}",
        "直前情報": f"{base}/beforeinfo?{query}",
        "3連単オッズ": f"{base}/odds3t?{query}",
        "結果": f"{base}/raceresult?{query}",
    }


def get_official(date8: str, code: str, race_no: int) -> dict[str, Any]:
    urls = official_urls(date8, code, race_no)
    output: dict[str, Any] = {"urls": urls, "errors": []}
    key_map = {"出走表":"list", "直前情報":"before", "3連単オッズ":"odds"}
    for label, key in key_map.items():
        try:
            html = fetch(urls[label])
            output[key] = {
                "html": html,
                "text": page_text(html),
                "tables": read_tables(html),
            }
        except Exception as exc:
            output["errors"].append(f"{label}: {exc}")
            output[key] = {"html":"", "text":"", "tables":[]}
    return output


def get_poseidon(date8: str, code: str, race_no: int) -> dict[str, Any]:
    url = f"https://poseidon-boatrace.net/race/{date8}/{int(code)}/{race_no}R"
    try:
        html = fetch(url)
        return {
            "url": url, "html": html, "text": page_text(html),
            "tables": read_tables(html), "error": None,
        }
    except Exception as exc:
        return {"url":url, "html":"", "text":"", "tables":[], "error":str(exc)}


def get_umepyon(date_iso: str, code: str, race_no: int) -> dict[str, Any]:
    # 検索ではなく公開されている予想ページ形式へ直接アクセス
    url = f"https://umepyon.com/predict.php?jcd={code}&racedate={date_iso}&racenum={race_no}"
    try:
        html = fetch(url)
        return {
            "url":url, "html":html, "text":page_text(html),
            "tables":read_tables(html), "error":None,
        }
    except Exception as exc:
        return {"url":url, "html":"", "text":"", "tables":[], "error":str(exc)}


# ---------- Parsing ----------
def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        cleaned = re.sub(r"[^0-9.\-]", "", str(value))
        return float(cleaned) if cleaned not in ("", "-", ".") else default
    except Exception:
        return default


def parse_poseidon_predictions(text: str) -> list[dict[str, float | str]]:
    patterns = [
        r"\b([1-6]-[1-6]-[1-6])\s+([0-9.]+)%\s+([0-9.]+)\s+([0-9.]+)pt",
        r"\b([1-6]-[1-6]-[1-6])\b.{0,30}?([0-9.]+)%"
        r".{0,30}?([0-9.]+).{0,30}?([0-9.]+)pt",
    ]
    found: dict[str, dict[str, float | str]] = {}
    for pattern in patterns:
        for combo, prob, odds, index in re.findall(pattern, text, re.S):
            if len(set(combo.split("-"))) != 3:
                continue
            row = {
                "combo": combo,
                "poseidon_prob": parse_float(prob),
                "odds": parse_float(odds),
                "poseidon_index": parse_float(index),
            }
            old = found.get(combo)
            if old is None or float(row["poseidon_prob"]) > float(old["poseidon_prob"]):
                found[combo] = row
    return list(found.values())


def parse_umepyon_predictions(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    patterns = [
        r"\b([1-6])[-－>]([1-6])[-－>]([1-6])\b[^%]{0,60}?([0-9.]+)\s*%",
        r"\b([1-6])\s+([1-6])\s+([1-6])\b[^%]{0,40}?([0-9.]+)\s*%",
    ]
    for pattern in patterns:
        for a, b, c, prob in re.findall(pattern, text, re.S):
            if len({a, b, c}) == 3:
                combo = f"{a}-{b}-{c}"
                result[combo] = max(result.get(combo, 0.0), parse_float(prob))
    return result


def parse_all_trifecta_odds(text: str) -> dict[str, float]:
    """複数サイトのテキストから「1 2 3 12.4」「1-2-3 12.4」を拾う。"""
    result: dict[str, float] = {}
    patterns = [
        r"\b([1-6])[-－\s]([1-6])[-－\s]([1-6])\s+([0-9]+(?:\.[0-9]+)?)\b",
        r"\b([1-6])-([1-6])-([1-6])\b.{0,12}?([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        for a, b, c, odds in re.findall(pattern, text):
            if len({a, b, c}) != 3:
                continue
            value = parse_float(odds)
            if value >= 1.0:
                result.setdefault(f"{a}-{b}-{c}", value)
    return result


def first_value(text: str, patterns: list[str], default: str = "取得エラー") -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return default


def parse_meta(official_text: str, poseidon_text: str) -> dict[str, Any]:
    combined = official_text + " " + poseidon_text
    wind = first_value(combined, [r"風速\s*([0-9.]+)\s*m", r"風\s*([0-9.]+)\s*m"])
    wave = first_value(combined, [r"波高\s*([0-9.]+)\s*cm", r"波\s*([0-9.]+)\s*cm"])
    wind_dir = first_value(combined, [r"風向\s*([^\s]+)"])
    weather = first_value(combined, [r"天候\s*([^\s]+)"])
    a1 = min(6, len(re.findall(r"\bA1\b", combined)))

    times: dict[str, str] = {}
    for lane, value in re.findall(r"\b([1-6])\s+([6-7]\.[0-9]{2})\b", combined):
        times.setdefault(lane, value)

    sts: dict[str, str] = {}
    for lane, value in re.findall(r"\b([1-6])\s+(F?[0-9]\.[0-9]{2})\b", combined):
        number = parse_float(value.replace("F", ""))
        if 0 <= number < 1:
            sts.setdefault(lane, value)

    rows = [
        {
            "艇": lane,
            "展示タイム": times.get(lane, "取得エラー"),
            "展示ST": sts.get(lane, "取得エラー"),
        }
        for lane in map(str, range(1, 7))
    ]
    return {
        "wind":wind, "wave":wave, "wind_dir":wind_dir, "weather":weather, "a1":a1,
        "exhibition_rows":rows,
        "exhibition_time_count":sum(r["展示タイム"] != "取得エラー" for r in rows),
        "exhibition_st_count":sum(r["展示ST"] != "取得エラー" for r in rows),
    }


def parse_lane_strength(poseidon_text: str, meta: dict[str, Any]) -> dict[str, float]:
    """
    データ不足時の順位生成用。枠の基礎値と展示順位を使う。
    実購入判定には使わず、参考予想の順位だけに使用する。
    """
    scores = {"1":100.0, "2":82.0, "3":73.0, "4":62.0, "5":50.0, "6":40.0}

    # 展示タイムは小さいほど加点
    valid_times = []
    for row in meta["exhibition_rows"]:
        tm = parse_float(row["展示タイム"], 0)
        if tm > 0:
            valid_times.append((row["艇"], tm))
    for rank, (lane, _) in enumerate(sorted(valid_times, key=lambda x:x[1]), start=1):
        scores[lane] += max(0, 15 - (rank - 1) * 3)

    # 展示STは小さいほど加点。F表記は少し減点。
    valid_st = []
    for row in meta["exhibition_rows"]:
        raw = str(row["展示ST"])
        st_value = parse_float(raw.replace("F", ""), 9)
        if st_value < 1:
            valid_st.append((row["艇"], st_value, raw.startswith("F")))
    for rank, (lane, _, is_f) in enumerate(sorted(valid_st, key=lambda x:x[1]), start=1):
        scores[lane] += max(0, 10 - (rank - 1) * 2)
        if is_f:
            scores[lane] -= 4

    # ポセイドン本文に級別が並ぶ場合の軽い補正
    for lane in map(str, range(1, 7)):
        block = re.search(rf"\b{lane}\b(.{{0,180}}?)(?=\b[1-6]\b|$)", poseidon_text, re.S)
        if block:
            body = block.group(1)
            if "A1" in body:
                scores[lane] += 18
            elif "A2" in body:
                scores[lane] += 10
            elif "B2" in body:
                scores[lane] -= 8
    return scores


def fallback_combos(scores: dict[str, float]) -> list[dict[str, Any]]:
    lanes = sorted(scores, key=scores.get, reverse=True)
    rows = []
    for combo in itertools.permutations(lanes[:5], 3):
        a, b, c = combo
        raw = scores[a] * 0.54 + scores[b] * 0.30 + scores[c] * 0.16
        rows.append({"combo":f"{a}-{b}-{c}", "raw":raw})
    rows.sort(key=lambda r:r["raw"], reverse=True)
    top = rows[:12]
    total = sum(math.exp((r["raw"] - top[0]["raw"]) / 18) for r in top)
    for row in top:
        row["model_prob"] = round(
            math.exp((row["raw"] - top[0]["raw"]) / 18) / total * 100, 2
        )
    return top


# ---------- Prediction ----------
def build_prediction(
    official: dict[str, Any],
    poseidon: dict[str, Any],
    umepyon: dict[str, Any],
) -> dict[str, Any]:
    official_text = " ".join(
        official[key]["text"] for key in ("list", "before", "odds")
    )
    meta = parse_meta(official_text, poseidon["text"])

    pose_rows = parse_poseidon_predictions(poseidon["text"])
    ume_probs = parse_umepyon_predictions(umepyon["text"])

    odds_map = {}
    odds_map.update(parse_all_trifecta_odds(official["odds"]["text"]))
    odds_map.update({
        k:v for k,v in parse_all_trifecta_odds(poseidon["text"]).items()
        if k not in odds_map
    })

    merged: dict[str, dict[str, Any]] = {}
    for row in pose_rows:
        combo = str(row["combo"])
        merged[combo] = dict(row)
    for combo, prob in ume_probs.items():
        merged.setdefault(combo, {
            "combo":combo, "poseidon_prob":0.0, "poseidon_index":0.0
        })
        merged[combo]["ume_prob"] = prob

    # 外部AIの買い目が取れない場合も、展示＋枠から参考順位を生成
    if not merged:
        scores = parse_lane_strength(poseidon["text"], meta)
        for row in fallback_combos(scores):
            merged[row["combo"]] = {
                "combo":row["combo"],
                "poseidon_prob":0.0,
                "ume_prob":0.0,
                "poseidon_index":0.0,
                "fallback_prob":row["model_prob"],
            }

    rows = []
    for combo, row in merged.items():
        pose_prob = parse_float(row.get("poseidon_prob"))
        ume_prob = parse_float(row.get("ume_prob"))
        fallback_prob = parse_float(row.get("fallback_prob"))

        available = [p for p in (pose_prob, ume_prob) if p > 0]
        if len(available) == 2:
            model_prob = pose_prob * 0.72 + ume_prob * 0.28
        elif len(available) == 1:
            model_prob = available[0]
        else:
            model_prob = fallback_prob

        odds = odds_map.get(combo, parse_float(row.get("odds")))
        market_prob = 100 / odds if odds > 0 else 0
        ev = model_prob / 100 * odds if odds > 0 else 0
        divergence = model_prob / market_prob if market_prob > 0 else 0
        rows.append({
            "combo":combo,
            "model_prob":round(model_prob, 2),
            "poseidon_prob":pose_prob or None,
            "ume_prob":ume_prob or None,
            "poseidon_index":parse_float(row.get("poseidon_index")),
            "odds":round(odds, 1) if odds else 0,
            "market_prob":round(market_prob, 2),
            "ev":round(ev, 3),
            "divergence":round(divergence, 2),
        })

    rows.sort(key=lambda r:(r["model_prob"], r["ev"]), reverse=True)

    # 少なくとも8点は順位表示
    if len(rows) < 8:
        existing = {r["combo"] for r in rows}
        scores = parse_lane_strength(poseidon["text"], meta)
        for fallback in fallback_combos(scores):
            if fallback["combo"] in existing:
                continue
            odds = odds_map.get(fallback["combo"], 0)
            prob = fallback["model_prob"]
            rows.append({
                "combo":fallback["combo"], "model_prob":prob,
                "poseidon_prob":None, "ume_prob":None, "poseidon_index":0,
                "odds":odds, "market_prob":round(100/odds,2) if odds else 0,
                "ev":round(prob/100*odds,3) if odds else 0,
                "divergence":round(prob/(100/odds),2) if odds else 0,
            })
            if len(rows) >= 8:
                break

    try:
        wind = float(meta["wind"])
    except Exception:
        wind = None

    official_ok = not official["errors"]
    poseidon_ok = not poseidon["error"]
    umepyon_ok = not umepyon["error"]
    exhibition_ok = (
        meta["exhibition_time_count"] >= 6
        and meta["exhibition_st_count"] >= 6
    )
    odds_count = sum(r["odds"] > 0 for r in rows)
    ev_rows = [r for r in rows if r["odds"] > 0 and r["ev"] >= 1.10]

    # 最新WDJ115-v2
    if not official_ok:
        classification = "見送り"
    elif wind is not None and wind >= 5:
        classification = "大荒れ候補"
    elif wind is not None and 2 <= wind <= 3 and meta["a1"] <= 1:
        classification = "中波乱"
    elif rows and rows[0]["model_prob"] >= 10:
        classification = "堅い"
    else:
        classification = "大荒れ候補"

    eligible = (
        classification == "中波乱"
        and official_ok
        and exhibition_ok
        and odds_count >= 6
        and len(ev_rows) >= 6
    )

    # 予想は毎回表示。購入は条件適合時だけ。
    if eligible:
        selected = sorted(ev_rows, key=lambda r:(r["ev"], r["model_prob"]), reverse=True)[:8]
        if len(selected) < 6:
            selected = rows[:8]
            eligible = False
    else:
        selected = rows[:8]

    stars = 1
    source_count = sum((official_ok, poseidon_ok, umepyon_ok))
    if source_count >= 1:
        stars += 1
    if source_count >= 2:
        stars += 1
    if exhibition_ok:
        stars += 1
    if eligible:
        stars += 1
    stars = min(stars, 5)

    action = "買い" if eligible else "見送り"
    budget = 5000 if eligible else 0

    # 100円単位の資金配分
    if budget:
        weights = []
        for row in selected:
            odds = max(row["odds"], 1.0)
            weights.append(
                max(row["model_prob"], 0.1) ** 0.7 / odds ** 0.35
            )
        total_weight = sum(weights) or 1
        amounts = [
            max(100, round(budget * w / total_weight / 100) * 100)
            for w in weights
        ]
        diff = budget - sum(amounts)
        idx = 0
        while diff != 0 and idx < 500:
            j = idx % len(amounts)
            step = 100 if diff > 0 else -100
            if amounts[j] + step >= 100:
                amounts[j] += step
                diff -= step
            idx += 1
    else:
        amounts = [0] * len(selected)

    predictions = []
    for rank, (row, amount) in enumerate(zip(selected, amounts), start=1):
        payout = int(amount * row["odds"]) if amount and row["odds"] else 0
        profit = payout - budget if payout else 0
        predictions.append({
            **row, "rank":rank, "amount":amount,
            "return":payout, "profit":profit,
        })

    # どの買い目でも赤字なら購入しない
    if eligible and any(r["return"] <= budget for r in predictions):
        eligible = False
        action = "見送り"
        budget = 0
        for row in predictions:
            row["amount"] = row["return"] = row["profit"] = 0
        stars = min(stars, 4)

    # 頭・相手順位
    head_score: dict[str, float] = {}
    opponent_score: dict[str, float] = {}
    for row in rows:
        a, b, c = row["combo"].split("-")
        head_score[a] = head_score.get(a, 0) + row["model_prob"]
        opponent_score[b] = opponent_score.get(b, 0) + row["model_prob"] * 0.65
        opponent_score[c] = opponent_score.get(c, 0) + row["model_prob"] * 0.35

    head_rank = sorted(head_score, key=head_score.get, reverse=True)[:2]
    opponent_rank = sorted(opponent_score, key=opponent_score.get, reverse=True)[:4]
    mismatch = max(rows, key=lambda r:r["divergence"]) if rows else None

    confidence = {1:15, 2:30, 3:50, 4:70, 5:86}[stars]
    reason = (
        f"取得源 {source_count}/3／展示タイム {meta['exhibition_time_count']}/6／"
        f"展示ST {meta['exhibition_st_count']}/6／A1 {meta['a1']}艇／"
        f"風速 {meta['wind']}m／オッズ取得 {odds_count}点／"
        + ("WDJ115-v2購入条件適合" if eligible else "WDJ115-v2購入条件未達")
    )

    return {
        "meta":meta, "rows":rows, "predictions":predictions,
        "classification":classification, "action":action,
        "eligible":eligible, "budget":budget, "stars":stars,
        "confidence":confidence, "head_rank":head_rank,
        "opponent_rank":opponent_rank, "mismatch":mismatch,
        "reason":reason,
        "avg_ev":round(
            sum(r["ev"] for r in selected) / max(len(selected), 1), 3
        ),
    }


def save_prediction(record: dict[str, Any]) -> None:
    con = db()
    con.execute("""
    INSERT INTO predictions(
      race_date,venue,race_no,mode,fixed_at,deadline,classification,
      decision,confidence,investment,expected_value,bets_json,
      weather_json,source_json,season,wind_dir
    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record["race_date"], record["venue"], record["race_no"],
        record["mode"], record["fixed_at"], record["deadline"],
        record["classification"], record["decision"], record["confidence"],
        record["investment"], record["expected_value"],
        json.dumps(record["bets"], ensure_ascii=False),
        json.dumps(record["weather"], ensure_ascii=False),
        json.dumps(record["sources"], ensure_ascii=False),
        record["season"], record["weather"]["wind_dir"],
    ))
    con.commit()


def season_name(month: int) -> str:
    if month in (3,4,5):
        return "春"
    if month in (6,7,8):
        return "夏"
    if month in (9,10,11):
        return "秋"
    return "冬"


# ---------- UI ----------
st.title("🚤 WDJ ボートレースAI Web版 v7")
st.caption(
    "公式・梅吉AI・ポセイドンを統合。予想は毎回表示し、"
    "WDJ115-v2条件に適合したレースだけ5,000円で仮想購入します。"
)

tabs = st.tabs(["予想", "締切3分前監視", "成績ダッシュボード", "結果登録"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    race_date = c1.date_input("開催日", datetime.now(JST).date())
    venue = c2.selectbox("場", list(VENUES))
    race_no = c3.selectbox("R", range(1,13))
    deadline = st.time_input("締切時刻")
    real_purchase = st.checkbox("実購入として記録する", value=False)

    if st.button("3サイト取得 → 予想固定", type="primary", use_container_width=True):
        date8 = race_date.strftime("%Y%m%d")
        date_iso = race_date.strftime("%Y-%m-%d")
        code = VENUES[venue]

        with st.spinner("公式・梅吉AI・ポセイドンを取得し、予想中…"):
            official = get_official(date8, code, race_no)
            poseidon = get_poseidon(date8, code, race_no)
            umepyon = get_umepyon(date_iso, code, race_no)
            calc = build_prediction(official, poseidon, umepyon)

        fixed_at = datetime.now(JST)
        mode = "real" if real_purchase else "virtual"

        record = {
            "race_date":str(race_date), "venue":venue, "race_no":race_no,
            "mode":mode, "fixed_at":fixed_at.isoformat(),
            "deadline":f"{race_date} {deadline}",
            "classification":calc["classification"],
            "decision":calc["action"],
            "confidence":calc["confidence"],
            "investment":calc["budget"] if real_purchase and calc["eligible"] else 0,
            "expected_value":calc["avg_ev"],
            "bets":calc["predictions"],
            "weather":calc["meta"],
            "season":season_name(race_date.month),
            "sources":{
                "official":official["urls"],
                "official_errors":official["errors"],
                "poseidon":poseidon["url"],
                "poseidon_error":poseidon["error"],
                "umepyon":umepyon["url"],
                "umepyon_error":umepyon["error"],
            },
        }
        save_prediction(record)

        star_text = "★" * calc["stars"] + "☆" * (5 - calc["stars"])
        st.success(f"{venue}{race_no}R｜締切前固定｜{calc['action']}")
        st.markdown(f"## {star_text}　{calc['action']}")

        a,b,c,d = st.columns(4)
        a.metric("分類", calc["classification"])
        b.metric("最終判断", calc["action"])
        c.metric("自信度", star_text)
        d.metric("参考期待値", calc["avg_ev"])

        st.subheader("3サイト取得状況")
        s1,s2,s3 = st.columns(3)
        s1.metric("BOAT RACE公式", "OK" if not official["errors"] else "一部エラー")
        s2.metric("ポセイドン", "OK" if not poseidon["error"] else "エラー")
        s3.metric("梅吉AI", "OK" if not umepyon["error"] else "エラー")

        meta = calc["meta"]
        st.subheader("直前情報")
        w1,w2,w3,w4 = st.columns(4)
        w1.metric("天候", meta["weather"])
        w2.metric("風向", meta["wind_dir"])
        w3.metric("風速", f"{meta['wind']}m")
        w4.metric("波高", f"{meta['wave']}cm")
        st.dataframe(
            pd.DataFrame(meta["exhibition_rows"]),
            hide_index=True, use_container_width=True
        )

        st.subheader("軸と相手順位")
        h1,h2 = st.columns(2)
        h1.write("**頭候補：** " + (" → ".join(calc["head_rank"]) or "取得エラー"))
        h2.write("**相手順位：** " + (" → ".join(calc["opponent_rank"]) or "取得エラー"))

        st.subheader("買い目・オッズ・購入額・払戻")
        prediction_df = pd.DataFrame(calc["predictions"]).rename(columns={
            "rank":"順位", "combo":"買い目", "odds":"直前オッズ",
            "amount":"購入額", "return":"的中時払戻", "profit":"純利益",
            "model_prob":"推定確率", "market_prob":"市場確率",
            "ev":"期待値", "divergence":"乖離倍率",
            "poseidon_prob":"ポセイドン確率", "ume_prob":"梅吉確率",
            "poseidon_index":"海神指数",
        })
        columns = [
            "順位","買い目","直前オッズ","購入額","的中時払戻","純利益",
            "推定確率","市場確率","期待値","乖離倍率",
            "ポセイドン確率","梅吉確率","海神指数",
        ]
        st.dataframe(
            prediction_df[[c for c in columns if c in prediction_df.columns]],
            hide_index=True, use_container_width=True
        )

        payouts = [r["return"] for r in calc["predictions"] if r["return"] > 0]
        profits = [r["profit"] for r in calc["predictions"] if r["return"] > 0]
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("総投資", f"{calc['budget']:,}円")
        m2.metric("最大払戻", f"{max(payouts):,}円" if payouts else "取得不可")
        m3.metric("最小払戻", f"{min(payouts):,}円" if payouts else "取得不可")
        m4.metric("最大純利益", f"{max(profits):,}円" if profits else "取得不可")

        mismatch = calc["mismatch"]
        st.subheader("オッズ矛盾候補")
        if mismatch:
            st.write(
                f"**{mismatch['combo']}**｜推定確率 {mismatch['model_prob']}%｜"
                f"市場確率 {mismatch['market_prob']}%｜期待値 {mismatch['ev']}｜"
                f"乖離倍率 {mismatch['divergence']}倍"
            )

        st.subheader("総合判断理由")
        st.write(calc["reason"])
        if calc["action"] == "買い":
            st.success("WDJ115-v2条件適合：総額5,000円で固定しました。")
        else:
            st.warning("予想順位は固定しましたが、購入条件未達のため0円見送りです。")

        with st.expander("参照URL・取得エラー"):
            for label,url in official["urls"].items():
                st.write(f"公式 {label}: {url}")
            st.write(f"ポセイドン: {poseidon['url']}")
            st.write(f"梅吉AI: {umepyon['url']}")
            if official["errors"]:
                st.write("公式エラー:", official["errors"])
            if poseidon["error"]:
                st.write("ポセイドンエラー:", poseidon["error"])
            if umepyon["error"]:
                st.write("梅吉AIエラー:", umepyon["error"])

with tabs[1]:
    st.write(
        "画面を開いている間は15秒ごとに確認します。"
        "締切3分前になったら予想タブで再取得・固定してください。"
    )
    mc1,mc2,mc3 = st.columns(3)
    monitor_date = mc1.date_input("監視日", key="monitor_date")
    monitor_venue = mc2.selectbox("監視場", list(VENUES), key="monitor_venue")
    monitor_race = mc3.selectbox("監視R", range(1,13), key="monitor_race")
    monitor_deadline = st.time_input("締切", key="monitor_deadline")
    active = st.toggle("自動監視ON", value=False)
    if active:
        target = datetime.combine(monitor_date, monitor_deadline, tzinfo=JST)
        remain = int((target - datetime.now(JST)).total_seconds())
        st.info(f"締切まで {max(0, remain)} 秒")
        if 0 < remain <= 180:
            st.warning("締切3分前です。予想タブで最新データを取得して固定してください。")
        time.sleep(15)
        st.rerun()

with tabs[2]:
    con = db()
    history = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
    if history.empty:
        st.info("予想データがまだありません。")
    else:
        settled = history[history["settled"] == 1]
        investment = int(settled["investment"].sum()) if not settled.empty else 0
        returned = int(settled["actual_return"].sum()) if not settled.empty else 0
        profit = returned - investment
        roi = returned / investment * 100 if investment else 0
        q1,q2,q3,q4 = st.columns(4)
        q1.metric("確定レース", len(settled))
        q2.metric("投資", f"{investment:,}円")
        q3.metric("収支", f"{profit:,}円")
        q4.metric("回収率", f"{roi:.1f}%")

        if not settled.empty:
            for group,label in [
                ("venue","場別"), ("season","季節別"), ("wind_dir","風向別")
            ]:
                report = settled.groupby(group, dropna=False).agg(
                    レース数=("id","count"),
                    投資=("investment","sum"),
                    回収=("actual_return","sum"),
                ).reset_index()
                report["収支"] = report["回収"] - report["投資"]
                report["回収率"] = report.apply(
                    lambda row: row["回収"]/row["投資"]*100 if row["投資"] else 0,
                    axis=1
                )
                st.subheader(label)
                st.dataframe(report, hide_index=True, use_container_width=True)

        st.subheader("固定予想履歴")
        st.dataframe(
            history[[
                "race_date","venue","race_no","mode","fixed_at",
                "classification","decision","investment",
                "expected_value","settled","profit"
            ]],
            hide_index=True, use_container_width=True
        )

with tabs[3]:
    con = db()
    history = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
    if history.empty:
        st.info("結果登録対象がありません。")
    else:
        labels = {
            int(row.id):f"#{row.id} {row.race_date} {row.venue}{row.race_no}R {row.decision}"
            for _,row in history.iterrows()
        }
        prediction_id = st.selectbox(
            "対象予想", list(labels), format_func=lambda x:labels[x]
        )
        result_combo = st.text_input("確定3連単（例：1-2-3）")
        official_payout = st.number_input(
            "公式3連単払戻（100円あたり）", min_value=0, step=10
        )
        actual_return = st.number_input("実回収額", min_value=0, step=100)

        if st.button("結果を確定"):
            investment = int(
                history.loc[history.id == prediction_id, "investment"].iloc[0]
            )
            profit = int(actual_return) - investment
            con.execute("""
            UPDATE predictions
            SET result_combo=?, official_payout=?, actual_return=?,
                profit=?, settled=1
            WHERE id=?
            """, (
                result_combo, int(official_payout), int(actual_return),
                profit, prediction_id
            ))
            con.commit()
            st.success("結果を確定しました。締切前予想は変更していません。")
