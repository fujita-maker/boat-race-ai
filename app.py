
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

st.set_page_config(page_title="WDJ Boat Race AI v14", layout="wide")


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
    con.execute("""
    CREATE TABLE IF NOT EXISTS model_config(
      id INTEGER PRIMARY KEY CHECK(id=1),
      poseidon_weight REAL DEFAULT 0.72,
      umepyon_weight REAL DEFAULT 0.28,
      ev_threshold REAL DEFAULT 1.10,
      max_bets INTEGER DEFAULT 8,
      updated_at TEXT,
      training_rows INTEGER DEFAULT 0,
      training_races INTEGER DEFAULT 0,
      backtest_roi REAL DEFAULT 0,
      backtest_profit INTEGER DEFAULT 0
    )""")
    con.execute("""
    INSERT OR IGNORE INTO model_config(
      id,poseidon_weight,umepyon_weight,ev_threshold,max_bets,updated_at
    ) VALUES(1,0.72,0.28,1.10,8,'初期値')
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS historical_candidates(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      race_date TEXT,
      venue TEXT,
      race_no INTEGER,
      combo TEXT,
      odds REAL,
      poseidon_prob REAL,
      umepyon_prob REAL,
      fallback_prob REAL,
      actual_combo TEXT,
      payout_100 INTEGER,
      imported_at TEXT
    )""")
    con.execute("""
    CREATE TABLE IF NOT EXISTS historical_results(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      race_date TEXT,
      venue TEXT,
      venue_code TEXT,
      race_no INTEGER,
      actual_combo TEXT,
      payout_100 INTEGER,
      weather TEXT,
      wind_dir TEXT,
      wind_speed REAL,
      wave_height REAL,
      winning_method TEXT,
      source_url TEXT,
      collected_at TEXT,
      UNIQUE(race_date,venue_code,race_no)
    )""")
    con.execute("""
    CREATE TABLE IF NOT EXISTS collection_state(
      id INTEGER PRIMARY KEY CHECK(id=1),
      next_date TEXT,
      start_date TEXT,
      end_date TEXT,
      selected_venues TEXT,
      updated_at TEXT,
      collected_races INTEGER DEFAULT 0,
      failed_requests INTEGER DEFAULT 0
    )""")
    con.execute("""
    INSERT OR IGNORE INTO collection_state(
      id,next_date,start_date,end_date,selected_venues,updated_at,
      collected_races,failed_requests
    ) VALUES(1,'','','','[]','未開始',0,0)
    """)
    con.commit()
    return con



# ---------- Learning configuration ----------
def get_model_config() -> dict[str, Any]:
    con = db()
    row = con.execute("""
      SELECT poseidon_weight,umepyon_weight,ev_threshold,max_bets,
             updated_at,training_rows,training_races,backtest_roi,backtest_profit
      FROM model_config WHERE id=1
    """).fetchone()
    if not row:
        return {
            "poseidon_weight":0.72, "umepyon_weight":0.28,
            "ev_threshold":1.10, "max_bets":8, "updated_at":"初期値",
            "training_rows":0, "training_races":0,
            "backtest_roi":0.0, "backtest_profit":0,
        }
    keys = [
        "poseidon_weight","umepyon_weight","ev_threshold","max_bets",
        "updated_at","training_rows","training_races","backtest_roi","backtest_profit",
    ]
    return dict(zip(keys, row))


def save_model_config(config: dict[str, Any]) -> None:
    con = db()
    con.execute("""
      UPDATE model_config SET
        poseidon_weight=?,umepyon_weight=?,ev_threshold=?,max_bets=?,
        updated_at=?,training_rows=?,training_races=?,backtest_roi=?,backtest_profit=?
      WHERE id=1
    """, (
        float(config["poseidon_weight"]),
        float(config["umepyon_weight"]),
        float(config["ev_threshold"]),
        int(config["max_bets"]),
        str(config["updated_at"]),
        int(config["training_rows"]),
        int(config["training_races"]),
        float(config["backtest_roi"]),
        int(config["backtest_profit"]),
    ))
    con.commit()


TRAINING_COLUMNS = [
    "race_date","venue","race_no","combo","odds",
    "poseidon_prob","umepyon_prob","fallback_prob",
    "actual_combo","payout_100",
]


def normalize_training_df(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    missing = [c for c in TRAINING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("不足列: " + ", ".join(missing))

    df = df[TRAINING_COLUMNS].copy()
    df["race_date"] = df["race_date"].astype(str)
    df["venue"] = df["venue"].astype(str)
    df["combo"] = df["combo"].astype(str).str.strip()
    df["actual_combo"] = df["actual_combo"].astype(str).str.strip()

    for col in [
        "race_no","odds","poseidon_prob","umepyon_prob",
        "fallback_prob","payout_100",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[
        df["combo"].str.match(r"^[1-6]-[1-6]-[1-6]$")
        & df["actual_combo"].str.match(r"^[1-6]-[1-6]-[1-6]$")
    ].copy()
    df = df[df["odds"] > 0].copy()
    df["race_id"] = (
        df["race_date"] + "|" + df["venue"] + "|" + df["race_no"].astype(int).astype(str)
    )
    return df


def candidate_probability(row: pd.Series, poseidon_weight: float) -> float:
    pose = float(row.get("poseidon_prob", 0) or 0)
    ume = float(row.get("umepyon_prob", 0) or 0)
    fallback = float(row.get("fallback_prob", 0) or 0)

    if pose > 0 and ume > 0:
        return pose * poseidon_weight + ume * (1 - poseidon_weight)
    if pose > 0:
        return pose
    if ume > 0:
        return ume
    return fallback


def backtest_training(
    df: pd.DataFrame,
    poseidon_weight: float,
    ev_threshold: float,
    max_bets: int,
) -> dict[str, Any]:
    investment = 0
    returned = 0
    purchased_races = 0
    hit_races = 0
    all_rows: list[dict[str, Any]] = []

    for race_id, race in df.groupby("race_id", sort=False):
        work = race.copy()
        work["model_prob"] = work.apply(
            lambda row: candidate_probability(row, poseidon_weight), axis=1
        )
        work["ev"] = work["model_prob"] / 100 * work["odds"]
        selected = work[work["ev"] >= ev_threshold].sort_values(
            ["ev","model_prob"], ascending=False
        ).head(max_bets)

        if selected.empty:
            continue

        purchased_races += 1
        race_investment = len(selected) * 100
        investment += race_investment
        actual_combo = str(work["actual_combo"].iloc[0])
        payout_100 = int(work["payout_100"].iloc[0])

        hit = actual_combo in set(selected["combo"])
        race_return = payout_100 if hit else 0
        if hit:
            hit_races += 1
        returned += race_return

        all_rows.append({
            "race_id":race_id,
            "購入点数":len(selected),
            "投資":race_investment,
            "的中":1 if hit else 0,
            "回収":race_return,
            "収支":race_return-race_investment,
        })

    profit = returned - investment
    roi = returned / investment * 100 if investment else 0
    hit_rate = hit_races / purchased_races * 100 if purchased_races else 0

    return {
        "investment":investment,
        "returned":returned,
        "profit":profit,
        "roi":roi,
        "purchased_races":purchased_races,
        "hit_races":hit_races,
        "hit_rate":hit_rate,
        "details":pd.DataFrame(all_rows),
    }


def optimize_training(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    results: list[dict[str, Any]] = []

    for weight_step in range(0, 21):
        poseidon_weight = weight_step / 20
        for ev_step in range(80, 161, 5):
            ev_threshold = ev_step / 100
            for max_bets in (3, 4, 5, 6, 8):
                report = backtest_training(
                    df, poseidon_weight, ev_threshold, max_bets
                )
                if report["purchased_races"] < 10:
                    continue
                results.append({
                    "poseidon_weight":poseidon_weight,
                    "umepyon_weight":1-poseidon_weight,
                    "ev_threshold":ev_threshold,
                    "max_bets":max_bets,
                    "investment":report["investment"],
                    "returned":report["returned"],
                    "profit":report["profit"],
                    "roi":report["roi"],
                    "purchased_races":report["purchased_races"],
                    "hit_rate":report["hit_rate"],
                })

    if not results:
        raise ValueError("学習可能な購入レースが10件未満です。データを増やしてください。")

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        ["profit","roi","purchased_races"], ascending=False
    ).reset_index(drop=True)
    return result_df.iloc[0].to_dict(), result_df



# ---------- Historical collection ----------
def parse_historical_result(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    combo_match = re.search(
        r"3連単\s*([1-6])\s*[-－]\s*([1-6])\s*[-－]\s*([1-6])"
        r"\s*[¥￥]?\s*([0-9,]+)",
        text,
    )
    if not combo_match:
        return None

    combo = "-".join(combo_match.group(i) for i in (1,2,3))
    payout = int(combo_match.group(4).replace(",", ""))

    weather = first_value(text, [r"水面気象情報.*?(晴|曇り|雨|雪|霧)", r"\b(晴|曇り|雨|雪|霧)\b"])
    wind_speed = parse_float(first_value(text, [r"風速\s*([0-9.]+)\s*m"], "0"))
    wave_height = parse_float(first_value(text, [r"波高\s*([0-9.]+)\s*cm"], "0"))
    wind_dir = first_value(text, [r"(北東|北西|南東|南西|北|南|東|西)\s*風"], "取得エラー")
    winning_method = first_value(
        text,
        [r"決まり手\s*(逃げ|差し|まくり|まくり差し|抜き|恵まれ)"],
        "取得エラー",
    )

    return {
        "actual_combo":combo,
        "payout_100":payout,
        "weather":weather,
        "wind_dir":wind_dir,
        "wind_speed":wind_speed,
        "wave_height":wave_height,
        "winning_method":winning_method,
    }


def collect_one_race(date8: str, venue: str, code: str, race_no: int) -> tuple[bool, str]:
    url = (
        "https://www.boatrace.jp/owpc/pc/race/raceresult"
        f"?hd={date8}&jcd={code}&rno={race_no}"
    )
    try:
        html = fetch(url, retries=0)
        parsed = parse_historical_result(page_text(html))
        if not parsed:
            return False, "結果なし"

        con = db()
        con.execute("""
          INSERT OR REPLACE INTO historical_results(
            race_date,venue,venue_code,race_no,actual_combo,payout_100,
            weather,wind_dir,wind_speed,wave_height,winning_method,
            source_url,collected_at
          ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.strptime(date8, "%Y%m%d").date().isoformat(),
            venue,code,race_no,
            parsed["actual_combo"],parsed["payout_100"],
            parsed["weather"],parsed["wind_dir"],
            parsed["wind_speed"],parsed["wave_height"],
            parsed["winning_method"],url,datetime.now(JST).isoformat(),
        ))
        con.commit()
        return True, "保存"
    except Exception as exc:
        return False, str(exc)


def historical_results_df() -> pd.DataFrame:
    con = db()
    return pd.read_sql_query(
        "SELECT * FROM historical_results ORDER BY race_date,venue_code,race_no",
        con,
    )


def collection_state() -> dict[str, Any]:
    con = db()
    row = con.execute("""
      SELECT next_date,start_date,end_date,selected_venues,updated_at,
             collected_races,failed_requests
      FROM collection_state WHERE id=1
    """).fetchone()
    keys = [
        "next_date","start_date","end_date","selected_venues","updated_at",
        "collected_races","failed_requests",
    ]
    return dict(zip(keys, row)) if row else {}


def save_collection_state(**kwargs: Any) -> None:
    current = collection_state()
    current.update(kwargs)
    con = db()
    con.execute("""
      UPDATE collection_state SET
        next_date=?,start_date=?,end_date=?,selected_venues=?,updated_at=?,
        collected_races=?,failed_requests=?
      WHERE id=1
    """, (
        current.get("next_date",""),
        current.get("start_date",""),
        current.get("end_date",""),
        current.get("selected_venues","[]"),
        current.get("updated_at",datetime.now(JST).isoformat()),
        int(current.get("collected_races",0)),
        int(current.get("failed_requests",0)),
    ))
    con.commit()


# ---------- HTTP ----------
MAX_RESPONSE_BYTES = 1_500_000  # 1.5MBを超えるページは途中で打ち切る


def fetch(url: str, retries: int = 1) -> str:
    """
    Render安定版:
    - 応答サイズを制限
    - apparent_encodingを使わない
    - 接続/読込タイムアウトを分離
    - 失敗しても呼び出し側で安全に処理
    """
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            with requests.get(
                url,
                headers=HEADERS,
                timeout=(5, 10),
                stream=True,
                allow_redirects=True,
            ) as response:
                response.raise_for_status()

                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_content(chunk_size=32_768):
                    if not chunk:
                        continue
                    remaining = MAX_RESPONSE_BYTES - total
                    if remaining <= 0:
                        break
                    chunks.append(chunk[:remaining])
                    total += min(len(chunk), remaining)

                raw = b"".join(chunks)
                encoding = response.encoding or "utf-8"
                return raw.decode(encoding, errors="replace")

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.8)

    raise RuntimeError(str(last_error))


def page_text(html: str) -> str:
    if not html:
        return ""
    try:
        # Python標準パーサーだけを使い、lxmlによるstatus 139を避ける
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    except Exception:
        # 最終保険。HTMLタグを簡易除去して文字列だけ返す
        return re.sub(r"<[^>]+>", " ", html)


def read_tables(html: str) -> list[pd.DataFrame]:
    """
    V8ではHTMLテーブル解析を行わない。
    pandas.read_html/lxmlがRender上でクラッシュする経路を完全に遮断する。
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
                "html": "",
                "text": page_text(html),
                "tables": [],
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
            "url": url, "html": "", "text": page_text(html),
            "tables": [], "error": None,
        }
    except Exception as exc:
        return {"url":url, "html":"", "text":"", "tables":[], "error":str(exc)}


def get_umepyon(date_iso: str, code: str, race_no: int) -> dict[str, Any]:
    # 検索ではなく公開されている予想ページ形式へ直接アクセス
    url = f"https://umepyon.com/predict.php?jcd={code}&racedate={date_iso}&racenum={race_no}"
    try:
        html = fetch(url)
        return {
            "url":url, "html":"", "text":page_text(html),
            "tables":[], "error":None,
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
        model_config = get_model_config()
        if len(available) == 2:
            model_prob = (
                pose_prob * float(model_config["poseidon_weight"])
                + ume_prob * float(model_config["umepyon_weight"])
            )
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
    model_config = get_model_config()
    ev_rows = [r for r in rows if r["odds"] > 0 and r["ev"] >= float(model_config["ev_threshold"])]

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
        selected = sorted(ev_rows, key=lambda r:(r["ev"], r["model_prob"]), reverse=True)[:int(model_config["max_bets"])]
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

    if not selected:
        scores = {"1":100.0, "2":82.0, "3":73.0, "4":62.0, "5":50.0, "6":40.0}
        selected = []
        for fallback in fallback_combos(scores)[:8]:
            selected.append({
                "combo": fallback["combo"],
                "model_prob": fallback["model_prob"],
                "poseidon_prob": None,
                "ume_prob": None,
                "poseidon_index": 0,
                "odds": 0,
                "market_prob": 0,
                "ev": 0,
                "divergence": 0,
            })
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
st.title("🚤 WDJ ボートレースAI Web版 v14")
st.caption(
    "開催日・場・Rの選択時に公式・梅吉AI・ポセイドンを自動取得し、"
    "WDJ115-v2条件に適合したレースだけ5,000円で仮想購入します。"
)
st.info("V14収集版：過去5年の公式結果を日付・場ごとに収集し、途中保存・再開・CSV出力できます。")

tabs = st.tabs(["予想", "締切3分前監視", "過去収集", "過去学習", "成績ダッシュボード", "結果登録"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    race_date = c1.date_input("開催日", datetime.now(JST).date())
    venue = c2.selectbox("場", list(VENUES))
    race_no = c3.selectbox("R", range(1,13))
    deadline = st.time_input("締切時刻")
    real_purchase = st.checkbox("実購入として記録する", value=False)

    selection_key = f"{race_date}-{venue}-{race_no}"

    def make_fallback_predictions() -> list[dict[str, Any]]:
        base_scores = {"1":100.0, "2":82.0, "3":73.0, "4":62.0, "5":50.0, "6":40.0}
        forced = []
        for rank, row in enumerate(fallback_combos(base_scores)[:8], start=1):
            forced.append({
                "rank": rank,
                "combo": row["combo"],
                "odds": 0,
                "amount": 0,
                "return": 0,
                "profit": 0,
                "model_prob": row["model_prob"],
                "market_prob": 0,
                "ev": 0,
                "divergence": 0,
                "poseidon_prob": None,
                "ume_prob": None,
                "poseidon_index": 0,
            })
        return forced

    def load_latest_prediction() -> dict[str, Any]:
        date8 = race_date.strftime("%Y%m%d")
        date_iso = race_date.strftime("%Y-%m-%d")
        code = VENUES[venue]

        official = get_official(date8, code, race_no)
        poseidon = get_poseidon(date8, code, race_no)
        umepyon = get_umepyon(date_iso, code, race_no)

        try:
            calc = build_prediction(official, poseidon, umepyon)
        except Exception as exc:
            calc = {
                "predictions": make_fallback_predictions(),
                "action": "見送り",
                "classification": "見送り",
                "budget": 0,
                "avg_ev": 0,
                "stars": 1,
                "confidence": 15,
                "eligible": False,
                "meta": {
                    "weather":"取得エラー", "wind_dir":"取得エラー",
                    "wind":"取得エラー", "wave":"取得エラー",
                    "exhibition_rows":[],
                },
                "head_rank":["1","2"],
                "opponent_rank":["2","3","4","5"],
                "reason":f"予想計算エラーのため参考順位を表示：{exc}",
            }

        if not calc.get("predictions"):
            calc["predictions"] = make_fallback_predictions()
            calc["action"] = "見送り"
            calc["eligible"] = False
            calc["budget"] = 0

        return {
            "calc": calc,
            "official": official,
            "poseidon": poseidon,
            "umepyon": umepyon,
            "selection_key": selection_key,
            "loaded_at": datetime.now(JST).strftime("%H:%M:%S"),
        }

    # 開催日・場・Rが変わったら、3サイトを自動取得して最初から最新買い目を表示
    if st.session_state.get("v11_selection_key") != selection_key:
        with st.spinner("公式・梅吉AI・ポセイドンを自動取得して予想中…"):
            st.session_state["v11_result"] = load_latest_prediction()
            st.session_state["v11_selection_key"] = selection_key

    # 手動で最新状態へ更新するボタン
    if st.button("🔄 3サイトを再取得して最新予想に更新", use_container_width=True):
        with st.spinner("3サイトを再取得中…"):
            st.session_state["v11_result"] = load_latest_prediction()
            st.session_state["v11_selection_key"] = selection_key
        st.rerun()

    result = st.session_state.get("v11_result")
    if not result:
        result = {
            "calc": {
                "predictions": make_fallback_predictions(),
                "action":"見送り", "classification":"見送り",
                "budget":0, "avg_ev":0, "stars":1, "confidence":15,
                "eligible":False,
                "meta":{"weather":"取得エラー","wind_dir":"取得エラー","wind":"取得エラー","wave":"取得エラー","exhibition_rows":[]},
                "head_rank":["1","2"], "opponent_rank":["2","3","4","5"],
                "reason":"取得結果がないため参考順位を表示しています。",
            },
            "official":{"errors":["未取得"],"urls":{}},
            "poseidon":{"error":"未取得","url":""},
            "umepyon":{"error":"未取得","url":""},
            "loaded_at":"未取得",
        }

    calc = result["calc"]
    official = result["official"]
    poseidon = result["poseidon"]
    umepyon = result["umepyon"]

    star_text = "★" * calc.get("stars", 1) + "☆" * (5 - calc.get("stars", 1))
    st.success(f"{venue}{race_no}R｜3サイト反映済み｜{calc.get('action', '見送り')}")
    st.caption(f"最終取得時刻：{result.get('loaded_at', '未取得')}")

    a,b,c,d = st.columns(4)
    a.metric("分類", calc.get("classification", "見送り"))
    b.metric("最終判断", calc.get("action", "見送り"))
    c.metric("自信度", star_text)
    d.metric("参考期待値", calc.get("avg_ev", 0))

    active_model = get_model_config()
    st.caption(
        f"学習設定：ポセイドン {active_model['poseidon_weight']:.0%}／"
        f"梅吉AI {active_model['umepyon_weight']:.0%}／"
        f"期待値基準 {active_model['ev_threshold']:.2f}／"
        f"最大 {int(active_model['max_bets'])}点"
    )

    st.subheader("🎯 3サイト反映後の買い目")
    compact_rows = []
    for row in calc["predictions"][:8]:
        odds = float(row.get("odds", 0) or 0)
        amount = int(row.get("amount", 0) or 0)
        payout = int(row.get("return", 0) or 0)
        compact_rows.append({
            "順位": int(row.get("rank", len(compact_rows) + 1)),
            "買い目": row.get("combo", "-"),
            "直前オッズ": f"{odds:.1f}倍" if odds else "未取得",
            "購入額": f"{amount:,}円",
            "的中時払戻": f"{payout:,}円" if payout else "未計算",
            "推定確率": f"{float(row.get('model_prob', 0) or 0):.2f}%",
        })
    st.table(pd.DataFrame(compact_rows))

    if calc.get("action") == "買い":
        st.success(f"購入条件適合：合計{int(calc.get('budget', 0)):,}円で仮想購入します。")
    else:
        st.warning("購入は見送りですが、上の買い目は3サイト取得後のAI予想です。")

    st.subheader("3サイト取得状況")
    s1,s2,s3 = st.columns(3)
    s1.metric("BOAT RACE公式", "OK" if not official.get("errors") else "一部エラー")
    s2.metric("ポセイドン", "OK" if not poseidon.get("error") else "エラー")
    s3.metric("梅吉AI", "OK" if not umepyon.get("error") else "エラー")

    meta = calc.get("meta", {})
    st.subheader("直前情報")
    w1,w2,w3,w4 = st.columns(4)
    w1.metric("天候", meta.get("weather", "取得エラー"))
    w2.metric("風向", meta.get("wind_dir", "取得エラー"))
    w3.metric("風速", f"{meta.get('wind', '取得エラー')}m")
    w4.metric("波高", f"{meta.get('wave', '取得エラー')}cm")

    if meta.get("exhibition_rows"):
        st.dataframe(
            pd.DataFrame(meta["exhibition_rows"]),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("軸と相手順位")
    h1,h2 = st.columns(2)
    h1.write("**頭候補：** " + (" → ".join(calc.get("head_rank", [])) or "取得エラー"))
    h2.write("**相手順位：** " + (" → ".join(calc.get("opponent_rank", [])) or "取得エラー"))

    st.subheader("総合判断理由")
    st.write(calc.get("reason", "3サイトを反映して予想しました。"))

    with st.expander("参照URL・取得エラー"):
        for label,url in official.get("urls", {}).items():
            st.write(f"公式 {label}: {url}")
        st.write(f"ポセイドン: {poseidon.get('url', '')}")
        st.write(f"梅吉AI: {umepyon.get('url', '')}")
        if official.get("errors"):
            st.write("公式エラー:", official["errors"])
        if poseidon.get("error"):
            st.write("ポセイドンエラー:", poseidon["error"])
        if umepyon.get("error"):
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
    st.header("📚 過去5年データ収集")
    st.write(
        "BOAT RACE公式の確定結果を日付・場・レース単位で保存します。"
        "5年分は件数が多いため、一度に全部ではなく小分けで収集し、途中から再開します。"
    )

    today = datetime.now(JST).date()
    five_years_ago = today - timedelta(days=365 * 5)

    state = collection_state()
    default_start = five_years_ago
    if state.get("next_date"):
        try:
            default_start = datetime.fromisoformat(state["next_date"]).date()
        except Exception:
            pass

    d1,d2 = st.columns(2)
    collect_start = d1.date_input(
        "今回の開始日",
        default_start,
        min_value=five_years_ago,
        max_value=today,
        key="collect_start",
    )
    collect_end = d2.date_input(
        "今回の終了日",
        min(collect_start + timedelta(days=2), today),
        min_value=collect_start,
        max_value=today,
        key="collect_end",
    )

    selected_venues = st.multiselect(
        "収集する場",
        list(VENUES),
        default=list(VENUES)[:4],
        help="最初は4場程度で動作確認してください。全24場は複数回に分けるのが安全です。",
    )
    race_limit = st.select_slider(
        "1場あたりの収集レース数",
        options=[1,3,6,12],
        value=12,
    )

    day_count = (collect_end - collect_start).days + 1
    request_estimate = day_count * len(selected_venues) * int(race_limit)
    st.info(
        f"今回の予定：{day_count}日 × {len(selected_venues)}場 × "
        f"{race_limit}R ＝ 最大{request_estimate:,}ページ"
    )

    if request_estimate > 300:
        st.warning(
            "一度の取得数が多いです。Renderで止まりやすいため、"
            "開始日と終了日を1〜3日、場数を4場程度にしてください。"
        )

    start_collect = st.button(
        "この範囲を収集",
        type="primary",
        use_container_width=True,
        disabled=not selected_venues,
    )

    if start_collect:
        progress = st.progress(0)
        status = st.empty()
        total = max(request_estimate, 1)
        done = 0
        saved = 0
        failed = 0
        current_date = collect_start

        while current_date <= collect_end:
            date8 = current_date.strftime("%Y%m%d")
            for venue_name in selected_venues:
                code = VENUES[venue_name]
                for race_no_item in range(1, int(race_limit) + 1):
                    status.write(
                        f"取得中：{current_date} {venue_name}{race_no_item}R"
                    )
                    ok, message = collect_one_race(
                        date8, venue_name, code, race_no_item
                    )
                    done += 1
                    if ok:
                        saved += 1
                    elif message != "結果なし":
                        failed += 1
                    progress.progress(min(done / total, 1.0))
                    time.sleep(0.08)

            next_day = current_date + timedelta(days=1)
            save_collection_state(
                next_date=next_day.isoformat(),
                start_date=collect_start.isoformat(),
                end_date=collect_end.isoformat(),
                selected_venues=json.dumps(selected_venues, ensure_ascii=False),
                updated_at=datetime.now(JST).isoformat(),
                collected_races=int(state.get("collected_races", 0)) + saved,
                failed_requests=int(state.get("failed_requests", 0)) + failed,
            )
            current_date = next_day

        status.success(f"完了：新規・更新 {saved}レース／通信失敗 {failed}件")
        st.session_state["v14_collection_done"] = True

    results = historical_results_df()
    r1,r2,r3,r4 = st.columns(4)
    r1.metric("保存済みレース", f"{len(results):,}")
    r2.metric(
        "保存済み日数",
        f"{results['race_date'].nunique():,}" if not results.empty else "0",
    )
    r3.metric(
        "保存済み場数",
        f"{results['venue'].nunique():,}" if not results.empty else "0",
    )
    r4.metric(
        "次回開始候補",
        collection_state().get("next_date") or "未開始",
    )

    if not results.empty:
        st.download_button(
            "収集済み公式結果CSVをダウンロード",
            results.to_csv(index=False).encode("utf-8-sig"),
            file_name="boatrace_official_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(results.tail(100), hide_index=True, use_container_width=True)

    st.warning(
        "この版で自動収集するのは公式の確定結果です。"
        "梅吉AI・ポセイドンの5年前の予想履歴は公開保存状況が一定でないため、"
        "取得可能性を確認しながら後続版で追加します。"
    )


with tabs[3]:
    st.header("🧠 過去レース学習")
    st.write(
        "1レースにつき複数の買い目候補をCSVで登録し、"
        "ポセイドンと梅吉AIの比重、期待値基準、購入点数を自動最適化します。"
    )

    current = get_model_config()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ポセイドン比重", f"{current['poseidon_weight']:.0%}")
    c2.metric("梅吉AI比重", f"{current['umepyon_weight']:.0%}")
    c3.metric("購入期待値基準", f"{current['ev_threshold']:.2f}")
    c4.metric("最大購入点数", int(current["max_bets"]))
    st.caption(
        f"最終更新：{current['updated_at']}／学習レース {int(current['training_races'])}件／"
        f"検証回収率 {float(current['backtest_roi']):.1f}%"
    )

    sample = pd.DataFrame([
        {
            "race_date":"2026-07-01","venue":"桐生","race_no":1,
            "combo":"1-2-3","odds":12.4,
            "poseidon_prob":11.2,"umepyon_prob":9.8,"fallback_prob":0,
            "actual_combo":"1-2-3","payout_100":1240,
        },
        {
            "race_date":"2026-07-01","venue":"桐生","race_no":1,
            "combo":"1-3-2","odds":18.6,
            "poseidon_prob":8.1,"umepyon_prob":10.5,"fallback_prob":0,
            "actual_combo":"1-2-3","payout_100":1240,
        },
    ])
    st.download_button(
        "学習CSVテンプレートをダウンロード",
        sample.to_csv(index=False).encode("utf-8-sig"),
        file_name="boat_race_training_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader(
        "過去データCSVを選択",
        type=["csv"],
        help="同じレースの候補買い目を複数行で登録してください。",
    )

    training_df = None
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            training_df = normalize_training_df(raw_df)
            st.success(
                f"読込成功：候補 {len(training_df):,}行／"
                f"レース {training_df['race_id'].nunique():,}件"
            )
            st.dataframe(training_df.head(30), hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"CSVを読み込めません：{exc}")

    if training_df is not None and not training_df.empty:
        b1,b2 = st.columns(2)

        if b1.button("現在設定で過去検証", use_container_width=True):
            report = backtest_training(
                training_df,
                float(current["poseidon_weight"]),
                float(current["ev_threshold"]),
                int(current["max_bets"]),
            )
            st.session_state["v13_backtest"] = report

        if b2.button("過去データから自動学習", type="primary", use_container_width=True):
            with st.spinner("重み・期待値基準・購入点数を総当たりで検証中…"):
                try:
                    best, ranking = optimize_training(training_df)
                    config = {
                        **best,
                        "updated_at":datetime.now(JST).isoformat(),
                        "training_rows":len(training_df),
                        "training_races":training_df["race_id"].nunique(),
                        "backtest_roi":best["roi"],
                        "backtest_profit":best["profit"],
                    }
                    save_model_config(config)
                    st.session_state["v13_best"] = best
                    st.session_state["v13_ranking"] = ranking.head(30)
                    st.success("学習結果を保存し、今後の予想へ反映しました。")
                except Exception as exc:
                    st.error(f"学習できません：{exc}")

    if "v13_backtest" in st.session_state:
        report = st.session_state["v13_backtest"]
        st.subheader("現在設定の検証結果")
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("購入レース", report["purchased_races"])
        r2.metric("的中率", f"{report['hit_rate']:.1f}%")
        r3.metric("収支", f"{report['profit']:,}円")
        r4.metric("回収率", f"{report['roi']:.1f}%")
        if not report["details"].empty:
            st.dataframe(report["details"], hide_index=True, use_container_width=True)

    if "v13_best" in st.session_state:
        best = st.session_state["v13_best"]
        st.subheader("最適化された設定")
        x1,x2,x3,x4 = st.columns(4)
        x1.metric("ポセイドン比重", f"{best['poseidon_weight']:.0%}")
        x2.metric("梅吉AI比重", f"{best['umepyon_weight']:.0%}")
        x3.metric("期待値基準", f"{best['ev_threshold']:.2f}")
        x4.metric("最大購入点数", int(best["max_bets"]))
        y1,y2,y3 = st.columns(3)
        y1.metric("検証収支", f"{int(best['profit']):,}円")
        y2.metric("検証回収率", f"{best['roi']:.1f}%")
        y3.metric("購入レース", int(best["purchased_races"]))

        st.subheader("上位設定30件")
        st.dataframe(
            st.session_state["v13_ranking"],
            hide_index=True,
            use_container_width=True,
        )

    st.warning(
        "最初は最低100レース、できれば1,000レース以上で学習してください。"
        "同じ期間だけで最適化すると過学習しやすいため、後半期間で別検証するのが安全です。"
    )


with tabs[4]:
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

with tabs[5]:
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
