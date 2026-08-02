
from __future__ import annotations

import itertools
import json
import math
import re
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


# =========================================================
# 基本設定
# =========================================================
APP_NAME = "WDJ Boat Race AI Web版 V21"
JST = timezone(timedelta(hours=9))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "boat_race_v21.db"

VENUES = {
    "桐生":"01","戸田":"02","江戸川":"03","平和島":"04","多摩川":"05","浜名湖":"06",
    "蒲郡":"07","常滑":"08","津":"09","三国":"10","びわこ":"11","住之江":"12",
    "尼崎":"13","鳴門":"14","丸亀":"15","児島":"16","宮島":"17","徳山":"18",
    "下関":"19","若松":"20","芦屋":"21","福岡":"22","唐津":"23","大村":"24",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151 Safari/537.36"
    )
}
REQUEST_TIMEOUT = (5, 10)
MAX_RESPONSE_BYTES = 1_500_000


# =========================================================
# DB
# =========================================================
def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    con = db()
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS historical_results(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          race_date TEXT NOT NULL,
          venue TEXT NOT NULL,
          venue_code TEXT NOT NULL,
          race_no INTEGER NOT NULL,
          actual_combo TEXT,
          payout_100 INTEGER DEFAULT 0,
          weather TEXT,
          wind_dir TEXT,
          wind_speed REAL,
          wave_height REAL,
          winning_method TEXT,
          source_url TEXT,
          collected_at TEXT,
          UNIQUE(race_date, venue_code, race_no)
        );

        CREATE TABLE IF NOT EXISTS model_config(
          id INTEGER PRIMARY KEY CHECK(id=1),
          poseidon_weight REAL DEFAULT 0.72,
          umepyon_weight REAL DEFAULT 0.28,
          ev_threshold REAL DEFAULT 1.10,
          max_bets INTEGER DEFAULT 8,
          updated_at TEXT DEFAULT '初期値',
          training_rows INTEGER DEFAULT 0,
          training_races INTEGER DEFAULT 0,
          backtest_roi REAL DEFAULT 0,
          backtest_profit INTEGER DEFAULT 0
        );

        INSERT OR IGNORE INTO model_config(
          id,poseidon_weight,umepyon_weight,ev_threshold,max_bets
        ) VALUES(1,0.72,0.28,1.10,8);

        CREATE TABLE IF NOT EXISTS predictions(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          race_date TEXT,
          venue TEXT,
          race_no INTEGER,
          action TEXT,
          classification TEXT,
          confidence REAL,
          expected_value REAL,
          bets_json TEXT,
          sources_json TEXT,
          created_at TEXT
        );
        """
    )
    con.commit()


def get_model_config() -> dict[str, Any]:
    row = db().execute("SELECT * FROM model_config WHERE id=1").fetchone()
    return dict(row) if row else {
        "poseidon_weight":0.72,
        "umepyon_weight":0.28,
        "ev_threshold":1.10,
        "max_bets":8,
        "updated_at":"初期値",
        "training_rows":0,
        "training_races":0,
        "backtest_roi":0.0,
        "backtest_profit":0,
    }


def save_model_config(config: dict[str, Any]) -> None:
    con = db()
    con.execute(
        """
        UPDATE model_config SET
          poseidon_weight=?,umepyon_weight=?,ev_threshold=?,max_bets=?,
          updated_at=?,training_rows=?,training_races=?,
          backtest_roi=?,backtest_profit=?
        WHERE id=1
        """,
        (
            float(config["poseidon_weight"]),
            float(config["umepyon_weight"]),
            float(config["ev_threshold"]),
            int(config["max_bets"]),
            str(config["updated_at"]),
            int(config["training_rows"]),
            int(config["training_races"]),
            float(config["backtest_roi"]),
            int(config["backtest_profit"]),
        ),
    )
    con.commit()


def save_prediction(record: dict[str, Any]) -> None:
    con = db()
    con.execute(
        """
        INSERT INTO predictions(
          race_date,venue,race_no,action,classification,confidence,
          expected_value,bets_json,sources_json,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (
            record["race_date"],
            record["venue"],
            int(record["race_no"]),
            record["action"],
            record["classification"],
            float(record["confidence"]),
            float(record["expected_value"]),
            json.dumps(record["bets"], ensure_ascii=False),
            json.dumps(record["sources"], ensure_ascii=False),
            datetime.now(JST).isoformat(),
        ),
    )
    con.commit()


# =========================================================
# HTTP
# =========================================================
def fetch_text(url: str, retries: int = 1) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                stream=True,
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_content(chunk_size=32768):
                    if not chunk:
                        continue
                    remaining = MAX_RESPONSE_BYTES - total
                    if remaining <= 0:
                        break
                    chunks.append(chunk[:remaining])
                    total += min(len(chunk), remaining)
                raw = b"".join(chunks)
                return raw.decode(response.encoding or "utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.8)
    raise RuntimeError(str(last_error))


def html_to_text(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


# =========================================================
# ライブ取得
# =========================================================
def live_urls(date8: str, date_iso: str, code: str, race_no: int) -> dict[str, str]:
    q = f"rno={race_no}&jcd={code}&hd={date8}"
    base = "https://www.boatrace.jp/owpc/pc/race"
    return {
        "racelist":f"{base}/racelist?{q}",
        "before":f"{base}/beforeinfo?{q}",
        "odds":f"{base}/odds3t?{q}",
        "poseidon":f"https://poseidon-boatrace.net/race/{date8}/{int(code)}/{race_no}R",
        "umepyon":(
            f"https://umepyon.com/predict.php?jcd={code}"
            f"&racedate={date_iso}&racenum={race_no}"
        ),
    }


def fetch_live_sources(date8: str, date_iso: str, code: str, race_no: int) -> dict[str, Any]:
    urls = live_urls(date8, date_iso, code, race_no)
    result: dict[str, Any] = {
        "official":{},
        "official_errors":[],
        "poseidon":"",
        "poseidon_error":None,
        "umepyon":"",
        "umepyon_error":None,
        "urls":urls,
    }

    for key in ("racelist", "before", "odds"):
        try:
            result["official"][key] = html_to_text(fetch_text(urls[key], retries=0))
        except Exception as exc:
            result["official"][key] = ""
            result["official_errors"].append(f"{key}: {exc}")

    try:
        result["poseidon"] = html_to_text(fetch_text(urls["poseidon"], retries=0))
    except Exception as exc:
        result["poseidon_error"] = str(exc)

    try:
        result["umepyon"] = html_to_text(fetch_text(urls["umepyon"], retries=0))
    except Exception as exc:
        result["umepyon_error"] = str(exc)

    return result


# =========================================================
# パーサー
# =========================================================
def parse_odds(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    patterns = [
        r"\b([1-6])[-－\s]([1-6])[-－\s]([1-6])\s+([0-9]+(?:\.[0-9]+)?)\b",
        r"\b([1-6])\s*-\s*([1-6])\s*-\s*([1-6])\s*([0-9]+(?:\.[0-9]+)?)\b",
    ]
    for pattern in patterns:
        for a, b, c, odds in re.findall(pattern, text):
            if len({a, b, c}) == 3:
                result.setdefault(f"{a}-{b}-{c}", float(odds))
    return result


def parse_poseidon(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for combo, prob in re.findall(
        r"\b([1-6]-[1-6]-[1-6])\b.{0,50}?([0-9.]+)\s*%",
        text,
        re.S,
    ):
        if len(set(combo.split("-"))) == 3:
            result[combo] = max(result.get(combo, 0), float(prob))
    return result


def parse_umepyon(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    patterns = [
        r"\b([1-6])[-－>]([1-6])[-－>]([1-6])\b[^%]{0,80}?([0-9.]+)\s*%",
        r"\b([1-6])\s+([1-6])\s+([1-6])\b[^%]{0,60}?([0-9.]+)\s*%",
    ]
    for pattern in patterns:
        for a, b, c, prob in re.findall(pattern, text, re.S):
            if len({a, b, c}) == 3:
                combo = f"{a}-{b}-{c}"
                result[combo] = max(result.get(combo, 0), float(prob))
    return result


def fallback_predictions() -> list[dict[str, Any]]:
    scores = {"1":100.0,"2":82.0,"3":73.0,"4":62.0,"5":50.0,"6":40.0}
    rows: list[dict[str, Any]] = []
    for a, b, c in itertools.permutations(scores, 3):
        raw = scores[a] * 0.54 + scores[b] * 0.30 + scores[c] * 0.16
        rows.append({"combo":f"{a}-{b}-{c}", "raw":raw})
    rows.sort(key=lambda row: row["raw"], reverse=True)
    top = rows[:12]
    total = sum(math.exp((row["raw"] - top[0]["raw"]) / 18) for row in top)
    for row in top:
        row["fallback_prob"] = round(
            math.exp((row["raw"] - top[0]["raw"]) / 18) / total * 100,
            2,
        )
    return top


# =========================================================
# 予想エンジン
# =========================================================
def build_prediction(source_data: dict[str, Any]) -> dict[str, Any]:
    config = get_model_config()

    official_text = " ".join(source_data.get("official", {}).values())
    odds_map = parse_odds(official_text)
    pose_probs = parse_poseidon(source_data.get("poseidon", ""))
    ume_probs = parse_umepyon(source_data.get("umepyon", ""))

    fallback_rows = fallback_predictions()
    fallback_map = {row["combo"]:row["fallback_prob"] for row in fallback_rows}

    combos = set(pose_probs) | set(ume_probs) | set(odds_map)
    if not combos:
        combos = set(fallback_map)

    rows: list[dict[str, Any]] = []
    for combo in combos:
        pose = float(pose_probs.get(combo, 0) or 0)
        ume = float(ume_probs.get(combo, 0) or 0)
        fallback = float(fallback_map.get(combo, 0) or 0)

        if pose > 0 and ume > 0:
            probability = (
                pose * float(config["poseidon_weight"])
                + ume * float(config["umepyon_weight"])
            )
        elif pose > 0:
            probability = pose
        elif ume > 0:
            probability = ume
        else:
            probability = fallback

        odds = float(odds_map.get(combo, 0) or 0)
        ev = probability / 100 * odds if odds else 0

        rows.append(
            {
                "combo":combo,
                "model_prob":round(probability, 2),
                "poseidon_prob":pose or None,
                "umepyon_prob":ume or None,
                "odds":round(odds, 1) if odds else 0,
                "ev":round(ev, 3),
            }
        )

    rows.sort(key=lambda row: (row["model_prob"], row["ev"]), reverse=True)

    eligible_rows = [
        row
        for row in rows
        if row["odds"] > 0 and row["ev"] >= float(config["ev_threshold"])
    ]
    eligible = len(eligible_rows) >= 3

    if eligible:
        selected = sorted(
            eligible_rows,
            key=lambda row: (row["ev"], row["model_prob"]),
            reverse=True,
        )[: int(config["max_bets"])]
    else:
        selected = rows[: int(config["max_bets"])]

    if not selected:
        selected = [
            {
                "combo":row["combo"],
                "model_prob":row["fallback_prob"],
                "poseidon_prob":None,
                "umepyon_prob":None,
                "odds":0,
                "ev":0,
            }
            for row in fallback_rows[:8]
        ]

    budget = 5000 if eligible else 0
    amount = 0
    if budget and selected:
        amount = max(100, (budget // len(selected) // 100) * 100)

    bets: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        payout = int(amount * row["odds"]) if amount and row["odds"] else 0
        bets.append(
            {
                **row,
                "rank":rank,
                "amount":amount if eligible else 0,
                "return":payout,
                "profit":payout - budget if payout else 0,
            }
        )

    source_count = sum(
        [
            bool(source_data.get("official")),
            not bool(source_data.get("poseidon_error")),
            not bool(source_data.get("umepyon_error")),
        ]
    )

    return {
        "action":"買い" if eligible else "見送り",
        "classification":"中波乱" if eligible else "参考予想",
        "confidence":min(90, 20 + source_count * 20 + (20 if eligible else 0)),
        "avg_ev":round(
            sum(float(row.get("ev", 0) or 0) for row in selected)
            / max(len(selected), 1),
            3,
        ),
        "bets":bets,
        "budget":budget,
        "model_config":config,
    }


# =========================================================
# 過去結果収集
# =========================================================
def parse_historical_result(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"3連単\s*([1-6])\s*[-－]\s*([1-6])\s*[-－]\s*([1-6])"
        r"\s*[¥￥]?\s*([0-9,]+)",
        text,
    )
    if not match:
        return None

    def first(patterns: list[str], default: str = "取得エラー") -> str:
        for pattern in patterns:
            found = re.search(pattern, text)
            if found:
                return found.group(1).strip()
        return default

    def as_float(value: str) -> float:
        try:
            return float(re.sub(r"[^0-9.\-]", "", value))
        except Exception:
            return 0.0

    return {
        "actual_combo":"-".join(match.group(i) for i in (1, 2, 3)),
        "payout_100":int(match.group(4).replace(",", "")),
        "weather":first([r"(晴|曇り|雨|雪|霧)"]),
        "wind_dir":first([r"(北東|北西|南東|南西|北|南|東|西)\s*風"]),
        "wind_speed":as_float(first([r"風速\s*([0-9.]+)\s*m"], "0")),
        "wave_height":as_float(first([r"波高\s*([0-9.]+)\s*cm"], "0")),
        "winning_method":first(
            [r"決まり手\s*(逃げ|差し|まくり|まくり差し|抜き|恵まれ)"]
        ),
    }


def collect_historical_result(
    race_date: date,
    venue: str,
    code: str,
    race_no: int,
) -> tuple[bool, str]:
    date8 = race_date.strftime("%Y%m%d")
    url = (
        "https://www.boatrace.jp/owpc/pc/race/raceresult"
        f"?hd={date8}&jcd={code}&rno={race_no}"
    )
    try:
        parsed = parse_historical_result(html_to_text(fetch_text(url, retries=0)))
        if not parsed:
            return False, "結果なし"

        con = db()
        con.execute(
            """
            INSERT OR REPLACE INTO historical_results(
              race_date,venue,venue_code,race_no,actual_combo,payout_100,
              weather,wind_dir,wind_speed,wave_height,winning_method,
              source_url,collected_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                race_date.isoformat(),
                venue,
                code,
                race_no,
                parsed["actual_combo"],
                parsed["payout_100"],
                parsed["weather"],
                parsed["wind_dir"],
                parsed["wind_speed"],
                parsed["wave_height"],
                parsed["winning_method"],
                url,
                datetime.now(JST).isoformat(),
            ),
        )
        con.commit()
        return True, "保存"
    except Exception as exc:
        return False, str(exc)


def historical_results_df() -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM historical_results ORDER BY race_date,venue_code,race_no",
        db(),
    )


# =========================================================
# 学習・検証
# =========================================================
TRAINING_COLUMNS = [
    "race_date","venue","race_no","combo","odds",
    "poseidon_prob","umepyon_prob","fallback_prob",
    "actual_combo","payout_100",
]


def normalize_training_df(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in TRAINING_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("不足列: " + ", ".join(missing))

    df = frame[TRAINING_COLUMNS].copy()
    for column in (
        "race_no","odds","poseidon_prob","umepyon_prob",
        "fallback_prob","payout_100",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in ("race_date","venue","combo","actual_combo"):
        df[column] = df[column].astype(str).str.strip()

    df = df[df["odds"] > 0].copy()
    df["race_id"] = (
        df["race_date"]
        + "|"
        + df["venue"]
        + "|"
        + df["race_no"].astype(int).astype(str)
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


def backtest(
    df: pd.DataFrame,
    poseidon_weight: float,
    ev_threshold: float,
    max_bets: int,
) -> dict[str, Any]:
    investment = 0
    returned = 0
    purchased_races = 0
    hit_races = 0

    for _, race in df.groupby("race_id", sort=False):
        work = race.copy()
        work["model_prob"] = work.apply(
            lambda row: candidate_probability(row, poseidon_weight),
            axis=1,
        )
        work["ev"] = work["model_prob"] / 100 * work["odds"]

        selected = work[work["ev"] >= ev_threshold].sort_values(
            ["ev","model_prob"],
            ascending=False,
        ).head(max_bets)

        if selected.empty:
            continue

        purchased_races += 1
        race_investment = len(selected) * 100
        investment += race_investment

        actual_combo = str(work["actual_combo"].iloc[0])
        payout_100 = int(work["payout_100"].iloc[0])
        hit = actual_combo in set(selected["combo"])

        if hit:
            hit_races += 1
            returned += payout_100

    return {
        "investment":investment,
        "returned":returned,
        "profit":returned - investment,
        "roi":returned / investment * 100 if investment else 0,
        "purchased_races":purchased_races,
        "hit_races":hit_races,
        "hit_rate":hit_races / purchased_races * 100 if purchased_races else 0,
    }


def optimize_training(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    results: list[dict[str, Any]] = []

    for weight_step in range(21):
        poseidon_weight = weight_step / 20
        for ev_step in range(80, 161, 5):
            ev_threshold = ev_step / 100
            for max_bets in (3, 4, 5, 6, 8):
                report = backtest(
                    df,
                    poseidon_weight,
                    ev_threshold,
                    max_bets,
                )
                if report["purchased_races"] < 10:
                    continue
                results.append(
                    {
                        "poseidon_weight":poseidon_weight,
                        "umepyon_weight":1 - poseidon_weight,
                        "ev_threshold":ev_threshold,
                        "max_bets":max_bets,
                        "profit":report["profit"],
                        "roi":report["roi"],
                        "purchased_races":report["purchased_races"],
                        "hit_rate":report["hit_rate"],
                    }
                )

    if not results:
        raise ValueError("学習可能な購入レースが10件未満です。")

    ranking = pd.DataFrame(results).sort_values(
        ["profit","roi","purchased_races"],
        ascending=False,
    ).reset_index(drop=True)

    best = ranking.iloc[0].to_dict()
    save_model_config(
        {
            **best,
            "updated_at":datetime.now(JST).isoformat(),
            "training_rows":len(df),
            "training_races":df["race_id"].nunique(),
            "backtest_roi":best["roi"],
            "backtest_profit":best["profit"],
        }
    )
    return best, ranking


# =========================================================
# UI
# =========================================================
init_db()

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title("🚤 WDJ ボートレースAI Web版 V21")
st.caption(
    "GitHubのWebアップロードで壊れにくい1ファイル構成。"
    "予想・過去収集・学習をこのapp.pyだけで動かします。"
)

tabs = st.tabs(["予想", "過去5年収集", "学習・検証", "データ確認"])


with tabs[0]:
    c1, c2, c3 = st.columns(3)
    race_date = c1.date_input("開催日", datetime.now(JST).date())
    venue = c2.selectbox("場", list(VENUES))
    race_no = c3.selectbox("R", range(1, 13))

    run_prediction = st.button(
        "3サイト取得 → 予想を表示",
        type="primary",
        use_container_width=True,
    )

    if run_prediction:
        with st.spinner("公式・梅吉AI・ポセイドンを取得中…"):
            sources = fetch_live_sources(
                race_date.strftime("%Y%m%d"),
                race_date.strftime("%Y-%m-%d"),
                VENUES[venue],
                race_no,
            )
            prediction = build_prediction(sources)

        st.session_state["v21_prediction"] = {
            "selection":f"{race_date}-{venue}-{race_no}",
            "sources":sources,
            "prediction":prediction,
        }

    result = st.session_state.get("v21_prediction")
    if result and result["selection"] == f"{race_date}-{venue}-{race_no}":
        prediction = result["prediction"]
        sources = result["sources"]

        st.success(f"{venue}{race_no}R｜{prediction['action']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最終判断", prediction["action"])
        m2.metric("分類", prediction["classification"])
        m3.metric("自信度", f"{prediction['confidence']}%")
        m4.metric("参考期待値", prediction["avg_ev"])

        st.subheader("🎯 買い目")
        table = pd.DataFrame(prediction["bets"]).rename(
            columns={
                "rank":"順位",
                "combo":"買い目",
                "odds":"オッズ",
                "amount":"購入額",
                "return":"的中時払戻",
                "profit":"純利益",
                "model_prob":"推定確率",
                "ev":"期待値",
                "poseidon_prob":"ポセイドン確率",
                "umepyon_prob":"梅吉確率",
            }
        )
        columns = [
            "順位","買い目","オッズ","購入額","的中時払戻","純利益",
            "推定確率","期待値","ポセイドン確率","梅吉確率",
        ]
        st.dataframe(
            table[[column for column in columns if column in table.columns]],
            hide_index=True,
            use_container_width=True,
        )

        s1, s2, s3 = st.columns(3)
        s1.metric(
            "BOAT RACE公式",
            "OK" if not sources["official_errors"] else "一部エラー",
        )
        s2.metric(
            "ポセイドン",
            "OK" if not sources["poseidon_error"] else "エラー",
        )
        s3.metric(
            "梅吉AI",
            "OK" if not sources["umepyon_error"] else "エラー",
        )

        save_key = f"{race_date}-{venue}-{race_no}"
        if st.session_state.get("v21_saved_key") != save_key:
            try:
                save_prediction(
                    {
                        "race_date":str(race_date),
                        "venue":venue,
                        "race_no":race_no,
                        "action":prediction["action"],
                        "classification":prediction["classification"],
                        "confidence":prediction["confidence"],
                        "expected_value":prediction["avg_ev"],
                        "bets":prediction["bets"],
                        "sources":sources["urls"],
                    }
                )
                st.session_state["v21_saved_key"] = save_key
            except Exception:
                pass
    else:
        st.info("開催日・場・Rを選び、予想ボタンを押してください。")


with tabs[1]:
    st.header("📚 過去5年公式結果収集")

    today = datetime.now(JST).date()
    five_years_ago = today - timedelta(days=365 * 5)

    d1, d2 = st.columns(2)
    start_date = d1.date_input(
        "開始日",
        five_years_ago,
        min_value=five_years_ago,
        max_value=today,
        key="history_start",
    )
    end_date = d2.date_input(
        "終了日",
        min(start_date + timedelta(days=1), today),
        min_value=start_date,
        max_value=today,
        key="history_end",
    )

    venues = st.multiselect(
        "収集する場",
        list(VENUES),
        default=list(VENUES)[:2],
    )
    race_limit = st.select_slider(
        "1場あたりのレース数",
        options=[1, 3, 6, 12],
        value=12,
    )

    total_requests = (
        (end_date - start_date).days + 1
    ) * len(venues) * int(race_limit)

    st.info(f"今回の最大取得数：{total_requests:,}ページ")

    if total_requests > 200:
        st.error(
            "安全のため200ページを超える収集は実行できません。"
            "日付・場数・レース数を減らしてください。"
        )

    collect_button = st.button(
        "この範囲を収集",
        type="primary",
        use_container_width=True,
        disabled=(not venues) or total_requests > 200,
    )

    if collect_button:
        progress = st.progress(0)
        status = st.empty()
        done = 0
        saved = 0
        failed = 0
        current_date = start_date

        while current_date <= end_date:
            for venue_name in venues:
                code = VENUES[venue_name]
                for race_no_item in range(1, int(race_limit) + 1):
                    status.write(
                        f"取得中：{current_date} {venue_name}{race_no_item}R"
                    )
                    ok, message = collect_historical_result(
                        current_date,
                        venue_name,
                        code,
                        race_no_item,
                    )
                    done += 1
                    if ok:
                        saved += 1
                    elif message != "結果なし":
                        failed += 1
                    progress.progress(
                        min(done / max(total_requests, 1), 1.0)
                    )
                    time.sleep(0.08)
            current_date += timedelta(days=1)

        status.success(
            f"完了：保存 {saved}レース／通信失敗 {failed}件"
        )

    results = historical_results_df()
    q1, q2, q3 = st.columns(3)
    q1.metric("保存済みレース", f"{len(results):,}")
    q2.metric(
        "保存済み日数",
        results["race_date"].nunique() if not results.empty else 0,
    )
    q3.metric(
        "保存済み場数",
        results["venue"].nunique() if not results.empty else 0,
    )

    if not results.empty:
        st.download_button(
            "公式結果CSVをダウンロード",
            results.to_csv(index=False).encode("utf-8-sig"),
            file_name="official_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


with tabs[2]:
    st.header("🧠 学習・バックテスト")
    config = get_model_config()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("ポセイドン比重", f"{float(config['poseidon_weight']):.0%}")
    a2.metric("梅吉AI比重", f"{float(config['umepyon_weight']):.0%}")
    a3.metric("期待値基準", f"{float(config['ev_threshold']):.2f}")
    a4.metric("最大点数", int(config["max_bets"]))

    sample = pd.DataFrame(
        [
            {
                "race_date":"2026-07-01",
                "venue":"桐生",
                "race_no":1,
                "combo":"1-2-3",
                "odds":12.4,
                "poseidon_prob":11.2,
                "umepyon_prob":9.8,
                "fallback_prob":0,
                "actual_combo":"1-2-3",
                "payout_100":1240,
            },
            {
                "race_date":"2026-07-01",
                "venue":"桐生",
                "race_no":1,
                "combo":"1-3-2",
                "odds":18.6,
                "poseidon_prob":8.1,
                "umepyon_prob":10.5,
                "fallback_prob":0,
                "actual_combo":"1-2-3",
                "payout_100":1240,
            },
        ]
    )

    st.download_button(
        "学習CSVテンプレート",
        sample.to_csv(index=False).encode("utf-8-sig"),
        file_name="training_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader("過去予想データCSV", type=["csv"])

    if uploaded is not None:
        try:
            training_df = normalize_training_df(pd.read_csv(uploaded))
            st.success(
                f"読込：{len(training_df):,}行／"
                f"{training_df['race_id'].nunique():,}レース"
            )

            b1, b2 = st.columns(2)

            if b1.button("現在設定で検証", use_container_width=True):
                st.session_state["v21_backtest"] = backtest(
                    training_df,
                    float(config["poseidon_weight"]),
                    float(config["ev_threshold"]),
                    int(config["max_bets"]),
                )

            if b2.button(
                "自動最適化",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("最適条件を検証中…"):
                    best, ranking = optimize_training(training_df)
                st.session_state["v21_best"] = best
                st.session_state["v21_ranking"] = ranking.head(30)
                st.success("学習設定を保存しました。")

        except Exception as exc:
            st.error(f"CSVエラー：{exc}")

    if "v21_backtest" in st.session_state:
        report = st.session_state["v21_backtest"]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("購入レース", report["purchased_races"])
        r2.metric("的中率", f"{report['hit_rate']:.1f}%")
        r3.metric("収支", f"{report['profit']:,}円")
        r4.metric("回収率", f"{report['roi']:.1f}%")

    if "v21_best" in st.session_state:
        st.subheader("最適設定")
        st.json(st.session_state["v21_best"])
        st.dataframe(
            st.session_state["v21_ranking"],
            hide_index=True,
            use_container_width=True,
        )


with tabs[3]:
    st.header("📊 データ確認")
    results = historical_results_df()

    if results.empty:
        st.info("過去結果はまだありません。")
    else:
        st.dataframe(
            results.tail(500),
            hide_index=True,
            use_container_width=True,
        )
