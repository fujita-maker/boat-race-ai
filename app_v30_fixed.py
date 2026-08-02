
from __future__ import annotations

import csv
import html as html_lib
import itertools
import json
import math
import re
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


APP_NAME = "WDJ Boat Race AI Web版 V30 Playwright"
JST = timezone(timedelta(hours=9))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "boat_race_v30.db"

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
CONNECT_TIMEOUT = 4
READ_TIMEOUT = 7
MAX_RESPONSE_BYTES = 900_000


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
          expected_value REAL,
          bets_json TEXT,
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
        "backtest_roi":0.0,
        "backtest_profit":0,
    }


def save_model_config(config: dict[str, Any]) -> None:
    con = db()
    con.execute(
        """
        UPDATE model_config SET
          poseidon_weight=?,umepyon_weight=?,ev_threshold=?,max_bets=?,
          updated_at=?,backtest_roi=?,backtest_profit=?
        WHERE id=1
        """,
        (
            float(config["poseidon_weight"]),
            float(config["umepyon_weight"]),
            float(config["ev_threshold"]),
            int(config["max_bets"]),
            str(config["updated_at"]),
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
          race_date,venue,race_no,action,expected_value,bets_json,created_at
        ) VALUES(?,?,?,?,?,?,?)
        """,
        (
            record["race_date"],
            record["venue"],
            int(record["race_no"]),
            record["action"],
            float(record["expected_value"]),
            json.dumps(record["bets"], ensure_ascii=False),
            datetime.now(JST).isoformat(),
        ),
    )
    con.commit()


def fetch_text(url: str) -> str:
    with requests.get(
        url,
        headers=HEADERS,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        stream=True,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=16384):
            if not chunk:
                continue
            remaining = MAX_RESPONSE_BYTES - total
            if remaining <= 0:
                break
            chunks.append(chunk[:remaining])
            total += min(len(chunk), remaining)
        raw = b"".join(chunks)
        return raw.decode(response.encoding or "utf-8", errors="replace")


def html_to_text(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()



def browser_fetch_pages(urls: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Chromiumで公式ページを表示し、描画後の本文を取得する。"""
    pages: dict[str, str] = {}
    errors: list[str] = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--disable-background-networking",
                    "--disable-extensions",
                    "--disable-sync",
                    "--metrics-recording-only",
                    "--mute-audio",
                    "--no-first-run",
                ],
            )
            context = browser.new_context(
                viewport={"width": 1440, "height": 1100},
                locale="ja-JP",
                user_agent=HEADERS["User-Agent"],
                service_workers="block",
            )

            def block_heavy(route: Any) -> None:
                resource_type = route.request.resource_type
                if resource_type in {"image", "media", "font"}:
                    route.abort()
                else:
                    route.continue_()

            context.route("**/*", block_heavy)

            for key, url in urls.items():
                page = context.new_page()
                try:
                    response = page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=12_000,
                    )
                    page.wait_for_timeout(1_500)
                    status = response.status if response else 0
                    body_text = page.locator("body").inner_text(timeout=4_000)
                    pages[key] = re.sub(r"\s+", " ", body_text).strip()
                    if status >= 400:
                        errors.append(f"{key}: HTTP {status}")
                    if not pages[key]:
                        errors.append(f"{key}: 本文なし")
                except PlaywrightTimeoutError:
                    errors.append(f"{key}: Chromiumタイムアウト")
                    pages[key] = ""
                except Exception as exc:
                    errors.append(f"{key}: {exc}")
                    pages[key] = ""
                finally:
                    page.close()

            context.close()
            browser.close()
    except Exception as exc:
        errors.append(f"Chromium起動: {exc}")

    return pages, errors


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


def safe_fetch(url: str) -> tuple[str, str | None]:
    try:
        return html_to_text(fetch_text(url)), None
    except Exception as exc:
        return "", str(exc)


def fetch_live_sources(date8: str, date_iso: str, code: str, race_no: int) -> dict[str, Any]:
    urls = live_urls(date8, date_iso, code, race_no)

    official_urls = {
        "racelist":urls["racelist"],
        "before":urls["before"],
        "odds":urls["odds"],
    }
    official, official_errors = browser_fetch_pages(official_urls)

    poseidon, poseidon_error = safe_fetch(urls["poseidon"])
    umepyon, umepyon_error = safe_fetch(urls["umepyon"])

    return {
        "official":official,
        "official_errors":official_errors,
        "poseidon":poseidon,
        "poseidon_error":poseidon_error,
        "umepyon":umepyon,
        "umepyon_error":umepyon_error,
        "urls":urls,
    }

def parse_odds(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for pattern in (
        r"\b([1-6])[-－\s]([1-6])[-－\s]([1-6])\s+([0-9]+(?:\.[0-9]+)?)\b",
        r"\b([1-6])\s*-\s*([1-6])\s*-\s*([1-6])\s*([0-9]+(?:\.[0-9]+)?)\b",
    ):
        for a, b, c, odds in re.findall(pattern, text):
            if len({a, b, c}) == 3:
                result.setdefault(f"{a}-{b}-{c}", float(odds))
    return result


def parse_site_probabilities(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for combo, prob in re.findall(
        r"\b([1-6]-[1-6]-[1-6])\b.{0,60}?([0-9.]+)\s*%",
        text,
        re.S,
    ):
        if len(set(combo.split("-"))) == 3:
            result[combo] = max(result.get(combo, 0), float(prob))

    for a, b, c, prob in re.findall(
        r"\b([1-6])[-－>]([1-6])[-－>]([1-6])\b[^%]{0,80}?([0-9.]+)\s*%",
        text,
        re.S,
    ):
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
    denominator = sum(math.exp((row["raw"] - top[0]["raw"]) / 18) for row in top)
    for row in top:
        row["fallback_prob"] = round(
            math.exp((row["raw"] - top[0]["raw"]) / 18) / denominator * 100,
            2,
        )
    return top


def build_prediction(source_data: dict[str, Any]) -> dict[str, Any]:
    config = get_model_config()
    official_text = " ".join(source_data.get("official", {}).values())

    odds_map = parse_odds(official_text)
    pose_probs = parse_site_probabilities(source_data.get("poseidon", ""))
    ume_probs = parse_site_probabilities(source_data.get("umepyon", ""))

    fallback_rows = fallback_predictions()
    fallback_map = {row["combo"]:row["fallback_prob"] for row in fallback_rows}

    combos = set(odds_map) | set(pose_probs) | set(ume_probs)
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

        rows.append({
            "combo":combo,
            "model_prob":round(probability, 2),
            "poseidon_prob":pose or None,
            "umepyon_prob":ume or None,
            "odds":round(odds, 1) if odds else 0,
            "ev":round(ev, 3),
        })

    rows.sort(key=lambda row: (row["model_prob"], row["ev"]), reverse=True)

    eligible_rows = [
        row for row in rows
        if row["odds"] > 0 and row["ev"] >= float(config["ev_threshold"])
    ]
    eligible = len(eligible_rows) >= 3

    if eligible:
        selected = sorted(
            eligible_rows,
            key=lambda row: (row["ev"], row["model_prob"]),
            reverse=True,
        )[:int(config["max_bets"])]
    else:
        selected = rows[:int(config["max_bets"])]

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
    amount = max(100, (budget // len(selected) // 100) * 100) if budget and selected else 0

    bets: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        payout = int(amount * row["odds"]) if amount and row["odds"] else 0
        bets.append({
            "順位":rank,
            "買い目":row["combo"],
            "オッズ":row["odds"] if row["odds"] else "未取得",
            "購入額":amount if eligible else 0,
            "的中時払戻":payout if payout else "未計算",
            "推定確率":f"{row['model_prob']:.2f}%",
            "期待値":row["ev"],
            "ポセイドン確率":(
                f"{row['poseidon_prob']:.2f}%"
                if row["poseidon_prob"] is not None else "-"
            ),
            "梅吉確率":(
                f"{row['umepyon_prob']:.2f}%"
                if row["umepyon_prob"] is not None else "-"
            ),
        })

    source_count = sum([
        bool(source_data.get("official")),
        not bool(source_data.get("poseidon_error")),
        not bool(source_data.get("umepyon_error")),
    ])

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
    }


def parse_historical_result(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"3連単\s*([1-6])\s*[-－]\s*([1-6])\s*[-－]\s*([1-6])"
        r"\s*[¥￥]?\s*([0-9,]+)",
        text,
    )
    if not match:
        return None
    return {
        "actual_combo":"-".join(match.group(i) for i in (1, 2, 3)),
        "payout_100":int(match.group(4).replace(",", "")),
    }


def collect_historical_result(
    race_date: date,
    venue: str,
    code: str,
    race_no: int,
) -> tuple[bool, str]:
    url = (
        "https://www.boatrace.jp/owpc/pc/race/raceresult"
        f"?hd={race_date.strftime('%Y%m%d')}&jcd={code}&rno={race_no}"
    )
    try:
        parsed = parse_historical_result(html_to_text(fetch_text(url)))
        if not parsed:
            return False, "結果なし"
        con = db()
        con.execute(
            """
            INSERT OR REPLACE INTO historical_results(
              race_date,venue,venue_code,race_no,actual_combo,payout_100,
              source_url,collected_at
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                race_date.isoformat(),
                venue,
                code,
                race_no,
                parsed["actual_combo"],
                parsed["payout_100"],
                url,
                datetime.now(JST).isoformat(),
            ),
        )
        con.commit()
        return True, "保存"
    except Exception as exc:
        return False, str(exc)


def historical_results_rows(limit: int = 500) -> list[dict[str, Any]]:
    rows = db().execute(
        """
        SELECT race_date,venue,race_no,actual_combo,payout_100
        FROM historical_results
        ORDER BY race_date DESC,venue_code,race_no
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def training_template_csv() -> bytes:
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "race_date","venue","race_no","combo","odds",
        "poseidon_prob","umepyon_prob","fallback_prob",
        "actual_combo","payout_100",
    ])
    writer.writerow([
        "2026-07-01","桐生",1,"1-2-3",12.4,
        11.2,9.8,0,"1-2-3",1240,
    ])
    writer.writerow([
        "2026-07-01","桐生",1,"1-3-2",18.6,
        8.1,10.5,0,"1-2-3",1240,
    ])
    return output.getvalue().encode("utf-8-sig")


def parse_training_csv(uploaded_file: Any) -> list[dict[str, Any]]:
    text = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(StringIO(text))
    required = {
        "race_date","venue","race_no","combo","odds",
        "poseidon_prob","umepyon_prob","fallback_prob",
        "actual_combo","payout_100",
    }
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        missing = sorted(required - set(reader.fieldnames or []))
        raise ValueError("不足列: " + ", ".join(missing))

    rows: list[dict[str, Any]] = []
    for raw in reader:
        try:
            rows.append({
                "race_date":str(raw["race_date"]).strip(),
                "venue":str(raw["venue"]).strip(),
                "race_no":int(float(raw["race_no"] or 0)),
                "combo":str(raw["combo"]).strip(),
                "odds":float(raw["odds"] or 0),
                "poseidon_prob":float(raw["poseidon_prob"] or 0),
                "umepyon_prob":float(raw["umepyon_prob"] or 0),
                "fallback_prob":float(raw["fallback_prob"] or 0),
                "actual_combo":str(raw["actual_combo"]).strip(),
                "payout_100":int(float(raw["payout_100"] or 0)),
            })
        except Exception:
            continue
    return [row for row in rows if row["odds"] > 0]


def backtest_rows(
    rows: list[dict[str, Any]],
    poseidon_weight: float,
    ev_threshold: float,
    max_bets: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        race_id = f"{row['race_date']}|{row['venue']}|{row['race_no']}"
        grouped.setdefault(race_id, []).append(row)

    investment = returned = purchased = hits = 0

    for race_rows in grouped.values():
        candidates = []
        for row in race_rows:
            pose = row["poseidon_prob"]
            ume = row["umepyon_prob"]
            fallback = row["fallback_prob"]
            if pose > 0 and ume > 0:
                probability = pose * poseidon_weight + ume * (1 - poseidon_weight)
            else:
                probability = pose or ume or fallback
            ev = probability / 100 * row["odds"]
            candidates.append({**row, "probability":probability, "ev":ev})

        selected = sorted(
            [row for row in candidates if row["ev"] >= ev_threshold],
            key=lambda row: (row["ev"], row["probability"]),
            reverse=True,
        )[:max_bets]

        if not selected:
            continue

        purchased += 1
        investment += len(selected) * 100
        actual_combo = race_rows[0]["actual_combo"]
        payout_100 = race_rows[0]["payout_100"]

        if any(row["combo"] == actual_combo for row in selected):
            hits += 1
            returned += payout_100

    return {
        "購入レース":purchased,
        "的中率":round(hits / purchased * 100, 1) if purchased else 0,
        "投資":investment,
        "回収":returned,
        "収支":returned - investment,
        "回収率":round(returned / investment * 100, 1) if investment else 0,
    }


def optimize_training(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranking: list[dict[str, Any]] = []

    for weight_step in range(0, 21, 2):
        poseidon_weight = weight_step / 20
        for ev_step in range(90, 151, 10):
            ev_threshold = ev_step / 100
            for max_bets in (3, 5, 8):
                report = backtest_rows(
                    rows,
                    poseidon_weight,
                    ev_threshold,
                    max_bets,
                )
                if report["購入レース"] < 10:
                    continue
                ranking.append({
                    "ポセイドン比重":poseidon_weight,
                    "梅吉AI比重":1 - poseidon_weight,
                    "期待値基準":ev_threshold,
                    "最大点数":max_bets,
                    **report,
                })

    if not ranking:
        raise ValueError("学習可能な購入レースが10件未満です。")

    ranking.sort(
        key=lambda row: (row["収支"], row["回収率"], row["購入レース"]),
        reverse=True,
    )
    best = ranking[0]

    save_model_config({
        "poseidon_weight":best["ポセイドン比重"],
        "umepyon_weight":best["梅吉AI比重"],
        "ev_threshold":best["期待値基準"],
        "max_bets":best["最大点数"],
        "updated_at":datetime.now(JST).isoformat(),
        "backtest_roi":best["回収率"],
        "backtest_profit":best["収支"],
    })
    return best, ranking[:30]


init_db()
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title("🚤 WDJ ボートレースAI Web版 V30 Playwright")
st.caption(
    "BOAT RACE公式はChromiumで表示後の画面を取得し、梅吉AI・ポセイドンは軽量HTTPで取得します。"
)

tabs = st.tabs(["予想", "過去5年収集", "学習・検証", "データ確認"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    race_date = c1.date_input("開催日", datetime.now(JST).date())
    venue = c2.selectbox("場", list(VENUES))
    race_no = c3.selectbox("R", range(1, 13))

    if st.button("3サイト取得 → 予想を表示", type="primary", use_container_width=True):
        with st.spinner("3サイトを取得中…"):
            sources = fetch_live_sources(
                race_date.strftime("%Y%m%d"),
                race_date.strftime("%Y-%m-%d"),
                VENUES[venue],
                race_no,
            )
            prediction = build_prediction(sources)

        st.session_state["v30_result"] = {
            "selection":f"{race_date}-{venue}-{race_no}",
            "sources":sources,
            "prediction":prediction,
        }

    result = st.session_state.get("v30_result")
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
        st.table(prediction["bets"])

        official_text = " ".join(sources.get("official", {}).values())
        official_odds_count = len(parse_odds(official_text))
        poseidon_count = len(parse_site_probabilities(sources.get("poseidon", "")))
        umepyon_count = len(parse_site_probabilities(sources.get("umepyon", "")))

        s1, s2, s3 = st.columns(3)
        if official_odds_count:
            s1.metric("BOAT RACE公式", f"オッズ取得 {official_odds_count}通り")
        elif any(sources.get("official", {}).values()):
            s1.metric("BOAT RACE公式", "ページ取得・オッズ未公開")
        else:
            s1.metric("BOAT RACE公式", "取得失敗")

        s2.metric(
            "ポセイドン",
            f"予想解析 {poseidon_count}点"
            if poseidon_count else
            ("取得失敗" if sources["poseidon_error"] else "解析0点"),
        )
        s3.metric(
            "梅吉AI",
            f"予想解析 {umepyon_count}点"
            if umepyon_count else
            ("取得失敗" if sources["umepyon_error"] else "解析0点"),
        )

        if official_odds_count == 0:
            st.info(
                "公式ページは取得できても、オッズが発売前・未公開の場合は"
                "期待値、購入額、払戻予測は計算しません。"
            )

        with st.expander("取得エラーの詳細"):
            if sources["official_errors"]:
                st.write("BOAT RACE公式:", sources["official_errors"])
            if sources["poseidon_error"]:
                st.write("ポセイドン:", sources["poseidon_error"])
            if sources["umepyon_error"]:
                st.write("梅吉AI:", sources["umepyon_error"])

        save_key = result["selection"]
        if st.session_state.get("v30_saved_key") != save_key:
            try:
                save_prediction({
                    "race_date":str(race_date),
                    "venue":venue,
                    "race_no":race_no,
                    "action":prediction["action"],
                    "expected_value":prediction["avg_ev"],
                    "bets":prediction["bets"],
                })
                st.session_state["v30_saved_key"] = save_key
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
        key="v30_start",
    )
    end_date = d2.date_input(
        "終了日",
        min(start_date + timedelta(days=1), today),
        min_value=start_date,
        max_value=today,
        key="v30_end",
    )

    venues = st.multiselect("収集する場", list(VENUES), default=list(VENUES)[:2])
    race_limit = st.select_slider(
        "1場あたりのレース数",
        options=[1, 3, 6, 12],
        value=12,
    )

    total_requests = ((end_date - start_date).days + 1) * len(venues) * int(race_limit)
    st.info(f"今回の最大取得数：{total_requests:,}ページ")

    if total_requests > 100:
        st.error(
            "安定運用のため100ページ以下にしてください。"
            "日付・場数・レース数を減らしてください。"
        )

    if st.button(
        "この範囲を収集",
        type="primary",
        use_container_width=True,
        disabled=(not venues) or total_requests > 100,
    ):
        progress = st.progress(0)
        status = st.empty()
        done = saved = failed = 0
        current_date = start_date

        while current_date <= end_date:
            for venue_name in venues:
                for rno in range(1, int(race_limit) + 1):
                    status.write(f"取得中：{current_date} {venue_name}{rno}R")
                    ok, message = collect_historical_result(
                        current_date,
                        venue_name,
                        VENUES[venue_name],
                        rno,
                    )
                    done += 1
                    if ok:
                        saved += 1
                    elif message != "結果なし":
                        failed += 1
                    progress.progress(min(done / max(total_requests, 1), 1.0))
                    time.sleep(0.05)
            current_date += timedelta(days=1)

        status.success(f"完了：保存 {saved}レース／通信失敗 {failed}件")

    saved_count = db().execute("SELECT COUNT(*) FROM historical_results").fetchone()[0]
    st.metric("保存済みレース", f"{saved_count:,}")

with tabs[2]:
    st.header("🧠 学習・バックテスト")
    config = get_model_config()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("ポセイドン比重", f"{float(config['poseidon_weight']):.0%}")
    a2.metric("梅吉AI比重", f"{float(config['umepyon_weight']):.0%}")
    a3.metric("期待値基準", f"{float(config['ev_threshold']):.2f}")
    a4.metric("最大点数", int(config["max_bets"]))

    st.download_button(
        "学習CSVテンプレート",
        training_template_csv(),
        file_name="training_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader("過去予想データCSV", type=["csv"])

    if uploaded is not None:
        try:
            training_rows = parse_training_csv(uploaded)
            race_count = len({
                f"{row['race_date']}|{row['venue']}|{row['race_no']}"
                for row in training_rows
            })
            st.success(f"読込：{len(training_rows):,}行／{race_count:,}レース")

            b1, b2 = st.columns(2)
            if b1.button("現在設定で検証", use_container_width=True):
                st.session_state["v30_backtest"] = backtest_rows(
                    training_rows,
                    float(config["poseidon_weight"]),
                    float(config["ev_threshold"]),
                    int(config["max_bets"]),
                )

            if b2.button("自動最適化", type="primary", use_container_width=True):
                with st.spinner("最適条件を検証中…"):
                    best, ranking = optimize_training(training_rows)
                st.session_state["v30_best"] = best
                st.session_state["v30_ranking"] = ranking
                st.success("学習設定を保存しました。")
        except Exception as exc:
            st.error(f"CSVエラー：{exc}")

    if "v30_backtest" in st.session_state:
        st.subheader("検証結果")
        st.json(st.session_state["v30_backtest"])

    if "v30_best" in st.session_state:
        st.subheader("最適設定")
        st.json(st.session_state["v30_best"])
        st.table(st.session_state["v30_ranking"])

with tabs[3]:
    st.header("📊 データ確認")
    rows = historical_results_rows()
    if rows:
        st.table(rows)
    else:
        st.info("過去結果はまだありません。")
