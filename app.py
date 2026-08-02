
from __future__ import annotations

import csv
import html as html_lib
import itertools
import json
import math
import os
import re
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import requests
import streamlit as st


APP_NAME = "WDJ Boat Race AI Web版 V30.4 VERIFIED"
JST = timezone(timedelta(hours=9))
PAYBACK_RATE = 0.75  # 3連単の平均払戻率(控除率25%). edge計測の基準に使用.

BASE_DIR = Path(__file__).resolve().parent
# APP_DATA_DIR: Renderの永続ディスクのマウント先(例 /var/data)を環境変数で指定できる。
# 未指定ならリポジトリ内 data/(Renderでは再起動で消える)にフォールバック。
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "boat_race_v30_1.db"

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
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}
CONNECT_TIMEOUT = 8
READ_TIMEOUT = 20  # boatrace.jpは応答が遅く7秒では読み取りタイムアウトしていたため延長
MAX_RESPONSE_BYTES = 1_400_000


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
    """公式ページをHTTP(requests)で取得し、本文テキストを返す。

    以前はPlaywright Chromiumで描画取得していたが、Renderの少メモリ環境で
    Chromiumが落ちて全ページ空(=取得失敗)になっていた。boatrace.jpは
    サーバーサイドで出走表・オッズを描画するため、requestsで同じ本文が取れる。
    (過去結果収集も同じサイトをrequestsで取得している)
    """
    pages: dict[str, str] = {}
    errors: list[str] = []

    for key, url in urls.items():
        text, error = safe_fetch(url)
        pages[key] = text
        if error:
            errors.append(f"{key}: {error}")
        elif not text:
            errors.append(f"{key}: 本文なし")

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


def safe_fetch(url: str, retries: int = 1) -> tuple[str, str | None]:
    """取得してテキスト化。タイムアウト等の一時的失敗に備え1回だけ再試行する。"""
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            return html_to_text(fetch_text(url)), None
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(1.0)
    return "", last_error


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

    # 梅吉AIなど、艇番をスペース区切りで並べるサイト向け ("1 2 3 7.95%")。
    # 直後に確率(%)が続く3艇のみを対象にし、誤検出を抑える。
    for a, b, c, prob in re.findall(
        r"\b([1-6])\s+([1-6])\s+([1-6])\s+([0-9]{1,3}(?:\.[0-9]+)?)\s*%",
        text,
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


# ============================================================
# 実データによる edge(優位性) 計測 と 資金戦略の検証 (V30.3)
# 予想は押すたびに predictions に保存済み。結果を突き合わせて「本当に当たっているか」を測る。
# ============================================================
def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace("%", "").replace(",", "").strip())
    except Exception:
        return None


def parse_bets_json(bets_json: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(bets_json)
    except Exception:
        return []
    out = []
    for e in data:
        combo = str(e.get("買い目", "")).strip()
        if not combo:
            continue
        out.append({
            "combo": combo,
            "odds": _to_float(e.get("オッズ")),
            "model_prob": _to_float(e.get("推定確率")),
        })
    return out


def predictions_missing_results() -> list[dict[str, Any]]:
    con = db()
    preds = con.execute(
        "SELECT DISTINCT race_date, venue, race_no FROM predictions"
    ).fetchall()
    have = {
        (str(r["race_date"])[:10], r["venue"], int(r["race_no"]))
        for r in con.execute(
            "SELECT race_date, venue, race_no FROM historical_results"
        ).fetchall()
    }
    return [
        dict(r) for r in preds
        if (str(r["race_date"])[:10], r["venue"], int(r["race_no"])) not in have
    ]


def auto_collect_results(max_races: int = 300) -> tuple[int, int]:
    """予想済みだが結果未取得のレースを、公式から自動収集する。"""
    saved = failed = 0
    today = datetime.now(JST).date()
    for m in predictions_missing_results()[:max_races]:
        try:
            rd = date.fromisoformat(str(m["race_date"])[:10])
        except Exception:
            continue
        if rd > today:
            continue  # まだ結果が出ていない未来のレース
        code = VENUES.get(m["venue"])
        if not code:
            continue
        ok, _ = collect_historical_result(rd, m["venue"], code, int(m["race_no"]))
        if ok:
            saved += 1
        else:
            failed += 1
        time.sleep(0.05)
    return saved, failed


def load_joined_races() -> list[dict[str, Any]]:
    """記録済み予想と実結果を突き合わせ、オッズ付きの買い目があるレースだけ返す。"""
    con = db()
    res = {
        (str(r["race_date"])[:10], r["venue"], int(r["race_no"])): (
            r["actual_combo"], r["payout_100"] or 0)
        for r in con.execute(
            "SELECT race_date, venue, race_no, actual_combo, payout_100 "
            "FROM historical_results"
        ).fetchall()
    }
    joined: list[dict[str, Any]] = []
    seen: set = set()
    for p in con.execute(
        "SELECT race_date, venue, race_no, bets_json FROM predictions "
        "ORDER BY race_date, race_no"
    ).fetchall():
        key = (str(p["race_date"])[:10], p["venue"], int(p["race_no"]))
        if key in seen or key not in res:
            continue
        seen.add(key)
        bets = [b for b in parse_bets_json(p["bets_json"]) if b["odds"]]
        if not bets:
            continue  # オッズ未取得(発売前)の予想はROI計算不可のため除外
        actual, payout = res[key]
        joined.append({
            "key": "｜".join(map(str, key)),
            "bets": bets,
            "actual": actual,
            "payout": payout,
        })
    return joined


def realdata_metrics(joined: list[dict[str, Any]]) -> dict[str, Any]:
    """実データの的中率・回収率・edge(優位性)を算出。¥100/点の均等買いを基準。"""
    n = len(joined)
    hits = invest = ret = 0
    implied_sum = 0.0
    for j in joined:
        combos = [b["combo"] for b in j["bets"]]
        invest += 100 * len(combos)
        # オッズが示す市場の的中確率(=払戻率/オッズ)の合計 → edgeの分母
        for b in j["bets"]:
            if b["odds"]:
                implied_sum += PAYBACK_RATE / b["odds"]
        if j["actual"] in combos:
            hits += 1
            ret += j["payout"]
    roi = (ret / invest - 1) if invest else 0.0
    hit_rate = (hits / n) if n else 0.0
    # edge = 実際の的中数 / 市場が期待する的中数. 1.0超なら優位性あり(要十分なサンプル)
    edge = (hits / implied_sum) if implied_sum > 0 else 0.0
    return {
        "races": n, "hits": hits, "hit_rate": hit_rate, "roi": roi,
        "invest": invest, "return": ret, "profit": ret - invest, "edge": edge,
    }


def replay_staking(joined: list[dict[str, Any]], init: float = 300000.0) -> list[dict[str, Any]]:
    """実際に起きたレース系列に各掛け金戦略を当てはめ、最終資金を実データで再現。"""
    def run(name: str):
        B = init; peak = B; mdd = 0.0; mart = 1000.0; ruined = False
        staked = 0.0; returned = 0.0
        for j in joined:
            k = len(j["bets"])
            if name == "flat_100_per_pt":
                S = 100.0 * k
            elif name == "flat_1000":
                S = 1000.0
            elif name == "frac_2pct":
                S = 0.02 * B
            elif name == "martingale":
                S = mart
            else:
                S = 1000.0
            S = min(S, B)
            if S < 1 or ruined:
                continue
            hit = j["actual"] in [b["combo"] for b in j["bets"]]
            delta = (S / k) * (j["payout"] / 100.0) - S if hit else -S
            B += delta
            staked += S
            returned += (S / k) * (j["payout"] / 100.0) if hit else 0.0
            if name == "martingale":
                mart = min(mart * 2, max(B, 100.0)) if not hit else 1000.0
            peak = max(peak, B)
            mdd = max(mdd, (peak - B) / peak if peak > 0 else 0)
            if B < 100:
                ruined = True
        roi = (returned / staked - 1) if staked else 0.0
        return {"戦略": name, "最終資金": round(B), "回収率": round(roi * 100, 1),
                "最大DD": round(mdd * 100, 1), "破産": ruined}
    labels = {"flat_100_per_pt": "¥100/点 均等", "flat_1000": "¥1000/レース固定",
              "frac_2pct": "資金の2%/レース", "martingale": "倍賭け(連敗で倍)"}
    out = []
    for key in ["flat_100_per_pt", "flat_1000", "frac_2pct", "martingale"]:
        r = run(key); r["戦略"] = labels[key]; out.append(r)
    return out


init_db()
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title("🚤 WDJ ボートレースAI Web版 V30.4 VERIFIED")
st.caption(
    "BUILD: V30.4-VERIFIED-20260803｜公式取得のタイムアウト延長＋再試行。予想の永続記録・実データでのedge/資金戦略検証。"
)

tabs = st.tabs(["予想", "過去5年収集", "学習・検証", "データ確認", "収支・edge検証"])

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

        if official_odds_count > 0:
            s1.metric("BOAT RACE公式", f"オッズ取得 {official_odds_count}通り")
        elif any(sources.get("official", {}).values()):
            s1.metric("BOAT RACE公式", "ページ取得・オッズ未公開")
        else:
            s1.metric("BOAT RACE公式", "取得失敗")

        if poseidon_count > 0:
            s2.metric("ポセイドン", f"予想解析 {poseidon_count}点")
        elif sources.get("poseidon_error"):
            s2.metric("ポセイドン", "取得失敗")
        else:
            s2.metric("ポセイドン", "解析0点")

        if umepyon_count > 0:
            s3.metric("梅吉AI", f"予想解析 {umepyon_count}点")
        elif sources.get("umepyon_error"):
            s3.metric("梅吉AI", "取得失敗")
        else:
            s3.metric("梅吉AI", "解析0点")

        if official_odds_count == 0:
            st.info(
                "公式ページは取得できても、オッズが発売前・未公開の場合は、"
                "期待値・購入額・払戻予測は計算しません。"
            )

        with st.expander("取得状況の詳細"):
            st.write("公式ページ取得:", {
                key: bool(value)
                for key, value in sources.get("official", {}).items()
            })
            if sources.get("official_errors"):
                st.write("BOAT RACE公式エラー:", sources["official_errors"])
            if sources.get("poseidon_error"):
                st.write("ポセイドンエラー:", sources["poseidon_error"])
            if sources.get("umepyon_error"):
                st.write("梅吉AIエラー:", sources["umepyon_error"])

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

with tabs[4]:
    st.header("💰 収支・edge検証（あなたの実データ）")
    st.caption(
        "「予想」タブで予想を出すたびに自動記録されます。ここでレース結果を突き合わせ、"
        "本当に的中優位性(edge)があるか、どの掛け金戦略が資金を増やすかを実データで測ります。"
    )

    pcount = db().execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    rcount = db().execute("SELECT COUNT(*) FROM historical_results").fetchone()[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("記録済み予想", f"{pcount:,}")
    m2.metric("収集済み結果", f"{rcount:,}")
    missing = len(predictions_missing_results())
    m3.metric("結果待ちの予想", f"{missing:,}")

    if st.button("予想済みレースの結果を自動収集", type="primary", use_container_width=True):
        with st.spinner("公式から結果を収集中…"):
            saved, failed = auto_collect_results()
        st.success(f"収集：{saved}レース保存／{failed}件は未確定・未取得")

    joined = load_joined_races()
    if not joined:
        st.info(
            "まだ突き合わせできるレースがありません。\n\n"
            "手順：①「予想」タブでオッズ発売後に予想を出す（自動記録）→ "
            "②レース確定後にこのタブで「結果を自動収集」→ ③ここに実績が出ます。"
        )
    else:
        met = realdata_metrics(joined)
        st.subheader("① 実データの成績")
        c = st.columns(4)
        c[0].metric("検証レース数", f"{met['races']:,}")
        c[1].metric("的中率", f"{met['hit_rate']*100:.1f}%")
        c[2].metric("回収率", f"{met['roi']*100:+.1f}%")
        c[3].metric("推定edge", f"{met['edge']:.2f}")
        st.write(
            f"投資 ¥{met['invest']:,}／払戻 ¥{met['return']:,}／"
            f"収支 ¥{met['profit']:,}（¥100/点の均等買い基準）"
        )
        if met["edge"] >= 1.0 and met["roi"] > 0:
            st.success("回収率プラス＝この期間はオッズを上回る優位性あり。")
        else:
            st.warning("回収率マイナス＝現状は優位性が確認できません（控除率25%の壁）。")

        if met["races"] < 100:
            st.info(
                f"⚠️ 検証レースが {met['races']} 件と少なく、数字はまだ運の影響が大きいです。"
                "目安として100レース以上でedgeの判断が安定します。"
            )

        st.subheader("② どの掛け金戦略が資金を増やすか（あなたの実データで再現）")
        st.caption("初期資金 ¥300,000 として、実際に起きたレース系列に各戦略を当てはめた結果。")
        st.table(replay_staking(joined))
        st.caption(
            "※ 優位性(回収率)がプラスでなければ、どの戦略でも長期的には資金は増えません。"
            "掛け金戦略は『リスクの形』を変えるだけで、勝ち負けの符号は変えられません。"
        )
