from __future__ import annotations

from pathlib import Path

APP_NAME = "WDJ Boat Race AI V20 Ultimate"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "boat_race_v20.db"

# Render/GitHubに空フォルダが無い場合でも起動時に必ず作成
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
