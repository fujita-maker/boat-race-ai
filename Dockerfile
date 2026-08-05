FROM python:3.12-slim

WORKDIR /app

# 依存だけ先に入れてキャッシュを効かせる
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY . .

# Render は $PORT を注入する。streamlit ではなく uvicorn で起動する。
ENV PORT=10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
