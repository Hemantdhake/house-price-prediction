# ─── Base Image ───────────────────────────────────────────────
FROM python:3.9-slim

# ─── Set Working Directory ────────────────────────────────────
WORKDIR /app

# ─── Environment Variables ────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app/app.py \
    FLASK_ENV=production

# ─── Install System Dependencies ──────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ─── Install Python Dependencies ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Copy Project Files ───────────────────────────────────────
COPY . .

# ─── Expose Port ──────────────────────────────────────────────
EXPOSE 5000

# ─── Health Check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# ─── Run Application ──────────────────────────────────────────
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app.app:app"]