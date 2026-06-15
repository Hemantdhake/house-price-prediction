# ─── Variables ────────────────────────────────────────────────
PYTHON = python
PIP    = pip
PYTEST = pytest
APP    = app/app.py

# ─── Help ─────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  House Price Prediction - Makefile"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make install       → Install dependencies"
	@echo "  make train         → Train the model"
	@echo "  make test          → Run all tests"
	@echo "  make run           → Start Flask app"
	@echo "  make docker-build  → Build Docker image"
	@echo "  make docker-run    → Run Docker container"
	@echo "  make clean         → Clean cache files"
	@echo "  make lint          → Run linter"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─── Installation ─────────────────────────────────────────────
.PHONY: install
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

# ─── Training ─────────────────────────────────────────────────
.PHONY: train
train:
	$(PYTHON) src/model.py
	@echo "✅ Model training complete"

# ─── Testing ──────────────────────────────────────────────────
.PHONY: test
test:
	$(PYTEST) tests/ -v --cov=src --cov-report=html
	@echo "✅ All tests passed"

# ─── Run App ──────────────────────────────────────────────────
.PHONY: run
run:
	$(PYTHON) $(APP)

# ─── Docker ───────────────────────────────────────────────────
.PHONY: docker-build
docker-build:
	docker build -t house-price-prediction:latest .
	@echo "✅ Docker image built"

.PHONY: docker-run
docker-run:
	docker run -p 5000:5000 house-price-prediction:latest

# ─── Clean ────────────────────────────────────────────────────
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	@echo "✅ Clean complete"

# ─── Lint ─────────────────────────────────────────────────────
.PHONY: lint
lint:
	flake8 src/ app/ tests/ --max-line-length=88
	black src/ app/ tests/ --check
	isort src/ app/ tests/ --check-only
	@echo "✅ Linting complete"

# ─── Format ───────────────────────────────────────────────────
.PHONY: format
format:
	black src/ app/ tests/
	isort src/ app/ tests/
	@echo "✅ Code formatted"