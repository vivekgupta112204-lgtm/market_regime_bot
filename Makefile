.PHONY: help install start test lint clean docker-build docker-up

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

help:
	@echo "Available Commands:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make start        - Start API and Streamlit Dashboard"
	@echo "  make test         - Run Pytest unit tests"
	@echo "  make lint         - Run Black, Flake8, and isort"
	@echo "  make clean        - Remove cached compilation files"
	@echo "  make docker-build - Build Docker container"
	@echo "  make docker-up    - Run Docker compose environment"

install:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

start:
	$(PYTHON) run_phase12.py

test:
	$(PYTHON) -m pytest tests/ --cov=ai --cov=analytics

lint:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .
	$(PYTHON) -m flake8 .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t market_regime_bot:latest .

docker-up:
	docker-compose up -d
