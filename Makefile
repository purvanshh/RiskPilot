ifeq ($(OS),Windows_NT)
    PYTHON = python
else
    PYTHON = python3
endif

.PHONY: install test run-demo lint format clean

install:
	pip install -r requirements.txt
	pip install black isort flake8 pre-commit
	pre-commit install

test:
	pytest -v --cov=src --cov-report=term-missing

run-demo:
	$(PYTHON) src/ui/app.py

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/
	rm -rf data/chroma_db/
