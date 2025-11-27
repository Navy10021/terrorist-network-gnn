.PHONY: help install install-dev test lint format clean docker-build docker-test docs

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make test-fast     - Run tests in parallel"
	@echo "  make lint          - Run all linters"
	@echo "  make format        - Format code with black and isort"
	@echo "  make type-check    - Run mypy type checking"
	@echo "  make security      - Run security checks"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-test   - Run tests in Docker"
	@echo "  make docs          - Build documentation"
	@echo "  make pre-commit    - Install pre-commit hooks"

install:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install torch-geometric
	pip install -r requirements.txt
	pip install -e .

install-dev: install
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -n auto --cov=src

test-watch:
	pytest-watch tests/ -v

lint:
	@echo "Running Black..."
	black --check src/ tests/
	@echo "Running isort..."
	isort --check-only --profile black src/ tests/
	@echo "Running Flake8..."
	flake8 src/ tests/
	@echo "Running Pylint..."
	pylint src/

format:
	black src/ tests/
	isort --profile black src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

security:
	bandit -r src/ -ll
	safety check

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

docker-build:
	docker-compose build

docker-test:
	docker-compose run --rm test

docker-jupyter:
	docker-compose up jupyter

docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

pre-commit:
	pre-commit install
	pre-commit run --all-files

build:
	python -m build

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

.DEFAULT_GOAL := help
