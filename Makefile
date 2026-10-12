.PHONY: setup test clean

setup:
	pip install -e .[dev]

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .coverage
	rm -rf data/processed/* src/knowtrace/models/*.pt src/knowtrace/models/*.pth
	rm -rf reports/figures/*