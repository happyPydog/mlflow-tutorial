.PHONY: test install

test:
	python -m pytest tests

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

init:
	@if [ ! -d ".venv" ]; then \
		python3.11 -m venv .venv; \
	fi
	@source .venv/bin/activate; \
	python -m pip install --upgrade pip
	python -m pip install wheel setuptools
	make install
