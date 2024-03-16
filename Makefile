.PHONY: test install

test:
	poetry run pytest 

install:
	poetry install --no-root
