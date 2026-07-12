.PHONY: up down ingest test

up:
	docker compose up -d --build

down:
	docker compose down

ingest:
	docker compose exec api python -m app.ingest /data/public-games-dataset

test:
	pytest -q
