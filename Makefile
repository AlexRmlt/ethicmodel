.PHONY: start-dev run-container

TEST_PATH=./
export FLASK_APP=./index.py

help:
	@echo "    start-dev"
	@echo "       Start Ethicmodel Flask server (development)"
	@echo "    run-container"
	@echo "       Start Ethicmodel server in Docker container"

start-dev:
	flask run -h 0.0.0.0 -p 5050

run-container:
	docker run \
		-p 5050:5050 \
		-v $(shell pwd):/app \
		aroemelt/ethicbot:ethicmodel \
		gunicorn -w 4 -b 0.0.0.0:5050 index:app