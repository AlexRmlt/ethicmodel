.PHONY: start-dev start-prod run-container restart-container stop-container

TEST_PATH=./
export FLASK_APP=./index.py

help:
	@echo "    start-dev"
	@echo "       Start Ethicmodel Flask server (development)"
	@echo "    start-prod"
	@echo "       Start Ethicmodel Gunicorn server (production)"
	@echo "    run-container"
	@echo "       Start Ethicmodel server in Docker container"
	@echo "    restart-container"
	@echo "       Restart running container"
	@echo "    stop-container"
	@echo "       Stop running container"

start-dev:
	flask run -h 0.0.0.0 -p 5050

start-prod:
	gunicorn --bind 0.0.0.0:5050 index:app

run-container:
	docker run \
		-p 5050:5050 \
		-v $(shell pwd):/app \
		--name ethicmodel \
		aroemelt/ethicbot:ethicmodel \
		gunicorn -w 4 -b 0.0.0.0:5050 index:app

restart-container:
	docker restart ethicmodel

stop-container:
	docker stop ethicmodel