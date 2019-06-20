.PHONY: start-dev run restart stop

TEST_PATH=./
export FLASK_APP=./index.py

help:
	@echo "    start-dev"
	@echo "       Start Ethicmodel Flask server (development)"
	@echo "    run"
	@echo "       Start Ethicmodel server in Docker container"
	@echo "    restart"
	@echo "       Restart running container"
	@echo "    stop"
	@echo "       Stop running container"

start-dev:
	flask run -h 0.0.0.0 -p 5050

run:
	docker run \
		-p 5050:5050 \
		-v $(shell pwd):/app \
		--name ethicmodel \
		aroemelt/ethicbot:ethicmodel \
		gunicorn -w 4 -b 0.0.0.0:5050 index:app

restart:
	docker restart ethicmodel

stop:
	docker stop ethicmodel