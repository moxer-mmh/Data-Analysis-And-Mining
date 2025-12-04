APP_NAME = data-analysis-app

.PHONY: help start

help:
	@echo "Available commands:"
	@echo "  make start  - Run the application locally using run_app.sh"

start:
	./run_app.sh
