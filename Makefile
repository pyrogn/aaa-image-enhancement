.PHONY: test up demo down help

help:
	@echo "Available targets:"
	@echo "  test  - Run tests with Docker Compose using the 'test' profile."
	@echo "  up    - Build and start all services by default in the background."
	@echo "  demo  - Build and start services using the 'demo' profile in the background."
	@echo "  down  - Stop and remove all services from all profiles."

test:
	docker compose --profile test up --build --abort-on-container-exit --exit-code-from test
	docker compose --profile test down

up:
	docker compose up --build -d

demo:
	docker compose --profile demo up --build -d

down:
	docker compose --profile "*" down
