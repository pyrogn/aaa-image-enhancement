.PHONY: test

test:
	docker compose --profile test up --build --abort-on-container-exit --exit-code-from test
	docker compose --profile test down
