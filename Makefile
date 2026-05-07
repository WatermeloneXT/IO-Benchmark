CONFIG ?= config.yaml
BOOK_DATA ?= data/book_final_155

generate:
	python3 econ_cli.py --config $(CONFIG) generate

generate-vllm:
	python3 econ_cli.py --config $(CONFIG) generate-vllm

evaluate:
	python3 econ_cli.py --config $(CONFIG) evaluate

evaluate-llm:
	python3 econ_cli.py --config $(CONFIG) evaluate-llm

compare-eval-errors:
	python3 econ_cli.py --config $(CONFIG) compare-eval-errors

reconvert:
	python3 econ_cli.py --config $(CONFIG) reconvert

rerun-ids:
	python3 econ_cli.py --config $(CONFIG) build-rerun-question-ids

token-limit-rerun-ids:
	python3 econ_cli.py --config $(CONFIG) build-token-limit-rerun-question-ids

.PHONY: generate generate-vllm evaluate evaluate-llm compare-eval-errors reconvert rerun-ids token-limit-rerun-ids
