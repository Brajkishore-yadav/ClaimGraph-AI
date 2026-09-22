# ============================================================
# ClaimGraph AI — Makefile
# Har ek command ka ek kaam — simple aur clear
# ============================================================

.PHONY: help install generate-data load-postgres load-neo4j train evaluate \
        run-api run-dashboard test lint docker-up docker-down clean generate-pptx

help:  ## Sabhi available commands dikhao
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Dependencies install karo
	pip install -r requirements.txt

generate-data:  ## Synthetic data generate karo (with fraud rings)
	python scripts/generate_data.py --customers 8000 --policies 15000 --claims 25000 --seed 42

load-postgres:  ## Data ko PostgreSQL mein load karo
	python scripts/load_postgres.py

load-neo4j:  ## Data ko Neo4j graph mein load karo
	python scripts/load_neo4j.py

graph-features:  ## Graph algorithms run karo aur features extract karo
	python scripts/extract_graph_features.py

train:  ## Saare ML models train karo (3 experiments × 2 models)
	python scripts/train_models.py

evaluate:  ## Evaluation run karo — ablation table generate karo
	python scripts/evaluate.py

run-api:  ## FastAPI backend start karo
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:  ## Streamlit dashboard start karo
	streamlit run frontend/app.py --server.port 8501

test:  ## Saare tests run karo
	pytest tests/ -v --tb=short

test-unit:  ## Sirf unit tests run karo
	pytest tests/unit/ -v --tb=short

test-integration:  ## Sirf integration tests run karo (Docker services chahiye)
	pytest tests/integration/ -v --tb=short

lint:  ## Code quality check karo
	ruff check src/ scripts/ tests/

generate-pptx:  ## Executive PowerPoint generate karo
	python scripts/generate_pptx.py

monitor:  ## Graph quality + PSI drift check run karo
	python scripts/run_monitoring.py

docker-up:  ## Docker Compose se full stack start karo
	docker compose up --build -d

docker-down:  ## Docker Compose band karo
	docker compose down -v

clean:  ## Generated files clean karo
	rm -rf data/generated/ experiments/runs/*.json models/ __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# === Full Pipeline (pehli baar chalane ke liye) ===
pipeline: generate-data load-postgres load-neo4j graph-features train evaluate  ## Full pipeline ek baar mein chalao
	@echo "✅ Full pipeline complete! Ab 'make run-api' aur 'make run-dashboard' chalao."
