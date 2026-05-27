serve:
	uvicorn app.main:app --reload --port 8085

test:
	pytest tests/ -v

demo:
	python notebooks/demo.py

generate-data:
	python scripts/generate_preference_data.py

train-dpo:
	python scripts/train_dpo.py

eval:
	python scripts/eval_comparison.py
