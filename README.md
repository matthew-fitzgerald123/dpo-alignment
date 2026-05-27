# DPO Alignment

A pipeline for collecting human preference data, training a language model with Direct Preference Optimization (DPO), and comparing aligned vs. base model quality. Built on Apple Silicon using mlx-lm.

## Stack

| Component | Library |
|---|---|
| API | FastAPI + uvicorn (port 8085) |
| Training | mlx-lm + mlx (Apple Silicon native) |
| Experiment tracking | MLflow |
| Preference store + eval logs | PostgreSQL + SQLAlchemy |

## Setup

```bash
# Create database
createdb dpo_alignment

# Install dependencies
pip install -r requirements.txt
```

## Running

```bash
# Generate synthetic preference data
make generate-data

# Run DPO training
make train-dpo

# Evaluate and compare SFT vs DPO
make eval

# Start API server
make serve

# Run end-to-end demo
make demo

# Run tests
make test
```

## API Endpoints

### Preference Data

| Method | Path | Description |
|---|---|---|
| POST | `/preferences` | Add a chosen/rejected preference pair |
| GET | `/preferences` | List pairs, filter by domain |
| GET | `/preferences/stats` | Total pairs by domain |

### Training

| Method | Path | Description |
|---|---|---|
| GET | `/training/runs` | List DPO training runs with hyperparameters |

### Evaluation

| Method | Path | Description |
|---|---|---|
| GET | `/eval/results` | Per-prompt eval scores (helpfulness, harmlessness, factuality) |
| GET | `/eval/comparison` | Averaged SFT vs DPO scores + winner |
| GET | `/health` | Server status |

Interactive docs at `http://localhost:8085/docs`.

## How DPO Works

DPO trains directly on preference pairs (prompt, chosen response, rejected response) without a separate reward model. The loss maximizes the likelihood ratio between chosen and rejected completions, scaled by a temperature parameter `beta`. The result is an adapter that shifts the model toward preferred outputs.

## Project Structure

```
app/
  main.py       FastAPI app (preferences, training runs, eval)
  models.py     SQLAlchemy models (PreferencePair, TrainingRun, EvalResult)
  database.py   engine + session
scripts/
  generate_preference_data.py   synthetic preference dataset generation
  train_dpo.py                  DPO training loop (mlx-lm)
  eval_comparison.py            score SFT vs DPO on helpfulness/harmlessness/factuality
adapters/       saved LoRA adapter weights (output of train_dpo.py)
notebooks/
  demo.py       end-to-end demo
tests/
```

## Notes

- `beta` controls the strength of the preference signal; lower values allow more deviation from the base model
- Eval scores are 0-1 rubric-based ratings stored per prompt per model, averaged in `/eval/comparison`
