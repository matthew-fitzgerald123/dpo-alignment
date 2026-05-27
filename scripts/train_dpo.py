"""
DPO fine-tuning using mlx-lm.
Logs training run to MLflow (P2 experiment tracker).
Run: make train-dpo
Prerequisites: make generate-data first
"""
from __future__ import annotations
import sys, os, uuid, subprocess, time, json
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import mlflow
from app.database import engine, SessionLocal
from app.models import Base, TrainingRun

Base.metadata.create_all(bind=engine)

BASE_MODEL   = os.getenv("BASE_MODEL", "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "postgresql://localhost/mlplatform")
ADAPTER_DIR  = "./adapters/dpo_financial"
DATA_PATH    = "./data/preference_pairs.jsonl"

# DPO hyperparameters
# beta controls KL divergence penalty; how far DPO can drift from SFT base
# Lower beta = more aggressive alignment, higher beta = stay closer to base
BETA          = 0.1
LEARNING_RATE = 5e-5    # same as P1, safe for 4-bit quantised models
NUM_EPOCHS    = 2
BATCH_SIZE    = 1       # keep at 1 for Apple Silicon memory stability
LORA_LAYERS   = 8
SAVE_EVERY    = 100

def check_data():
    if not os.path.exists(DATA_PATH):
        print(f"No data found at {DATA_PATH}")
        print("Run: make generate-data first")
        sys.exit(1)
    with open(DATA_PATH) as f:
        lines = [l for l in f if l.strip()]
    print(f"Training on {len(lines)} preference pairs")
    return len(lines)

def train():
    num_pairs = check_data()
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("dpo_alignment")

    with mlflow.start_run(run_name=f"dpo_{run_id}") as run:
        mlflow_run_id = run.info.run_id
        mlflow.log_params({
            "base_model":    BASE_MODEL,
            "beta":          BETA,
            "learning_rate": LEARNING_RATE,
            "num_epochs":    NUM_EPOCHS,
            "batch_size":    BATCH_SIZE,
            "lora_layers":   LORA_LAYERS,
            "num_pairs":     num_pairs,
            "training_type": "dpo",
        })

        cmd = [
            "python", "-m", "mlx_lm", "lora",
            "--train",
            "--model",         BASE_MODEL,
            "--data",          "./data",
            "--adapter-path",  ADAPTER_DIR,
            "--num-layers",    str(LORA_LAYERS),
            "--batch-size",    str(BATCH_SIZE),
            "--iters",         str(num_pairs * NUM_EPOCHS),
            "--learning-rate", str(LEARNING_RATE),
            "--save-every",    str(SAVE_EVERY),
            "--grad-checkpoint",
        ]

        print(f"\nStarting DPO training: run_id={run_id}")
        print(f"Command: {' '.join(cmd)}\n")
        print("caffeinate -i wrapping to prevent sleep...")

        t_start = time.time()

        result = subprocess.run(
            ["caffeinate", "-i"] + cmd,
            capture_output=False,
        )

        duration = round(time.time() - t_start, 1)

        if result.returncode != 0:
            mlflow.set_tag("status", "failed")
            print(f"\nTraining failed with return code {result.returncode}")
            sys.exit(1)

        mlflow.log_metrics({
            "training_duration_seconds": duration,
            "pairs_per_second": round(num_pairs / duration, 2),
        })
        mlflow.set_tag("status", "completed")
        mlflow.set_tag("adapter_path", ADAPTER_DIR)

        print(f"\nTraining complete in {duration}s")
        print(f"Adapter saved to: {ADAPTER_DIR}")
        print(f"MLflow run: {mlflow_run_id}")

        db = SessionLocal()
        try:
            tr = TrainingRun(
                run_id=run_id,
                base_model=BASE_MODEL,
                num_pairs=num_pairs,
                beta=BETA,
                learning_rate=LEARNING_RATE,
                num_epochs=NUM_EPOCHS,
                mlflow_run_id=mlflow_run_id,
                adapter_path=ADAPTER_DIR,
            )
            db.add(tr)
            db.commit()
        finally:
            db.close()

if __name__ == "__main__":
    train()
