from __future__ import annotations
import uuid
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Any
import os
from dotenv import load_dotenv

from app.database import get_db, engine
from app.models import Base, PreferencePair, EvalResult, TrainingRun

load_dotenv()
Base.metadata.create_all(bind=engine)

app = FastAPI(title="DPO Alignment", version="1.0.0")

# Preference data

class PairReq(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    domain: str
    source: str = "human"

@app.post("/preferences", tags=["data"])
def add_preference_pair(req: PairReq, db: Session = Depends(get_db)):
    pair = PreferencePair(
        pair_id=str(uuid.uuid4())[:8],
        prompt=req.prompt,
        chosen=req.chosen,
        rejected=req.rejected,
        domain=req.domain,
        source=req.source,
    )
    db.add(pair)
    db.commit()
    db.refresh(pair)
    return {"pair_id": pair.pair_id, "domain": pair.domain}

@app.get("/preferences", tags=["data"])
def list_preference_pairs(domain: str = None, limit: int = 20, db: Session = Depends(get_db)):
    q = db.query(PreferencePair)
    if domain:
        q = q.filter_by(domain=domain)
    pairs = q.order_by(PreferencePair.created_at.desc()).limit(limit).all()
    return [
        {
            "pair_id":  p.pair_id,
            "domain":   p.domain,
            "source":   p.source,
            "prompt":   p.prompt[:100] + "..." if len(p.prompt) > 100 else p.prompt,
            "chosen":   p.chosen[:100] + "...",
            "rejected": p.rejected[:100] + "...",
        }
        for p in pairs
    ]

@app.get("/preferences/stats", tags=["data"])
def preference_stats(db: Session = Depends(get_db)):
    total = db.query(PreferencePair).count()
    from sqlalchemy import func
    by_domain = db.query(
        PreferencePair.domain,
        func.count(PreferencePair.id)
    ).group_by(PreferencePair.domain).all()
    return {
        "total_pairs": total,
        "by_domain": {d: c for d, c in by_domain},
    }

@app.get("/preferences/quality", tags=["data"])
def preference_quality_check(threshold: float = 0.8, db: Session = Depends(get_db)):
    """Flags pairs where chosen and rejected responses are near-identical."""
    pairs = db.query(PreferencePair).all()
    flagged = []
    for p in pairs:
        chosen_tokens   = set(p.chosen.lower().split())
        rejected_tokens = set(p.rejected.lower().split())
        union        = len(chosen_tokens | rejected_tokens)
        intersection = len(chosen_tokens & rejected_tokens)
        similarity   = round(intersection / union, 4) if union > 0 else 0.0
        if similarity >= threshold:
            flagged.append({
                "pair_id":    p.pair_id,
                "domain":     p.domain,
                "similarity": similarity,
                "issue":      "chosen and rejected are near-identical",
            })
    return {
        "total_pairs":   len(pairs),
        "flagged":       len(flagged),
        "threshold":     threshold,
        "flagged_pairs": flagged,
    }

@app.get("/preferences/{pair_id}", tags=["data"])
def get_preference_pair(pair_id: str, db: Session = Depends(get_db)):
    pair = db.query(PreferencePair).filter_by(pair_id=pair_id).first()
    if not pair:
        raise HTTPException(404, f"Preference pair '{pair_id}' not found")
    return {
        "pair_id":  pair.pair_id,
        "domain":   pair.domain,
        "source":   pair.source,
        "prompt":   pair.prompt,
        "chosen":   pair.chosen,
        "rejected": pair.rejected,
        "created_at": str(pair.created_at),
    }

# Training runs

@app.get("/training/runs", tags=["training"])
def list_training_runs(db: Session = Depends(get_db)):
    runs = db.query(TrainingRun).order_by(TrainingRun.created_at.desc()).all()
    return [
        {
            "run_id":       r.run_id,
            "base_model":   r.base_model,
            "num_pairs":    r.num_pairs,
            "beta":         r.beta,
            "learning_rate": r.learning_rate,
            "adapter_path": r.adapter_path,
            "created_at":   str(r.created_at),
        }
        for r in runs
    ]

# Eval results

@app.get("/eval/results", tags=["eval"])
def eval_results(model_name: str = None, db: Session = Depends(get_db)):
    q = db.query(EvalResult)
    if model_name:
        q = q.filter_by(model_name=model_name)
    results = q.order_by(EvalResult.created_at.desc()).all()
    return [
        {
            "eval_id":      r.eval_id,
            "model":        r.model_name,
            "prompt":       r.prompt[:80] + "...",
            "helpfulness":  r.helpfulness,
            "harmlessness": r.harmlessness,
            "factuality":   r.factuality,
        }
        for r in results
    ]

@app.get("/eval/comparison", tags=["eval"])
def eval_comparison(db: Session = Depends(get_db)):
    """Side-by-side SFT vs DPO comparison."""
    results = db.query(EvalResult).all()
    if not results:
        return {"message": "No eval results yet. Run: make eval"}

    summary = {}
    for r in results:
        if r.model_name not in summary:
            summary[r.model_name] = {
                "helpfulness": [], "harmlessness": [], "factuality": []
            }
        if r.helpfulness  is not None: summary[r.model_name]["helpfulness"].append(r.helpfulness)
        if r.harmlessness is not None: summary[r.model_name]["harmlessness"].append(r.harmlessness)
        if r.factuality   is not None: summary[r.model_name]["factuality"].append(r.factuality)

    comparison = {}
    for model, scores in summary.items():
        comparison[model] = {
            k: round(sum(v) / len(v), 4) if v else None
            for k, v in scores.items()
        }

    return {
        "comparison": comparison,
        "winner": _pick_winner(comparison),
    }

def _pick_winner(comparison: dict) -> str:
    if len(comparison) < 2:
        return list(comparison.keys())[0] if comparison else "no data"
    scores = {
        m: sum(v for v in metrics.values() if v is not None)
        for m, metrics in comparison.items()
    }
    return max(scores, key=scores.get)

@app.get("/eval/winrate", tags=["eval"])
def eval_winrate(db: Session = Depends(get_db)):
    """Per-metric win rate: fraction of prompts where DPO outscores SFT."""
    results = db.query(EvalResult).all()
    if not results:
        return {"message": "No eval results yet. Run: make eval"}

    by_prompt: dict[str, dict] = {}
    for r in results:
        if r.prompt not in by_prompt:
            by_prompt[r.prompt] = {}
        by_prompt[r.prompt][r.model_name] = r

    metrics = ["helpfulness", "harmlessness", "factuality"]
    wins  = {m: 0 for m in metrics}
    ties  = {m: 0 for m in metrics}
    total = 0

    for models in by_prompt.values():
        if "sft" not in models or "dpo" not in models:
            continue
        sft = models["sft"]
        dpo = models["dpo"]
        total += 1
        for m in metrics:
            dpo_score = getattr(dpo, m) or 0.0
            sft_score = getattr(sft, m) or 0.0
            if dpo_score > sft_score:
                wins[m] += 1
            elif abs(dpo_score - sft_score) < 1e-6:
                ties[m] += 1

    if total == 0:
        return {"message": "Need both sft and dpo results. Run: make eval"}

    return {
        "total_prompts_compared": total,
        "win_rates": {m: round(wins[m] / total, 4) for m in metrics},
        "tie_rates": {m: round(ties[m] / total, 4) for m in metrics},
        "overall_win_rate": round(sum(wins.values()) / (len(metrics) * total), 4),
    }

@app.get("/health")
def health():
    return {"status": "ok"}
