from __future__ import annotations
from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class PreferencePair(Base):
    __tablename__ = "preference_pairs"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    pair_id     = Column(String, unique=True, nullable=False, index=True)
    prompt      = Column(Text, nullable=False)
    chosen      = Column(Text, nullable=False)    # preferred response
    rejected    = Column(Text, nullable=False)    # dispreferred response
    domain      = Column(String, nullable=False, index=True)
    source      = Column(String, default="synthetic")
    created_at  = Column(DateTime, default=datetime.utcnow)

class EvalResult(Base):
    __tablename__ = "eval_results"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    eval_id         = Column(String, unique=True, nullable=False)
    model_name      = Column(String, nullable=False)
    prompt          = Column(Text, nullable=False)
    response        = Column(Text, nullable=False)
    helpfulness     = Column(Float, nullable=True)
    harmlessness    = Column(Float, nullable=True)
    factuality      = Column(Float, nullable=True)
    preferred_over_sft = Column(Boolean, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow, index=True)

class TrainingRun(Base):
    __tablename__ = "training_runs"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_id          = Column(String, unique=True, nullable=False)
    base_model      = Column(String, nullable=False)
    num_pairs       = Column(Integer, nullable=False)
    beta            = Column(Float, nullable=False)
    learning_rate   = Column(Float, nullable=False)
    num_epochs      = Column(Integer, nullable=False)
    final_loss      = Column(Float, nullable=True)
    mlflow_run_id   = Column(String, nullable=True)
    adapter_path    = Column(String, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
