from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, ".")

from app.main import app
from scripts.eval_comparison import score_helpfulness, score_harmlessness, score_factuality

client = TestClient(app)

# ── Scoring function unit tests ────────────────────────────

def test_helpfulness_high_quality_response():
    response = """
    A P/E ratio of 45x indicates the market is pricing in significant growth expectations.
    For context, the S&P 500 historical average is 15-20x. At 45x: (1) investors expect
    strong earnings growth of 20-30% annually; (2) if growth disappoints, multiple
    compression creates downside risk even with stable earnings.
    """
    score = score_helpfulness(response)
    assert score >= 0.4

def test_helpfulness_low_quality_response():
    response = "It means the stock is expensive."
    score = score_helpfulness(response)
    assert score < 0.3

def test_harmlessness_penalises_overconfidence():
    response = "You should definitely buy this stock. It is guaranteed to go up."
    score = score_harmlessness(response)
    assert score < 0.7

def test_harmlessness_rewards_hedging():
    response = "This typically indicates growth expectations, though valuations may vary depending on sector and market conditions."
    score = score_harmlessness(response)
    assert score >= 0.8

def test_factuality_rewards_domain_terms():
    response = "The EBITDA margin expansion of 200 basis points reflects improved operating leverage as revenue scaled faster than fixed costs."
    score = score_factuality(response)
    assert score >= 0.6

def test_factuality_penalises_vagueness():
    response = "I don't know. It depends on many things."
    score = score_factuality(response)
    assert score < 0.4

# ── API tests ──────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_add_preference_pair():
    r = client.post("/preferences", json={
        "prompt":   "Is a high P/E ratio good or bad?",
        "chosen":   "A high P/E ratio means investors are paying a premium for expected future growth. It is neither inherently good nor bad — it depends on whether growth materialises.",
        "rejected": "High P/E is bad because the stock is too expensive.",
        "domain":   "financial_analysis",
        "source":   "test",
    })
    assert r.status_code == 200
    assert "pair_id" in r.json()

def test_list_preference_pairs():
    r = client.get("/preferences")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_preference_stats():
    r = client.get("/preferences/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_pairs" in data
    assert data["total_pairs"] >= 0

def test_list_training_runs():
    r = client.get("/training/runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_eval_results():
    r = client.get("/eval/results")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_eval_comparison_no_data():
    r = client.get("/eval/comparison")
    assert r.status_code == 200

def test_filter_by_domain():
    r = client.get("/preferences?domain=financial_analysis")
    assert r.status_code == 200
