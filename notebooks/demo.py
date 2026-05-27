"""
DPO Alignment demo: preference data, training run summary, and eval comparison.
Run: make serve (in project_07) then make demo
"""
from __future__ import annotations
import requests

BASE = "http://localhost:8085"

def get(path, **kw):  return requests.get(f"{BASE}{path}", params=kw).json()
def post(path, body): return requests.post(f"{BASE}{path}", json=body).json()

print("\n=== DPO Alignment Demo ===\n")

# 1. Dataset stats
print("1. Preference dataset...")
stats = get("/preferences/stats")
print(f"   Total pairs: {stats.get('total_pairs', 0)}")
for domain, count in stats.get("by_domain", {}).items():
    print(f"   {domain}: {count} pairs")

# 2. Sample a preference pair
print("\n2. Sample preference pair...")
pairs = get("/preferences", limit=1)
if pairs:
    p = pairs[0]
    print(f"   Domain:   {p['domain']}")
    print(f"   Prompt:   {p['prompt'][:80]}...")
    print(f"   Chosen:   {p['chosen'][:80]}...")
    print(f"   Rejected: {p['rejected'][:80]}...")

# 3. Training run history
print("\n3. Training runs...")
runs = get("/training/runs")
if runs:
    r = runs[0]
    print(f"   run_id={r['run_id']}  beta={r['beta']}  lr={r['learning_rate']}  pairs={r['num_pairs']}")
else:
    print("   No runs yet. Run: make train-dpo")

# 4. Eval comparison
print("\n4. SFT vs DPO eval comparison...")
comp = get("/eval/comparison")
if "comparison" in comp:
    for model, scores in comp["comparison"].items():
        print(f"   {model.upper()}: helpfulness={scores.get('helpfulness')}  harmlessness={scores.get('harmlessness')}  factuality={scores.get('factuality')}")
    print(f"   Winner: {comp.get('winner')}")
else:
    print(f"   {comp.get('message', 'Run: make eval')}")

# 5. Win rate
print("\n5. Per-metric win rate (DPO vs SFT)...")
wr = get("/eval/winrate")
if "win_rates" in wr:
    for metric, rate in wr["win_rates"].items():
        print(f"   {metric}: {rate:.0%}")
    print(f"   Overall: {wr['overall_win_rate']:.0%}")
else:
    print(f"   {wr.get('message', 'Run: make eval')}")

# 6. Quality check
print("\n6. Preference pair quality check...")
quality = get("/preferences/quality")
print(f"   Total: {quality['total_pairs']}  Flagged: {quality['flagged']}  Threshold: {quality['threshold']}")
if quality["flagged_pairs"]:
    for f in quality["flagged_pairs"][:3]:
        print(f"   [FLAG] {f['pair_id']} domain={f['domain']} similarity={f['similarity']}")

print(f"\nFull API docs at {BASE}/docs")
print("Done.")
