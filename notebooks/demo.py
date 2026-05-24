"""
DPO Alignment demo.
Run: make serve then make demo
For full eval: make generate-data, make train-dpo, make eval first.
"""
from __future__ import annotations
import requests, json

BASE = "http://localhost:8085"

def post(path, payload): return requests.post(f"{BASE}{path}", json=payload).json()
def get(path):           return requests.get(f"{BASE}{path}").json()

print("\n=== DPO Alignment Demo ===\n")

# 1. Health
print(f"1. Health: {get('/health')}")

# 2. Check preference data
print("\n2. Preference dataset stats:")
stats = get("/preferences/stats")
print(f"   {json.dumps(stats, indent=4)}")

# 3. Add a new preference pair
print("\n3. Adding a human preference pair...")
r = post("/preferences", {
    "prompt":   "What does negative free cash flow indicate?",
    "chosen":   "Negative free cash flow means the company is spending more cash than it generates from operations, after capital expenditures. This can be intentional during high-growth phases where companies reinvest heavily, or a warning sign if persistent without a clear growth thesis. Context is critical: Amazon ran negative FCF for years while building infrastructure.",
    "rejected": "Negative free cash flow is bad. The company is losing money.",
    "domain":   "financial_analysis",
    "source":   "human",
})
print(f"   Added pair: {r}")

# 4. List recent pairs
print("\n4. Recent preference pairs:")
pairs = get("/preferences?limit=3")
for p in pairs:
    print(f"   [{p['domain']}] {p['prompt'][:60]}...")
    print(f"     chosen:   {p['chosen'][:80]}...")
    print(f"     rejected: {p['rejected'][:60]}...")

# 5. Training runs
print("\n5. Training runs:")
runs = get("/training/runs")
if runs:
    for r in runs:
        print(f"   run_id={r['run_id']}  beta={r['beta']}  pairs={r['num_pairs']}")
else:
    print("   No runs yet — run: make train-dpo")

# 6. Eval comparison
print("\n6. SFT vs DPO eval comparison:")
comparison = get("/eval/comparison")
if "message" in comparison:
    print(f"   {comparison['message']}")
    print("   Run: make eval to compare models")
else:
    print(f"   {json.dumps(comparison, indent=4)}")

print(f"\nAPI docs  → http://localhost:8085/docs")
print("Next steps:")
print("  make generate-data  — populate preference dataset")
print("  make train-dpo      — run DPO training")
print("  make eval           — compare SFT vs DPO")
print("\nDone.")
