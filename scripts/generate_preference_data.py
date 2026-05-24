"""
Generates synthetic preference pairs for DPO training.
Domain: financial sentiment and analysis.
Chosen responses are accurate, well-reasoned, and cite evidence.
Rejected responses are vague, overconfident, or hallucinated.
Run: make generate-data
"""
from __future__ import annotations
import sys, uuid, json
sys.path.insert(0, ".")

from app.database import engine, SessionLocal
from app.models import Base, PreferencePair

Base.metadata.create_all(bind=engine)

PREFERENCE_PAIRS = [
    # ── Sentiment analysis pairs ───────────────────────────
    {
        "prompt": "Analyse the sentiment of this financial statement: 'Revenue grew 23% year-over-year driven by strong enterprise demand and new product launches.'",
        "chosen": "This statement reflects positive sentiment. The 23% YoY revenue growth is a strong quantitative signal, and the attribution to enterprise demand and new product launches indicates sustainable growth drivers rather than one-off factors. Overall: bullish.",
        "rejected": "The sentiment is positive because it mentions growth. Revenue going up is always good for a company.",
        "domain": "financial_sentiment",
    },
    {
        "prompt": "Analyse the sentiment of this financial statement: 'Operating margins contracted 400 basis points as input costs rose faster than pricing power allowed.'",
        "chosen": "This is negative sentiment. A 400bps margin contraction is significant — it signals that cost inflation is outpacing the company's ability to pass costs onto customers, which compresses profitability. This is a bearish indicator for near-term earnings.",
        "rejected": "This sounds negative. Margins going down is bad. The company has cost problems.",
        "domain": "financial_sentiment",
    },
    {
        "prompt": "Analyse the sentiment of this financial statement: 'The company maintained flat revenue while reducing headcount by 12%, resulting in improved EBITDA margins.'",
        "chosen": "Mixed-to-positive sentiment. Flat revenue signals a challenging growth environment, but the 12% headcount reduction driving EBITDA margin improvement suggests disciplined cost management. Markets typically react positively to margin expansion even with flat topline, as it signals operational leverage.",
        "rejected": "This is confusing. Revenue is flat which is bad but margins are up which is good. Hard to say.",
        "domain": "financial_sentiment",
    },
    {
        "prompt": "What does a price-to-earnings ratio of 45x indicate about a stock?",
        "chosen": "A P/E of 45x indicates the market is paying a significant premium relative to current earnings — either because investors expect strong future earnings growth, or the stock is overvalued. For context, the S&P 500 historical average P/E is around 15-20x. At 45x, the stock is priced for high growth. If growth disappoints, the multiple will compress, creating downside risk even if earnings are stable.",
        "rejected": "A P/E of 45 means the stock is expensive. You should probably not buy it because it costs too much.",
        "domain": "financial_analysis",
    },
    {
        "prompt": "Explain the difference between gross margin and operating margin.",
        "chosen": "Gross margin measures profitability after deducting the direct cost of goods sold (COGS) from revenue: (Revenue - COGS) / Revenue. Operating margin goes further, also deducting operating expenses like R&D, sales, and general and administrative costs: Operating Income / Revenue. A company can have a high gross margin but a low operating margin if it spends heavily on R&D or sales — common in software companies investing in growth.",
        "rejected": "Gross margin is about the product cost and operating margin includes more expenses. Both measure how profitable a company is.",
        "domain": "financial_analysis",
    },
    {
        "prompt": "A company reports EPS of $2.50 versus analyst consensus of $2.20. How should this be interpreted?",
        "chosen": "This is a positive earnings surprise — the company beat consensus estimates by $0.30 or approximately 13.6%. Earnings beats of this magnitude typically drive short-term stock price appreciation as the market reprices expectations upward. However, the quality of the beat matters: was it driven by revenue outperformance, cost cuts, or one-time items? Revenue-driven beats are more sustainable and typically rewarded with higher multiple expansion.",
        "rejected": "The company did better than expected. This is good news and the stock will probably go up.",
        "domain": "financial_analysis",
    },
    {
        "prompt": "What are the risks of high financial leverage for a company?",
        "chosen": "High financial leverage — a large proportion of debt relative to equity — amplifies both gains and losses. Key risks include: (1) interest coverage risk — if earnings decline, the company may struggle to service debt; (2) refinancing risk — when debt matures in a high-rate environment, refinancing at higher rates compresses margins; (3) covenant risk — debt covenants may restrict operational flexibility; (4) bankruptcy risk — in a severe downturn, overleveraged companies face insolvency. Leverage is particularly dangerous in cyclical businesses with volatile cash flows.",
        "rejected": "Too much debt is risky because you have to pay it back. If the company can't pay it back they could go bankrupt.",
        "domain": "financial_analysis",
    },
    {
        "prompt": "Explain what free cash flow yield indicates about a company's valuation.",
        "chosen": "Free cash flow yield — FCF divided by market capitalisation — measures how much cash a company generates relative to its market value. A high FCF yield (e.g. above 6-8%) suggests the stock may be undervalued or that the company is highly cash generative. It is often preferred over earnings yield because FCF is harder to manipulate through accounting choices. Value investors use it as a primary valuation screen. A declining FCF yield over time, driven by rising price rather than falling FCF, signals increasing valuation risk.",
        "rejected": "Free cash flow yield shows if a company makes cash. Higher is better because it means the company is making more money.",
        "domain": "financial_valuation",
    },
    {
        "prompt": "What does an inverted yield curve historically signal?",
        "chosen": "An inverted yield curve — where short-term bond yields exceed long-term yields — has historically been one of the most reliable leading indicators of recession. The inversion occurs when markets expect future interest rates to fall, typically in anticipation of economic slowdown. The 2-year/10-year spread inversion has preceded every US recession since 1955, with a typical lead time of 12-18 months. However, the signal has false positives and the lead time varies significantly, making it a probabilistic rather than deterministic indicator.",
        "rejected": "An inverted yield curve means short term rates are higher than long term rates. This is unusual and means the economy might have problems.",
        "domain": "macroeconomics",
    },
    {
        "prompt": "How should an investor interpret rising days sales outstanding (DSO)?",
        "chosen": "Rising DSO — the average number of days a company takes to collect payment after a sale — is a warning signal. It can indicate: (1) customers are under financial stress and paying more slowly; (2) the company is extending looser credit terms to inflate revenue recognition; (3) revenue may be overstated if receivables are growing faster than sales. Sustained DSO increases, especially when diverging from industry peers, warrant scrutiny of revenue quality and working capital management.",
        "rejected": "DSO going up means the company is taking longer to get paid. This could be a problem if it gets too high.",
        "domain": "financial_analysis",
    },
]

def generate():
    db = SessionLocal()
    try:
        added = 0
        for pair in PREFERENCE_PAIRS:
            pair_id = str(uuid.uuid4())[:8]
            existing = db.query(PreferencePair).filter_by(
                prompt=pair["prompt"]
            ).first()
            if existing:
                continue
            pp = PreferencePair(
                pair_id=pair_id,
                prompt=pair["prompt"],
                chosen=pair["chosen"],
                rejected=pair["rejected"],
                domain=pair["domain"],
                source="synthetic",
            )
            db.add(pp)
            added += 1
        db.commit()
        total = db.query(PreferencePair).count()
        print(f"Added {added} new pairs. Total: {total}")

        # Export raw pairs (for reference)
        pairs = db.query(PreferencePair).all()
        with open("data/preference_pairs.jsonl", "w") as f:
            for p in pairs:
                f.write(json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}) + "\n")

        # mlx-lm lora expects train.jsonl / valid.jsonl with {"text": "..."} rows
        # Use Mistral chat format so training and inference use the same tokens
        formatted = [
            {"text": f"<s>[INST] {p.prompt} [/INST] {p.chosen}</s>"}
            for p in pairs
        ]
        split = max(1, len(formatted) - 2)
        with open("data/train.jsonl", "w") as f:
            for r in formatted[:split]:
                f.write(json.dumps(r) + "\n")
        with open("data/valid.jsonl", "w") as f:
            for r in formatted[split:]:
                f.write(json.dumps(r) + "\n")
        print(f"Exported {len(pairs)} pairs → data/train.jsonl ({split}) + valid.jsonl ({len(formatted)-split})")
    finally:
        db.close()

if __name__ == "__main__":
    generate()
