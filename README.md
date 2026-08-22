# ACSS Demo — Protocol Parsing & Fact-Based Validation

Two-part live demo:
1. An LLM parses a real synthesis protocol into a hierarchical task DAG (material-flow
   dependencies, not text order).
2. A reagent's hidden role is proposed by an LLM, then checked by a separate model
   against real retrieved evidence before it's trusted.

## Requirements

- Python 3.10+ (conda or any virtualenv)
- An **Anthropic API key** — required. Both parts call Claude live; Part 1
  (protocol parsing) has no offline mode, so it needs network access and makes a
  real, billed call every time it runs.
- An **OpenAI API key** — optional. Part 2's judge step calls GPT, but falls back
  to a bundled cached response for the 3 built-in cases if this isn't set.

## Setup

```bash
conda create -n acss-demo python=3.11 -y
conda activate acss-demo
pip install -r requirements.txt

cp .env.template .env   # then edit .env and add your own key(s)
```

Then launch from *this* directory (the notebook loads files by relative path):

```bash
jupyter lab
```

Open `acss_demo.ipynb` and run cells top to bottom.

## What's here

| File | Purpose |
|---|---|
| `acss_demo.ipynb` | The demo notebook |
| `pipeline.py`, `llm_prompts.py` | Protocol → DAG parsing (Part 1) |
| `llm_cache.py` | Live-call-with-cached-fallback wrapper (Part 2) |
| `demo_protocols.json` | 3 sample protocols for Part 1 |
| `validation_reference.json`, `validation_evidence.json` | Historical run data for Part 2 |
| `demo_cache/` | Cached LLM responses so Part 2 still works offline |
