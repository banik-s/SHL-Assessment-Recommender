# SHL Assessment Recommender

An intelligent recommendation system that maps natural language job descriptions — or job posting URLs — to the most relevant SHL assessments from the [SHL product catalogue](https://www.shl.com/solutions/products/product-catalog/).

### Test the system 
    https://banik-shl-assessment-recommender.hf.space
---

## Architecture Overview

```
User Query (text / URL)
        │
        ▼
┌─────────────────────┐
│   Query Expander    │  OpenAI gpt-4o-mini
│  query_expander.py  │  → expanded query, required SHL types, duration constraint
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│     Retriever       │  OpenAI text-embedding-3-small + FAISS (377 assessments)
│    retrieve.py      │  → top-K candidates + type-boosted supplemental search
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Duration Pre-Filter│  Soft filter — removes clear duration violators
│  (in evaluate.py /  │  before the reranker wastes budget on them
│   api/main.py)      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│     Reranker        │  cross-encoder/ms-marco-MiniLM-L-6-v2
│    reranker.py      │  → precision-scored candidates
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Context Ranker    │  OpenAI gpt-4o-mini (optional)
│  context_ranker.py  │  → per-assessment relevance reason
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Context Builder    │  Balance logic for multi-domain queries
│ context_builder.py  │  → final top-10 recommendations
└─────────────────────┘
```

---

## Project Structure

```
Assesment Recommendation/
├── api/
│   └── main.py                # FastAPI endpoints: /recommend, /health
│
├── pipeline/
│   ├── ingest.py              # Load & clean raw assessment data, build composite_text
│   ├── embed.py               # Generate OpenAI embeddings + build FAISS index
│   ├── retrieve.py            # FAISS semantic search + URL text extraction
│   ├── query_expander.py      # LLM query understanding (types, duration, expansion)
│   ├── reranker.py            # Cross-encoder reranking
│   ├── context_ranker.py      # LLM per-assessment relevance reasoning
│   └── context_builder.py     # Balance logic, filtering, final output formatting
│
├── scraper/
│   ├── scrape_shl.py          # Primary SHL catalogue scraper (377 assessments)
│   ├── scrape_missing.py      # Fill in missing descriptions
│   └── enrich_descriptions.py # Enrich short descriptions via web content
│
├── evaluation/
│   ├── evaluate.py            # Mean Recall@K evaluation + test-set predictions
│   ├── train.csv              # 10 labelled queries for iteration
│   ├── test.csv               # 9 unlabelled queries for final submission
│   └── results.json           # Evaluation output (per-query recall)
│
├── data/
│   ├── shl_assessments.json   # Raw scraped assessment catalogue (377 items)
│   ├── predictions.csv        # Final predictions on test set (submission file)
│   └── vector_store/
│       ├── index.faiss        # FAISS vector index
│       └── metadata.json      # Assessment metadata aligned to index
│
├── frontend/
│   └── app.py                 # Streamlit web UI
│
├── Dockerfile                 # For Hugging Face Spaces (Docker SDK) API deployment
└── requirements.txt
```

---

## Key Technical Decisions

| Component | Choice | Rationale |
|---|---|---|
| Embedding model | `text-embedding-3-small` (OpenAI) | Best recall/cost trade-off for domain-specific HR text |
| Vector DB | FAISS (flat IP, L2-normalised) | Fast exact search over 377 docs; no infra overhead |
| Query understanding | `gpt-4o-mini` | Extracts SHL test types, duration constraints, and generates richer semantic queries |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder precision scoring on the full candidate pool |
| Context ranker | `gpt-4o-mini` | Reads JD + each assessment description → per-item relevance reason |
| Balance logic | Slot-based per type | Ensures multi-domain queries get a mix of K, P, A types |

---

## Setup & Running Locally

### 1. Clone and install
```bash
git clone <your-repo-url>
cd "Assesment Recommendation"
pip install -r requirements.txt
```

### 2. Set environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 3. Build the FAISS index
```bash
python pipeline/embed.py
```

### 4. Start the API
```bash
uvicorn api.main:app --port 8000 --reload
```

### 5. Start the frontend (separate terminal)
```bash
streamlit run frontend/app.py
```

Then open `http://localhost:8501`.

---

## API Reference

### `POST /recommend`

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I am hiring Java developers who collaborate with business teams",
    "top_k": 10
  }'
```

**Request fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Job description text or URL |
| `top_k` | int | 10 | Number of results (1–25) |
| `remote_only` | bool | false | Only remote-testing enabled assessments |
| `adaptive_only` | bool | false | Only adaptive/IRT assessments |
| `max_duration` | int\|null | null | Max assessment duration in minutes |
| `test_types` | list\|null | null | Filter by type codes: A, B, C, K, P, S, E |
| `use_context_ranker` | bool | true | Use LLM to reason over descriptions |

**Response:**
```json
{
  "query": "...",
  "expanded_query": "...",
  "llm_reasoning": "...",
  "total_returned": 10,
  "recommendations": [
    {
      "name": "Core Java (Entry Level)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
      "description": "...",
      "test_type": "K",
      "test_type_label": "Knowledge & Skills",
      "job_levels": ["Entry Level"],
      "duration_mins": 15,
      "remote_testing": true,
      "adaptive_irt": false,
      "reason": "..."
    }
  ]
}
```

### `GET /health`
Returns API status, number of indexed assessments, and whether the LLM is enabled.

---

## Evaluation

Run evaluation against the labelled train set (10 queries):
```bash
python evaluation/evaluate.py --k 10
```

Generate predictions on the unlabelled test set:
```bash
python evaluation/evaluate.py --k 10   # also generates data/predictions.csv
```

**Current performance:**

| Metric | Score |
|---|---|
| Mean Recall@10 | **0.3400** |
| Best single query | 0.8333 |

---

| Code | Category |
|---|---|
| A | Ability & Aptitude |
| B | Biodata / Situational Judgement |
| C | Competencies |
| E | Assessment Exercises |
| K | Knowledge & Skills |
| P | Personality & Behaviour |
| S | Simulations |

---

## Data Sources

- **Assessment catalogue:** Scraped from [SHL Product Catalog](https://www.shl.com/solutions/products/product-catalog/) — 377 Individual Test Solutions
- **Train set:** 10 human-labelled queries with relevant assessment URLs
- **Test set:** 9 unlabelled queries for final evaluation
