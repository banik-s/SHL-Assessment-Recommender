
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.retrieve        import Retriever, resolve_query
from pipeline.reranker        import Reranker
from pipeline.query_expander  import expand_query
from pipeline.context_builder import build_context

BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
EVAL_DIR     = os.path.dirname(__file__)
TRAIN_CSV    = os.path.join(EVAL_DIR, "train.csv")
TEST_CSV     = os.path.join(EVAL_DIR, "test.csv")
RESULTS_PATH = os.path.join(EVAL_DIR, "results.json")
PRED_PATH    = os.path.join(BASE_DIR, "data", "predictions.csv")

# Type-specific vocabulary hints — mirrors api/main.py's _TYPE_SEARCH_HINTS
_TYPE_SEARCH_HINTS: dict = {
    "A": "verbal reasoning ability aptitude cognitive english language comprehension numerical deductive inductive reasoning test",
    "P": "personality behaviour occupational questionnaire OPQ interpersonal motivation style trait competency",
    "K": "knowledge skills technical domain programming software proficiency certification",
    "C": "competencies leadership management organisational business judgement executive",
    "B": "biodata situational judgement work experience background scenarios",
    "S": "simulation work sample job simulation realistic preview",
    "E": "exercise assessment centre role play group exercise written presentation",
}



def normalise_url(url: str) -> str:
    """
    Extract the assessment slug from the SHL URL for robust matching.

    Both URL forms map to the same slug:
      https://www.shl.com/solutions/products/product-catalog/view/java-8-new/
      https://www.shl.com/products/product-catalog/view/java-8-new/
                                                                   ^^^^^^^^^^^
    """
    from urllib.parse import urlparse
    path = urlparse(str(url).strip().lower()).path.rstrip("/")
    slug = path.split("/")[-1]
    return slug



def load_train(path: str = TRAIN_CSV) -> list[dict]:
    """
    Load train.csv and group by query.
    Returns: [{"query": str, "relevant_urls": [str, ...]}, ...]
    """
    df = pd.read_csv(path, encoding="utf-8")
    # Normalize column names (trim whitespace)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Expected: query, assessment_url
    q_col = [c for c in df.columns if "query" in c][0]
    url_col = [c for c in df.columns if "url" in c or "assessment" in c][0]

    df[q_col]   = df[q_col].astype(str).str.strip()
    df[url_col] = df[url_col].astype(str).str.strip()

    grouped = df.groupby(q_col, sort=False)[url_col].apply(list).reset_index()
    rows = []
    for _, row in grouped.iterrows():
        rows.append({
            "query":         row[q_col],
            "relevant_urls": [normalise_url(u) for u in row[url_col]],
        })
    return rows


def load_test(path: str = TEST_CSV) -> list[str]:
    """Load test.csv — returns list of query strings (single column)."""
    df = pd.read_csv(path, encoding="utf-8")
    return [str(q).strip() for q in df.iloc[:, 0].dropna().tolist()]



def recall_at_k(predicted_urls: list[str], relevant_urls: list[str], k: int) -> float:
    #Recall@K = |relevant ∩ top-K predicted| / |relevant|
    if not relevant_urls:
        return 0.0
    predicted_k  = {normalise_url(u) for u in predicted_urls[:k]}
    relevant_set = set(relevant_urls)   # already normalised
    hits = len(predicted_k & relevant_set)
    return hits / len(relevant_set)


def run_query(
    query:     str,
    retriever: Retriever,
    reranker:  Reranker,
    k:         int,
    skip_llm:  bool = False,
    fetch_factor: int = 3,
) -> list[str]:
    # Step 1 — query expansion
    if skip_llm:
        exp = {
            "expanded_query":    query,
            "required_types":    [],
            "is_multi_domain":   False,
            "max_duration_mins": None,
            "reasoning":         "",
            "llm_used":          False,
        }
    else:
        try:
            exp = expand_query(query)
        except Exception as e:
            print(f"    [QueryExpander] failed ({e}) — using raw query")
            exp = {
                "expanded_query":    query,
                "required_types":    [],
                "is_multi_domain":   False,
                "max_duration_mins": None,
                "reasoning":         "",
                "llm_used":          False,
            }

    search_text       = exp["expanded_query"]
    required_types    = exp.get("required_types", [])
    max_duration_mins = exp.get("max_duration_mins")   # int or None

    # Step 2 — resolve URL → text (passthrough for plain text queries)
    try:
        resolved = resolve_query(search_text)
    except Exception:
        resolved = search_text

    # FAISS retrieval with large headroom so duration filtering still leaves enough
    # candidates (retrieve 10× k to absorb post-filter shrinkage)
    try:
        candidates = retriever.search(search_text, top_k=k * 10, expand=False)
    except Exception as e:
        print(f"    [Retriever] failed ({e})")
        return []

    if not candidates:
        return []

    # Step 2a — type-targeted supplemental FAISS search
    # Trigger for ANY query that has required_types (not just multi-domain)
    # This injects type-specific candidates that pure semantic search may miss.
    if required_types:
        seen_names = {c["name"] for c in candidates}
        for t in required_types:
            hint = _TYPE_SEARCH_HINTS.get(t, "")
            if not hint:
                continue
            sub_query = f"{hint} | {search_text[:300]}"
            try:
                extra = retriever.search(sub_query, top_k=k * 3, expand=False)
                for item in extra:
                    if item["name"] not in seen_names:
                        item["score"] = round(item["score"] * 0.9, 4)
                        candidates.append(item)
                        seen_names.add(item["name"])
            except Exception:
                pass

    # Step 2b — Soft duration pre-filter: remove clear violators BEFORE reranking
    # so the cross-encoder doesn't waste budget scoring out-of-range assessments.
    # "Soft" = keep candidates with unknown duration (duration_mins is None).
    if max_duration_mins is not None:
        filtered_dur = [
            c for c in candidates
            if c.get("duration_mins") is None or c["duration_mins"] <= max_duration_mins
        ]
        # Safety net: if filter is too aggressive, relax it by 50% (e.g. 40 min → 60 min)
        if len(filtered_dur) >= k:
            candidates = filtered_dur
        else:
            relaxed = max_duration_mins * 1.5
            candidates = [
                c for c in candidates
                if c.get("duration_mins") is None or c["duration_mins"] <= relaxed
            ] or candidates   # last resort: unfiltered

    # Step 3 — cross-encoder reranking on FULL candidate pool
    try:
        reranked = reranker.rerank(resolved, candidates, top_k=len(candidates))
    except Exception as e:
        print(f"    [Reranker] failed ({e}) — using FAISS order")
        reranked = candidates

    # Step 4 — balance + build output with duration filter applied cleanly
    context = build_context(
        query            = query,
        candidates       = reranked,
        top_k            = k,
        required_types   = required_types,
        is_multi_domain  = exp.get("is_multi_domain", False),
        max_duration     = max_duration_mins,
        expanded_query   = search_text,
        llm_reasoning    = exp.get("reasoning", ""),
        llm_context_used = False,
    )
    return [r["url"] for r in context["recommendations"]]



def run_evaluation(
    retriever: Retriever,
    reranker:  Reranker,
    k:         int,
    skip_llm:  bool = False,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  EVALUATION  —  Mean Recall@{k}")
    print(f"{'='*60}\n")

    train  = load_train()
    print(f"Train queries : {len(train)}")
    print(f"K             : {k}")
    print(f"LLM (OpenAI)  : {'disabled (--skip-llm)' if skip_llm else 'enabled'}\n")

    results = []
    recalls = []

    for i, row in enumerate(train, 1):
        t0        = time.time()
        pred_urls = run_query(row["query"], retriever, reranker, k, skip_llm=skip_llm)
        recall    = recall_at_k(pred_urls, row["relevant_urls"], k)
        recalls.append(recall)

        hits = int(round(recall * len(row["relevant_urls"])))
        print(f"[{i:>2}/{len(train)}] Recall@{k} = {recall:.3f}  "
              f"hits={hits}/{len(row['relevant_urls'])}  "
              f"{round(time.time()-t0, 1)}s")
        print(f"         Query: {row['query'][:90]}")

        results.append({
            "query":          row["query"],
            "recall_at_k":    round(recall, 4),
            "predicted_urls": pred_urls,
            "relevant_urls":  row["relevant_urls"],
        })

    mean_r = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"\n{'─'*60}")
    print(f"  Mean Recall@{k} : {mean_r:.4f}")
    print(f"  Min            : {min(recalls):.4f}")
    print(f"  Max            : {max(recalls):.4f}")
    print(f"{'─'*60}\n")

    output = {
        "mean_recall_at_k": round(mean_r, 4),
        "k":                k,
        "n_queries":        len(train),
        "per_query":        results,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {RESULTS_PATH}")
    return output



def run_predictions(
    retriever: Retriever,
    reranker:  Reranker,
    k:         int,
    skip_llm:  bool = False,
) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  PREDICTIONS  —  Test-Set ({TEST_CSV})")
    print(f"{'='*60}\n")

    test_queries = load_test()
    print(f"Test queries  : {len(test_queries)}")
    print(f"LLM (OpenAI)  : {'disabled (--skip-llm)' if skip_llm else 'enabled'}\n")

    rows = []
    for i, q in enumerate(test_queries, 1):
        t0        = time.time()
        pred_urls = run_query(q, retriever, reranker, k, skip_llm=skip_llm)
        elapsed   = round(time.time() - t0, 1)
        print(f"[{i:>2}/{len(test_queries)}] {q[:80]}  ({elapsed}s)")
        rows.append({
            "query":       q,
            "predictions": ",".join(pred_urls),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(PRED_PATH), exist_ok=True)
    df.to_csv(PRED_PATH, index=False, encoding="utf-8")
    print(f"\nPredictions saved to {PRED_PATH}")
    print(df[["query", "predictions"]].to_string(max_colwidth=100))
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate + generate predictions")
    parser.add_argument("--k",         type=int,  default=10,  help="K for Recall@K (default: 10)")
    parser.add_argument("--eval-only", action="store_true",    help="Skip test predictions")
    parser.add_argument("--pred-only", action="store_true",    help="Skip train evaluation")
    parser.add_argument("--skip-llm",  action="store_true",    help="Disable OpenAI (fast, no API cost)")
    args = parser.parse_args()

    print("Loading pipeline models …")
    retriever = Retriever()
    reranker  = Reranker()
    print("Models ready.\n")

    if not args.pred_only:
        run_evaluation(retriever, reranker, k=args.k, skip_llm=args.skip_llm)

    if not args.eval_only:
        run_predictions(retriever, reranker, k=args.k, skip_llm=args.skip_llm)
