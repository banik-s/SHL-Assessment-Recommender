# ── Test type code → full label ────────────────────────────────────────────────
TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

_INTERNAL_FIELDS = {"score", "rerank_score"}



def _slug(url: str) -> str:
    """Extract the final path segment as a stable dedup key."""
    return url.rstrip("/").split("/")[-1].lower()


def balance_results(
    candidates:     list[dict],
    required_types: list[str],
    top_k:          int,
) -> list[dict]:
    """
    Ensure the final results contain a representative mix when the query
    spans multiple SHL test type domains (e.g. A + P for verbal + personality).

    Strategy:
      - Deduplicate by URL slug first (catches same assessment with different URL forms)
      - Reserve floor(top_k / len(required_types)) slots per required type
        pulling from the FULL candidate pool (not just top_k) so type-specific
        assessments aren't lost due to cross-encoder score ordering
      - Fill remaining slots with highest-scored candidates from any type
      - Final sort by rerank_score / context_score

    Args:
        candidates:     ALL reranked candidates (sorted descending), deduplication applied here
        required_types: list of type codes LLM identified (e.g. ["A", "P"])
        top_k:          total results to return
    """
    # ── Deduplicate by URL slug (catches both name AND URL duplicates) ──────────
    seen_slugs: set = set()
    deduped = []
    for c in candidates:
        s = _slug(c.get("url", c.get("name", "")))
        if s not in seen_slugs:
            seen_slugs.add(s)
            deduped.append(c)

    if not required_types or len(required_types) < 2:
        # Single-type or no type — just return the top deduped candidates
        return deduped[:top_k]

    # Bucket ALL deduplicated candidates by test_type
    buckets: dict[str, list[dict]] = {t: [] for t in required_types}
    overflow: list[dict] = []

    for c in deduped:
        t = c.get("test_type", "").upper()
        if t in buckets:
            buckets[t].append(c)
        else:
            overflow.append(c)

    # Slots per type — at least 2, but proportional
    per_type = max(2, top_k // len(required_types))

    selected = []
    selected_slugs: set = set()
    for t in required_types:
        for item in buckets[t][:per_type]:
            s = _slug(item.get("url", item.get("name", "")))
            if s not in selected_slugs:
                selected.append(item)
                selected_slugs.add(s)

    # Fill remaining slots greedily from highest-scored remainder
    remaining_budget = top_k - len(selected)
    if remaining_budget > 0:
        rest = [c for c in deduped if _slug(c.get("url", c.get("name", ""))) not in selected_slugs]
        rest.sort(key=lambda x: x.get("context_score", x.get("rerank_score", x.get("score", 0))), reverse=True)
        selected.extend(rest[:remaining_budget])

    # Final sort by best available score
    selected.sort(
        key=lambda x: x.get("context_score", x.get("rerank_score", x.get("score", 0))),
        reverse=True,
    )
    return selected[:top_k]


def apply_filters(
    candidates:    list[dict],
    remote_only:   bool        = False,
    adaptive_only: bool        = False,
    max_duration:  int | None  = None,
    test_types:    list[str]   = None,
) -> list[dict]:
    """Filter candidates by metadata constraints."""
    result = candidates
    if remote_only:
        result = [c for c in result if c.get("remote_testing")]
    if adaptive_only:
        result = [c for c in result if c.get("adaptive_irt")]
    if max_duration is not None:
        result = [
            c for c in result
            if c.get("duration_mins") is None or c["duration_mins"] <= max_duration
        ]
    if test_types:
        allowed = {t.upper() for t in test_types}
        result  = [c for c in result if c.get("test_type", "").upper() in allowed]
    return result


def build_context(
    query:          str,
    candidates:     list[dict],
    top_k:          int         = 10,
    # balance
    required_types: list[str]   = None,
    is_multi_domain: bool       = False,
    # hard filters
    remote_only:    bool        = False,
    adaptive_only:  bool        = False,
    max_duration:   int | None  = None,
    test_types:     list[str]   = None,
    # llm metadata passthrough
    expanded_query: str         = "",
    llm_reasoning:  str         = "",
    llm_context_used: bool      = False,
) -> dict:
    """
    Format the final recommendation response.

    Returns:
        {
            "query": str,
            "expanded_query": str,
            "llm_reasoning": str,
            "llm_context_used": bool,
            "total_returned": int,
            "recommendations": [ { ...assessment fields... , "reason": str } ]
        }
    """
    # Step 1 — apply balance if multi-domain query
    if is_multi_domain and required_types and len(required_types) >= 2:
        balanced = balance_results(candidates, required_types, top_k)
    else:
        balanced = candidates[:top_k]

    # Step 2 — apply hard filters
    filtered = apply_filters(
        balanced,
        remote_only=remote_only,
        adaptive_only=adaptive_only,
        max_duration=max_duration,
        test_types=test_types,
    )

    top = filtered[:top_k]

    # Step 3 — build clean output records
    recommendations = []
    for item in top:
        rec = {
            "name":             item.get("name", ""),
            "url":              item.get("url", ""),
            "description":      item.get("description", ""),
            "test_type":        item.get("test_type", ""),
            "test_type_label":  TEST_TYPE_LABELS.get(item.get("test_type", ""), ""),
            "job_levels":       item.get("job_levels") or [],
            "duration_mins":    item.get("duration_mins"),
            "remote_testing":   item.get("remote_testing", False),
            "adaptive_irt":     item.get("adaptive_irt", False),
            # LLM context ranker output — empty string when LLM not used
            "reason":           item.get("reason", ""),
        }
        recommendations.append(rec)

    return {
        "query":             query,
        "expanded_query":    expanded_query or query,
        "llm_reasoning":     llm_reasoning,
        "llm_context_used":  llm_context_used,
        "total_returned":    len(recommendations),
        "recommendations":   recommendations,
    }



if __name__ == "__main__":
    import os, sys, json
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from pipeline.retrieve   import Retriever, resolve_query
    from pipeline.reranker   import Reranker
    from pipeline.query_expander import expand_query

    retriever = Retriever()
    reranker  = Reranker()

    q       = "I am hiring for Java developers who can also collaborate effectively with my business teams."
    exp     = expand_query(q)
    print(f"\nExpanded: {exp['expanded_query'][:100]}")
    print(f"Types   : {exp['required_types']}  multi={exp['is_multi_domain']}")

    resolved   = resolve_query(exp["expanded_query"])
    candidates = retriever.search(exp["expanded_query"], top_k=10, expand=True)
    reranked   = reranker.rerank(resolved, candidates, top_k=10)

    context = build_context(
        query           = q,
        candidates      = reranked,
        top_k           = 10,
        required_types  = exp["required_types"],
        is_multi_domain = exp["is_multi_domain"],
        expanded_query  = exp["expanded_query"],
        llm_reasoning   = exp["reasoning"],
    )

    print(f"\nQuery: {context['query']}")
    print(f"Returned: {context['total_returned']}")
    for r in context["recommendations"]:
        print(f"  [{r['test_type']}] {r['name']}")
