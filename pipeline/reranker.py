import os
from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 10


def _build_candidate_text(candidate: dict) -> str:
    from pipeline.ingest import TEST_TYPE_MAP   # reuse the type label map

    name      = candidate.get("name", "")
    desc      = candidate.get("description", "")
    test_type = candidate.get("test_type", "")
    levels    = candidate.get("job_levels") or []
    duration  = candidate.get("duration_mins")
    remote    = candidate.get("remote_testing", False)
    adaptive  = candidate.get("adaptive_irt", False)

    parts = [name]
    type_label = TEST_TYPE_MAP.get(test_type, "")
    if type_label:
        parts.append(f"Type: {type_label}")
    if levels:
        parts.append(f"Levels: {', '.join(levels)}")
    if duration:
        parts.append(f"Duration: {duration} mins")
    flags = []
    if remote:
        flags.append("Remote Testing")
    if adaptive:
        flags.append("Adaptive/IRT")
    if flags:
        parts.append(", ".join(flags))
    if desc:
        parts.append(f"Measures: {desc}")

    return " | ".join(parts)


class Reranker:


    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        print(f"[Reranker] Loading cross-encoder '{model_name}' …")
        self.model = CrossEncoder(model_name, max_length=512)
        print("[Reranker] Ready.")

    def rerank(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """
        Score every (query, candidate_text) pair and return the top-K
        sorted by cross-encoder score descending.

        Args:
            query:      resolved query text (already URL-extracted if needed)
            candidates: list of dicts from retrieve.search()
            top_k:      how many to return after reranking

        Returns:
            top_k dicts, each with an added 'rerank_score' field.
        """
        if not candidates:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [
            (query, _build_candidate_text(c))
            for c in candidates
        ]

        # Score all pairs in one batch
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = round(float(score), 4)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        # Only truncate if caller explicitly wants fewer than all candidates
        if top_k < len(ranked):
            return ranked[:top_k]
        return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from pipeline.retrieve import Retriever, resolve_query

    retriever = Retriever()
    reranker  = Reranker()

    QUERIES = [
        "cognitive ability test for software engineer hiring",
        "personality and behaviour assessment for a senior sales manager role",
        "entry level data entry clerk with numerical accuracy",
        "Python developer test intermediate level",
    ]

    for q in QUERIES:
        print(f"\n{'═'*60}")
        print(f"Query: {q}")
        print(f"{'─'*60}")

        resolved   = resolve_query(q)
        candidates = retriever.search(q, top_k=10, expand=True)   # fetch top-20
        reranked   = reranker.rerank(resolved, candidates, top_k=5)

        print(f"{'Rank':<5}{'Score':>8}   {'Name'}")
        print(f"{'─'*5}{'─'*8}   {'─'*40}")
        for rank, r in enumerate(reranked, 1):
            print(f"#{rank:<4}{r['rerank_score']:>8.4f}   {r['name']}")
