
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.retrieve        import Retriever, resolve_query
from pipeline.reranker        import Reranker
from pipeline.query_expander  import expand_query
from pipeline.context_ranker  import rank_with_context
from pipeline.context_builder import build_context



_TYPE_SEARCH_HINTS: dict = {
    "A": "verbal reasoning ability aptitude cognitive english language comprehension numerical deductive inductive reasoning test",
    "P": "personality behaviour occupational questionnaire OPQ interpersonal motivation style trait competency",
    "K": "knowledge skills technical domain programming software proficiency certification",
    "C": "competencies leadership management organisational business judgement executive",
    "B": "biodata situational judgement work experience background scenarios",
    "S": "simulation work sample job simulation realistic preview",
    "E": "exercise assessment centre role play group exercise written presentation",
}


app = FastAPI(
    title       = "SHL Assessment Recommender",
    description = (
        "Semantic search + LLM query understanding over the full SHL product catalog. "
        "Accepts a plain-text job description or a job-posting URL "
        "and returns the most relevant SHL assessments."
    ),
    version = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


@app.on_event("startup")
async def load_models():
    print("[startup] Loading Retriever …")
    app.state.retriever = Retriever()

    print("[startup] Loading Reranker …")
    app.state.reranker  = Reranker()

    app.state.started_at = time.time()
    openai_key = os.getenv("OPENAI_API_KEY", "")
    print(f"[startup] OpenAI LLM: {'enabled' if openai_key else 'disabled (no API key)'}")
    print("[startup] Ready to serve requests.")



class RecommendRequest(BaseModel):
    query: str = Field(
        ...,
        min_length  = 3,
        description = "Job description text or a URL to a job posting page.",
        examples    = [
            "Looking for a cognitive ability test for a software engineer",
            "https://example.com/jobs/python-developer",
        ],
    )
    top_k: int = Field(
        default     = 10,
        ge          = 1,
        le          = 25,
        description = "Number of recommendations to return (1–25).",
    )
    # Optional hard filters
    remote_only:   bool             = Field(False, description="Only remote-testing enabled assessments.")
    adaptive_only: bool             = Field(False, description="Only adaptive/IRT assessments.")
    max_duration:  int | None       = Field(None,  ge=1, description="Max duration in minutes.")
    test_types:    list[str] | None = Field(None,  description="Filter by type codes: A,B,C,D,E,K,P,S.")
    # Context ranker toggle — when True (default), OpenAI reads descriptions for per-assessment reasons
    use_context_ranker: bool        = Field(True,  description="Use OpenAI gpt-4o-mini to reason over assessment descriptions.")


class AssessmentOut(BaseModel):
    name:            str
    url:             str
    description:     str
    test_type:       str
    test_type_label: str
    job_levels:      list[str]
    duration_mins:   int | None
    remote_testing:  bool
    adaptive_irt:    bool
    reason:          str = ""   # LLM context ranker relevance reason (empty if LLM not used)


class RecommendResponse(BaseModel):
    query:             str
    expanded_query:    str
    llm_reasoning:     str
    llm_context_used:  bool = False
    total_returned:    int
    recommendations:   list[AssessmentOut]



@app.get("/health", tags=["System"])
def health():
    """Liveness check — confirms models are loaded and index is ready."""
    retriever: Retriever = app.state.retriever
    return {
        "status":        "ok",
        "indexed_docs":  retriever.index.ntotal,
        "uptime_seconds": round(time.time() - app.state.started_at, 1),
        "llm_enabled":   bool(os.getenv("OPENAI_API_KEY", "")),
    }


@app.post(
    "/recommend",
    response_model = RecommendResponse,
    tags           = ["Recommendations"],
    summary        = "Get SHL assessment recommendations",
)
def recommend(req: RecommendRequest):
    #Main recommendation endpoint.

    retriever: Retriever = app.state.retriever
    reranker:  Reranker  = app.state.reranker

    # Step 1 — LLM query understanding
    try:
        exp = expand_query(req.query)
    except Exception as e:
        exp = {
            "expanded_query":    req.query,
            "required_types":    [],
            "is_multi_domain":   False,
            "max_duration_mins": None,
            "reasoning":         "",
            "llm_used":          False,
        }

    search_text    = exp["expanded_query"]
    required_types = exp.get("required_types", [])

    # If user didn't specify max_duration, use the LLM-extracted duration from the query text
    effective_max_duration = req.max_duration
    if effective_max_duration is None and exp.get("max_duration_mins"):
        effective_max_duration = exp["max_duration_mins"]

    # Step 2 — resolve URL → text if needed
    try:
        resolved = resolve_query(search_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to resolve query: {e}")

    if not resolved or len(resolved.strip()) < 3:
        raise HTTPException(status_code=400, detail="Could not extract usable text.")

    # Step 3 — FAISS retrieval (10× headroom to absorb duration filter shrinkage)
    FETCH_FACTOR = 10
    try:
        candidates = retriever.search(search_text, top_k=req.top_k * FETCH_FACTOR, expand=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    if not candidates:
        raise HTTPException(status_code=404, detail="No assessments found. Try a different query.")

    # Step 3a — supplemental type-targeted searches for any query with detected types
    # This ensures type-specific assessments appear even when JD language dominates
    if required_types:
        seen_names = {c["name"] for c in candidates}
        for t in required_types:
            hint = _TYPE_SEARCH_HINTS.get(t, "")
            if not hint:
                continue
            sub_query = f"{hint} | {search_text[:300]}"
            try:
                extra = retriever.search(sub_query, top_k=req.top_k * 3, expand=False)
                for item in extra:
                    if item["name"] not in seen_names:
                        item["score"] = round(item["score"] * 0.9, 4)
                        candidates.append(item)
                        seen_names.add(item["name"])
            except Exception:
                pass
        print(f"[API] Type-boosted pool size: {len(candidates)} candidates for types {required_types}")

    # Step 3b — Soft duration pre-filter before reranking
    if effective_max_duration is not None:
        filtered_dur = [
            c for c in candidates
            if c.get("duration_mins") is None or c["duration_mins"] <= effective_max_duration
        ]
        if len(filtered_dur) >= req.top_k:
            candidates = filtered_dur
        else:
            relaxed = effective_max_duration * 1.5
            candidates = [
                c for c in candidates
                if c.get("duration_mins") is None or c["duration_mins"] <= relaxed
            ] or candidates

    # Step 3 — cross-encoder reranking
    # Score ALL candidates, don't truncate yet — balance_results needs the full pool
    try:
        reranked = reranker.rerank(resolved, candidates, top_k=len(candidates))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking error: {e}")

    # Step 3.5 — OpenAI gpt-4o-mini context ranker: reads JD + descriptions, returns reason per assessment
    context_used = False
    try:
        if req.use_context_ranker:
            reranked, context_used = rank_with_context(
                job_description = req.query,
                candidates      = reranked[:req.top_k * 2],   # send top 2× to LLM for ranking, rest kept below
                top_k           = req.top_k,
            )
            # Re-merge: context-ranked top items + remaining unranked items
            reranked_names = {r["name"] for r in reranked}
            reranked = reranked + [c for c in candidates if c["name"] not in reranked_names]
    except Exception as e:
        print(f"[API] Context ranker failed ({e}) — continuing without it.")

    # Step 4 — balance + format
    context = build_context(
        query             = req.query,
        candidates        = reranked,
        top_k             = req.top_k,
        required_types    = exp.get("required_types", []),
        is_multi_domain   = exp.get("is_multi_domain", False),
        expanded_query    = exp.get("expanded_query", req.query),
        llm_reasoning     = exp.get("reasoning", ""),
        llm_context_used  = context_used,
        remote_only       = req.remote_only,
        adaptive_only     = req.adaptive_only,
        max_duration      = effective_max_duration,
        test_types        = req.test_types,
    )

    return context
