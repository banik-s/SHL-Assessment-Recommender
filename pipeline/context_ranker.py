import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4o-mini"

# How many candidates to send to LLM (cap to avoid huge prompts)
MAX_CANDIDATES_FOR_LLM = 15



def _build_prompt(job_description: str, candidates: list[dict]) -> str:
    
    candidate_lines = []
    for i, c in enumerate(candidates, 1):
        name  = c.get("name", "")
        desc  = c.get("description", "").strip()
        ttype = c.get("test_type", "")
        levels = ", ".join(c.get("job_levels") or [])
        dur   = c.get("duration_mins")

        meta_parts = []
        if ttype:
            meta_parts.append(f"Type: {ttype}")
        if levels:
            meta_parts.append(f"Levels: {levels}")
        if dur:
            meta_parts.append(f"Duration: {dur} mins")
        meta = " | ".join(meta_parts)

        candidate_lines.append(
            f"{i}. **{name}**\n"
            f"   Meta: {meta}\n"
            f"   Description: {desc if desc else 'No description available.'}"
        )

    candidates_block = "\n\n".join(candidate_lines)

    prompt = f"""You are a senior HR assessment consultant specialising in SHL psychometric tools.

---
JOB DESCRIPTION / QUERY:
{job_description}

---
CANDIDATE SHL ASSESSMENTS:

{candidates_block}

---
TASK:
For each assessment above, score its relevance to the job description on a scale of 0-10,
and provide a single concise sentence explaining WHY it is or isn't relevant.

Return ONLY valid JSON (no markdown, no code fences):
{{
  "ranked": [
    {{
      "name": "<exact assessment name from above>",
      "relevance_score": <0-10 integer or float>,
      "reason": "<one sentence explanation>"
    }},
    ...
  ]
}}

Score all {len(candidates)} assessments. Use the exact name string as given above.
Higher score = more relevant to the JD. Be strict — only score high when the description clearly matches the JD requirements."""

    return prompt


def _call_openai_context(job_description: str, candidates: list[dict]) -> list[dict]:
    """
    Call OpenAI gpt-4o-mini with the JD + all candidate descriptions.
    Returns list of {name, relevance_score, reason}.
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = _build_prompt(job_description, candidates)

    response = client.chat.completions.create(
        model       = OPENAI_MODEL,
        temperature = 0.1,
        max_tokens  = 2048,
        messages    = [
            {"role": "system", "content": "You are a senior HR assessment consultant specialising in SHL psychometric tools. Always respond with valid JSON only."},
            {"role": "user",   "content": prompt},
        ],
        response_format = {"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    parsed = json.loads(raw)
    return parsed.get("ranked", [])



def rank_with_context(
    job_description: str,
    candidates:      list[dict],
    top_k:           int = 10,
) -> tuple[list[dict], bool]:
    
    # Add empty reason as default so downstream code always has the field
    for c in candidates:
        c.setdefault("reason", "")
        c.setdefault("context_score", c.get("rerank_score", c.get("score", 0)))

    if not OPENAI_API_KEY:
        print("[ContextRanker] No OPENAI_API_KEY — skipping LLM context ranking.")
        return candidates, False

    # Cap how many we send to avoid huge prompts
    to_rank = candidates[:MAX_CANDIDATES_FOR_LLM]

    try:
        llm_ranked = _call_openai_context(job_description, to_rank)

        # Build a lookup from name → {score, reason}
        name_to_llm: dict[str, dict] = {}
        for item in llm_ranked:
            name = (item.get("name") or "").strip()
            if name:
                name_to_llm[name] = {
                    "context_score": float(item.get("relevance_score", 0)),
                    "reason":        str(item.get("reason", "")).strip(),
                }

        # Merge LLM scores back into candidates
        for c in candidates:
            cname = c.get("name", "").strip()
            if cname in name_to_llm:
                c["context_score"] = name_to_llm[cname]["context_score"]
                c["reason"]        = name_to_llm[cname]["reason"]

        # Sort by LLM context score (descending), keep the ones OpenAI didn't rank last
        ranked = sorted(
            candidates,
            key=lambda x: x.get("context_score", 0),
            reverse=True,
        )

        print(f"[ContextRanker] OpenAI context-ranked {len(llm_ranked)} candidates.")
        return ranked, True

    except Exception as e:
        print(f"[ContextRanker] OpenAI call failed ({e}) — using reranker order.")
        return candidates, False


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from pipeline.retrieve      import Retriever, resolve_query
    from pipeline.reranker      import Reranker
    from pipeline.query_expander import expand_query

    retriever = Retriever()
    reranker  = Reranker()

    TEST_JD = (
        "I am hiring Java developers who should also be able to "
        "collaborate effectively with business teams and communicate clearly."
    )

    print(f"\nJob Description: {TEST_JD}\n")

    exp        = expand_query(TEST_JD)
    resolved   = resolve_query(exp["expanded_query"])
    candidates = retriever.search(exp["expanded_query"], top_k=10, expand=True)
    reranked   = reranker.rerank(resolved, candidates, top_k=10)

    context_ranked, llm_used = rank_with_context(TEST_JD, reranked, top_k=10)

    print(f"\nLLM used: {llm_used}")
    print(f"\n{'='*70}")
    print(f"{'Rank':<5} {'Score':>6}  {'Name':<40}  Reason")
    print(f"{'─'*5} {'─'*6}  {'─'*40}  {'─'*30}")
    for rank, r in enumerate(context_ranked[:10], 1):
        score  = r.get("context_score", 0)
        reason = r.get("reason", "")[:60]
        print(f"#{rank:<4} {score:>6.2f}  {r['name']:<40}  {reason}")
