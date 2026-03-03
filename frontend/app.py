import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


st.set_page_config(
    page_title = "SHL Assessment Recommender",
    page_icon  = "SHL",
    layout     = "wide",
)


st.markdown("""
<style>
  .stApp { background-color: #0f1117; color: #e0e0e0; }

  .main-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .subtitle { color: #888; font-size: 1rem; margin-bottom: 1.5rem; }

  .card {
    background: #1c1f2a; border-radius: 10px; padding: 16px 20px;
    margin-bottom: 12px; border-left: 4px solid #4f8ef7;
    transition: transform 0.15s;
  }
  .card:hover { transform: translateX(4px); }
  .card-title { font-size: 1.05rem; font-weight: 700; color: #e8e8f0; }
  .card-meta  { font-size: 0.85rem; color: #9090a0; margin-top: 4px; }
  .tag {
    display: inline-block; border-radius: 12px; padding: 2px 10px;
    font-size: 0.75rem; font-weight: 600; margin-right: 6px; margin-top: 4px;
  }
  .tag-K { background: #1d3557; color: #4f8ef7; }
  .tag-P { background: #2d1b4e; color: #a78bfa; }
  .tag-A { background: #1a3a2a; color: #4caf86; }
  .tag-C { background: #3b2a1a; color: #f7975f; }
  .tag-S { background: #2a1a2d; color: #f06292; }
  .tag-B { background: #1a2a2d; color: #4dd0e1; }
  .tag-E { background: #2d2a1a; color: #ffd54f; }
  .tag-default { background: #1f1f2e; color: #aaa; }
  .chip {
    display: inline-block; background: #252836; border-radius: 6px;
    padding: 2px 8px; font-size: 0.78rem; color: #aaa; margin-right: 5px;
  }
  .insight-box {
    background: #16192a; border: 1px solid #2c2f45; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 20px; font-size: 0.9rem; color: #b0b8d0;
  }
  /* Per-assessment LLM reason pill */
  .reason-box {
    background: #0f1f0f; border: 1px solid #1e4d2b; border-radius: 7px;
    padding: 7px 12px; margin-top: 9px; font-size: 0.82rem;
    color: #6ddb95; line-height: 1.4;
  }
  .reason-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
    color: #3a9f5c; text-transform: uppercase; margin-bottom: 3px;
  }
  /* Context ranker active badge */
  .ctx-badge {
    display: inline-block; background: #0f2a1a;
    border: 1px solid #2a6b3f; border-radius: 12px;
    padding: 2px 10px; font-size: 0.75rem; color: #4cdb86;
    margin-left: 10px; vertical-align: middle;
  }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">SHL Assessment Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Powered by OpenAI gpt-4o-mini + Semantic Search + Cross-encoder Reranking</div>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Search Settings")
    top_k         = st.slider("Max results", 5, 10, 10)
    remote_only   = st.checkbox("Remote testing only")
    adaptive_only = st.checkbox("Adaptive/IRT only")
    max_dur_check = st.checkbox("Limit duration")
    max_duration  = st.number_input("Max duration (mins)", 5, 120, 60) if max_dur_check else None
    use_ctx       = st.checkbox("Context Ranker (OpenAI reads descriptions)", value=True,
                                help="When on, OpenAI gpt-4o-mini reads each candidate's description against your JD and explains why it's relevant.")

    st.markdown("---")
    st.caption("**Test type codes**")
    st.caption("K = Knowledge & Skills")
    st.caption("P = Personality & Behavior")
    st.caption("A = Ability & Aptitude")
    st.caption("C = Competencies")
    st.caption("S = Simulations")
    st.caption("B = Biodata / SJT")
    st.caption("E = Assessment Exercises")

    st.markdown("---")
    try:
        h = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success(f"API online - {h['indexed_docs']} assessments indexed")
        if h.get("llm_enabled"):
            st.success("OpenAI LLM enabled")
        else:
            st.warning("OpenAI LLM disabled (no API key)")
    except Exception:
        st.error("API offline - start uvicorn")



SAMPLES = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
    "I need a cognitive ability and personality test for an analyst role.",
    "Entry level customer service representative for a retail contact center.",
    "Senior sales manager with strong leadership and communication skills.",
]

with st.expander("Sample queries - click to use"):
    for s in SAMPLES:
        if st.button(s, key=s):
            st.session_state["query_input"] = s


query = st.text_area(
    "Enter a job description, query, or URL:",
    value       = st.session_state.get("query_input", ""),
    height      = 120,
    placeholder = "E.g. 'I am hiring Java developers who collaborate with business teams' or paste a JD URL",
    key         = "query_text",
)

col1, col2 = st.columns([1, 5])
with col1:
    search_clicked = st.button("Search", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=False):
        st.session_state["query_input"] = ""
        st.rerun()



def tag_html(test_type: str, label: str) -> str:
    cls = f"tag-{test_type}" if test_type in "KPACBSE" else "tag-default"
    return f'<span class="tag {cls}">{test_type} - {label}</span>'


def chip_html(text: str) -> str:
    return f'<span class="chip">{text}</span>'


if search_clicked and query.strip():
    payload = {
        "query":              query.strip(),
        "top_k":              top_k,
        "remote_only":        remote_only,
        "adaptive_only":      adaptive_only,
        "max_duration":       max_duration,
        "use_context_ranker": use_ctx,
    }

    with st.spinner("Searching across 377 SHL assessments ..."):
        try:
            resp = requests.post(f"{API_BASE}/recommend", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Run: uvicorn api.main:app --port 8000")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    recs = data.get("recommendations", [])

    # LLM insight box (query expansion reasoning)
    llm_reasoning = data.get("llm_reasoning", "")
    expanded      = data.get("expanded_query", "")
    ctx_used      = data.get("llm_context_used", False)

    if llm_reasoning and expanded != query.strip():
        ctx_badge = '<span class="ctx-badge">Context Ranker Active</span>' if ctx_used else ""
        st.markdown(
            f'<div class="insight-box">'
            f'<b>OpenAI understood:</b> {llm_reasoning}{ctx_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif ctx_used:
        st.markdown(
            '<div class="insight-box">'
            '<span class="ctx-badge">Context Ranker Active</span>&nbsp;'
            'OpenAI has read each assessment description and ranked them against your job description.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Results summary
    st.markdown(f"### {data['total_returned']} Recommendations")

    type_counts = {}
    for r in recs:
        t = r.get("test_type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1

    mix_html = " ".join(
        f'<span class="chip">{t}: {c}</span>'
        for t, c in sorted(type_counts.items())
    )
    st.markdown(f"**Type mix:** {mix_html}", unsafe_allow_html=True)
    st.markdown("---")

    # Assessment cards
    for i, rec in enumerate(recs, 1):
        tt     = rec.get("test_type", "")
        ttl    = rec.get("test_type_label", "")
        dur    = rec.get("duration_mins")
        levels = rec.get("job_levels") or []
        remote = rec.get("remote_testing", False)
        adapt  = rec.get("adaptive_irt", False)

        chips = ""
        if dur:
            chips += chip_html(f"{dur} mins")
        if remote:
            chips += chip_html("Remote")
        if adapt:
            chips += chip_html("Adaptive")
        if levels:
            chips += chip_html(f"{levels[0]}" + (f" +{len(levels)-1}" if len(levels) > 1 else ""))

        desc = rec.get("description", "")
        desc_preview = (desc[:180] + "...") if len(desc) > 180 else desc

        reason = rec.get("reason", "").strip()
        reason_html = (
            f'<div class="reason-box">'
            f'<div class="reason-label">Why this assessment fits</div>'
            f'{reason}'
            f'</div>'
        ) if reason else ""

        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">#{i} &nbsp;<a href="{rec["url"]}" target="_blank" '
            f'style="color:#4f8ef7;text-decoration:none;">{rec["name"]}</a></div>'
            f'{tag_html(tt, ttl)}'
            f'<div class="card-meta">{chips}</div>'
            f'<div class="card-meta" style="margin-top:8px;">{desc_preview}</div>'
            f'{reason_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Table + download
    st.markdown("---")
    import pandas as pd
    df = pd.DataFrame([
        {
            "Name":     r["name"],
            "URL":      r["url"],
            "Type":     f"{r['test_type']} - {r['test_type_label']}",
            "Duration": f"{r['duration_mins']} mins" if r["duration_mins"] else "N/A",
            "Remote":   "Yes" if r["remote_testing"] else "No",
        }
        for r in recs
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv_bytes, "recommendations.csv", "text/csv")

elif search_clicked:
    st.warning("Please enter a query.")
