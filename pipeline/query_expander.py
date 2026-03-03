import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4o-mini"

# SHL test type codes reference (sent to OpenAI as context)
_TYPE_DESCRIPTIONS = {
    "A": "Ability & Aptitude (numerical, verbal, deductive, inductive reasoning)",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies (leadership, management, business competencies)",
    "D": "Development & 360-degree feedback",
    "E": "Assessment Exercises (role plays, group exercises)",
    "K": "Knowledge & Skills (technical skills, programming, domain knowledge)",
    "P": "Personality & Behavior (OPQ, personality questionnaires)",
    "S": "Simulations (job simulations, work samples)",
}

_SYSTEM_PROMPT = """You are an expert HR assessment consultant mapping job requirements to SHL assessment types.

SHL Test Type Codes:
{type_descriptions}

Given a job query or description, respond with ONLY valid JSON (no markdown, no code blocks):
{{
  "expanded_query": "<richer version of the query with synonyms, role context, key skills, and SHL product family names for semantic search>",
  "required_types": ["<type codes that are clearly relevant, e.g. K, P, A>"],
  "is_multi_domain": <true if query covers both technical AND behavioral/personality aspects, else false>,
  "max_duration_mins": <integer max assessment duration in minutes extracted from the query, or null if not mentioned>,
  "reasoning": "<1 sentence why these types were chosen>"
}}

DURATION EXTRACTION RULES:
- Extract the MAXIMUM allowed assessment duration in MINUTES as an integer
- "40 minutes" → 40
- "about an hour" or "1 hour" → 60
- "1-2 hour long" → 120  (use the upper bound)
- "30-40 mins" → 40  (use the upper bound)
- "90 mins" or "at most 90 mins" → 90
- "no bar on duration" or "no limit" or not mentioned → null

TYPE SELECTION RULES — read carefully, do NOT only look for explicit test keywords:
- Include A (Ability & Aptitude) when ANY of these appear:
    * Communication skills, verbal communication, written communication
    * Must read/write/speak a language (English, French, etc.)
    * Language proficiency, comprehension, verbal reasoning
    * Aptitude, cognitive ability, reasoning, problem solving
    * Content creation, script writing, copywriting, creative writing
    * Radio, media, broadcasting, journalism roles where language is central
- Include P (Personality & Behavior) when:
    * Interpersonal skills, teamwork, collaboration, stakeholder management
    * People management, coaching, mentoring, leadership style
    * Personality, behaviour, attitude, cultural fit, motivation
- Include K (Knowledge & Skills) when:
    * Specific technical domain, programming languages, tools
    * Domain expertise (marketing, finance, HR, legal, digital advertising, etc.)
    * Software, platforms, certifications
- Include C (Competencies) when:
    * Senior leadership, strategic thinking, organisational competencies
    * Management competency framework, business judgement
- is_multi_domain = true when TWO OR MORE types from [A, K, P, C] are needed

EXPANDED QUERY RULES:
- Add SHL product family synonyms: e.g. "Java" → add "Automata coding test Java Core Java"
- Add role synonyms: "COO" → "Chief Operating Officer executive leadership"
- Add assessment vocabulary: e.g. "personality" → "OPQ questionnaire personality behaviour"
- Add cognitive synonyms: "aptitude" → "Verify cognitive ability numerical verbal reasoning"

EXAMPLE 1: "Must know to read, write and speak English. Excellent communication skills. People management."
→ required_types: ["A", "P"]  is_multi_domain: true  max_duration_mins: null

EXAMPLE 2: "Java developer who collaborates with business teams, assessment within 40 minutes."
→ required_types: ["K", "P"]  is_multi_domain: true  max_duration_mins: 40

EXAMPLE 3: "Cognitive ability test for a data analyst, 1 hour budget."
→ required_types: ["A"]  is_multi_domain: false  max_duration_mins: 60
""".format(type_descriptions="\n".join(f"  {k}: {v}" for k, v in _TYPE_DESCRIPTIONS.items()))




def _call_openai(query: str) -> dict:
    """Call OpenAI gpt-4o-mini and return parsed JSON response."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model       = OPENAI_MODEL,
        temperature = 0.1,
        max_tokens  = 400,
        messages    = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Query: {query}"},
        ],
        response_format = {"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(raw)



def expand_query(query: str) -> dict:
    #Expand and structure a user query using OpenAI gpt-4o-mini.
    fallback = {
        "expanded_query":    query,
        "required_types":    [],
        "is_multi_domain":   False,
        "max_duration_mins": None,
        "reasoning":         "LLM not used (fallback mode)",
        "llm_used":          False,
    }

    if not OPENAI_API_KEY:
        print("[QueryExpander] No OPENAI_API_KEY — using raw query as-is.")
        return fallback

    try:
        parsed = _call_openai(query)

        # Extract max_duration_mins — must be int or None
        raw_dur = parsed.get("max_duration_mins")
        if isinstance(raw_dur, (int, float)) and raw_dur > 0:
            max_duration_mins = int(raw_dur)
        else:
            max_duration_mins = None

        return {
            "expanded_query":    parsed.get("expanded_query", query).strip() or query,
            "required_types":    [t.upper() for t in parsed.get("required_types", [])],
            "is_multi_domain":   bool(parsed.get("is_multi_domain", False)),
            "max_duration_mins": max_duration_mins,
            "reasoning":         parsed.get("reasoning", ""),
            "llm_used":          True,
        }

    except Exception as e:
        print(f"[QueryExpander] OpenAI call failed ({e}) — falling back to raw query.")
        return fallback


if __name__ == "__main__":
    TEST_QUERIES = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
        "cognitive ability test for software engineer",
        "personality assessment for a senior sales manager",
    ]

    for q in TEST_QUERIES:
        print(f"\nQuery: {q}")
        result = expand_query(q)
        print(f"  Expanded : {result['expanded_query'][:100]}")
        print(f"  Types    : {result['required_types']}")
        print(f"  Multi    : {result['is_multi_domain']}")
        print(f"  Reason   : {result['reasoning']}")
        print(f"  LLM used : {result['llm_used']}")
