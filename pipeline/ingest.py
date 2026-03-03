import json
import os
import re

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "shl_assessments.json")

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

JUNK_PATTERNS = [
    r"^report for .+$",              # "Report for Verify Interactive G+"
    r"^.{0,40}report$",              # very short strings ending in "report"
    r"^.{0,40}interpretation report$",
]

WEAK_DESC_THRESHOLD = 60   # chars — descriptions below this get type-level augmentation

TYPE_CONTEXT = {
    "A": "Measures cognitive ability, aptitude, numerical reasoning, verbal reasoning, deductive and inductive reasoning skills.",
    "B": "Measures biodata, situational judgement, and behavioural tendencies relevant to the role.",
    "C": "Measures competencies such as leadership, management, collaboration, and business judgement.",
    "D": "Supports development and 360-degree feedback for individual growth and career planning.",
    "E": "Assessment exercise such as group exercise, role play, analysis presentation, or written exercise.",
    "K": "Knowledge and skills test measuring technical proficiency, programming ability, or domain-specific knowledge.",
    "P": "Personality and behaviour questionnaire measuring work-related personality traits and behavioural styles.",
    "S": "Simulation-based assessment measuring practical job skills in a realistic task environment.",
}


# ── Name-derived keyword synonyms injected for weak-description assessments ────
# Maps name fragments (lowercase) to extra search terms. Each entry is triggered
# when the fragment appears anywhere in the assessment's normalised name.
_NAME_KEYWORD_MAP: list[tuple[str, str]] = [
    # Java / JVM
    ("java",        "Java programming OOP object-oriented JVM developer backend"),
    ("core java",   "Core Java fundamentals entry level beginner junior programmer"),
    ("automata",    "Automata coding challenge programming test live code browser"),
    ("selenium",    "Selenium WebDriver test automation browser testing QA"),
    # Web / JS
    ("javascript",  "JavaScript JS frontend web development browser scripting"),
    ("html",        "HTML CSS web markup frontend developer"),
    ("css",         "CSS styling frontend web developer"),
    ("react",       "React.js frontend JavaScript UI developer"),
    # Data / DB
    ("sql",         "SQL database query relational RDBMS data analyst"),
    ("python",      "Python programming data science scripting analyst developer"),
    ("excel",       "Microsoft Excel spreadsheet data analysis formulas"),
    ("tableau",     "Tableau data visualisation BI business intelligence"),
    ("data warehouse", "data warehousing ETL dimensional modelling BI"),
    # Testing / QA
    ("manual testing", "manual QA quality assurance software testing bug"),
    # Personality / Leadership
    ("opq",         "OPQ personality questionnaire occupational occupational behaviour traits work style"),
    ("hipo",        "high potential leadership talent identification executive"),
    ("enterprise leadership", "executive leadership COO CEO C-suite senior leader"),
    # Cognitive
    ("verify",      "Verify cognitive ability aptitude reasoning test SHL"),
    ("inductive reasoning",  "inductive logical pattern abstract reasoning"),
    ("deductive reasoning",  "deductive logical reasoning structured abstract"),
    ("numerical",   "numerical data analysis quantitative maths calculation"),
    ("verbal",      "verbal reasoning comprehension language grammar writing"),
    # Banking / Admin
    ("bank",        "banking finance BFSI financial services clerk admin"),
    ("administrative", "admin office administration clerical assistant"),
    ("financial professional", "finance accounting banking analyst professional"),
    # Sales / Service
    ("sales",       "sales representative account manager business development revenue"),
    ("entry level sales", "sales new graduate fresher entry level"),
    ("customer serv", "customer service support call centre helpdesk"),
    # Marketing / Content
    ("marketing",   "marketing digital brand content SEO strategy campaigns"),
    ("seo",         "SEO search engine optimisation digital marketing content"),
    ("written english", "writing grammar English proficiency content"),
    # HR / People
    ("human resources", "HR human resources talent management people ops"),
    # Global
    ("global skills", "global cross-cultural international cultural fit"),
    ("interpersonal", "interpersonal communication collaboration teamwork soft skills"),
]


def _name_keywords(name: str) -> str:
    
    name_lower = name.lower()
    keywords: list[str] = []
    for fragment, synonyms in sorted(_NAME_KEYWORD_MAP, key=lambda x: -len(x[0])):
        if fragment in name_lower:
            keywords.append(synonyms)
    return " | ".join(keywords)



def is_junk_description(name: str, desc: str) -> bool:
    """Return True if the description is useless (empty, too short, or just echoes name)."""
    if not desc or len(desc.strip()) < 30:   # raised from 20 → 30
        return True
    # If description is almost identical to the name → junk
    desc_clean = desc.strip().lower()
    name_clean = name.strip().lower()
    if desc_clean == name_clean:
        return True
    if len(desc) < 50 and name_clean in desc_clean:   # raised from 40 → 50
        return True
    # Junk regex patterns
    for pattern in JUNK_PATTERNS:
        if re.match(pattern, desc_clean):
            return True
    return False


def description_quality(desc: str) -> str:
    """Return 'good' if description is informative, 'weak' if short/sparse."""
    return "good" if len(desc.strip()) >= WEAK_DESC_THRESHOLD else "weak"


def clean_description(name: str, desc: str) -> str:
    
    desc = (desc or "").strip()

    if is_junk_description(name, desc):
        # Generate a minimal fallback so the composite text still has context
        return f"An SHL assessment named '{name}'."

    # Normalise whitespace
    desc = re.sub(r"\s+", " ", desc)
    return desc


def clean_job_levels(levels: list) -> list[str]:
    """Deduplicate and strip job levels."""
    if not levels:
        return []
    seen = set()
    result = []
    for l in levels:
        l = l.strip().rstrip(",").strip()
        if l and l not in seen:
            seen.add(l)
            result.append(l)
    return result


def build_composite_text(assessment: dict) -> str:
    """
    Build a rich single string from all available fields.
    This is what gets embedded — richer text = better semantic search.

    Format:
      <name> | Type: <type_label> | Levels: <levels> | Duration: <N> mins |
      [Remote Testing available] [Adaptive/IRT] |
      Category: <type_context_sentence> |   ← injected for weak descriptions only
      Measures: <description>
    """
    name      = assessment.get("name", "").strip()
    desc      = clean_description(name, assessment.get("description", ""))
    test_type = assessment.get("test_type", "")
    levels    = clean_job_levels(assessment.get("job_levels") or [])
    duration  = assessment.get("duration_mins")
    remote    = assessment.get("remote_testing", False)
    adaptive  = assessment.get("adaptive_irt", False)
    qual      = description_quality(desc)

    parts = [name]

    # Test type
    type_label = TEST_TYPE_MAP.get(test_type, "")
    if type_label:
        parts.append(f"Type: {type_label}")

    # Job levels
    if levels:
        parts.append(f"Levels: {', '.join(levels)}")

    # Duration
    if duration:
        parts.append(f"Duration: {duration} mins")

    # Remote / Adaptive flags (adds signal for filtered queries)
    flags = []
    if remote:
        flags.append("Remote Testing available")
    if adaptive:
        flags.append("Adaptive/IRT")
    if flags:
        parts.append(", ".join(flags))

    # For weak descriptions: inject type-level context + name-derived keyword synonyms
    if qual == "weak":
        if test_type in TYPE_CONTEXT:
            parts.append(f"Category: {TYPE_CONTEXT[test_type]}")
        extra_kw = _name_keywords(name)
        if extra_kw:
            parts.append(f"Keywords: {extra_kw}")

    # Description — most important, always last
    parts.append(f"Measures: {desc}")

    return " | ".join(parts)



def load_and_prepare(data_path: str = DATA_PATH) -> list[dict]:
    """
    Load assessments from JSON, clean all fields, and add composite_text.

    Returns:
        List of dicts, each with:
          - name           : str
          - url            : str
          - description    : str (cleaned)
          - test_type      : str
          - job_levels     : list[str]
          - duration_mins  : int | None
          - remote_testing : bool
          - adaptive_irt   : bool
          - composite_text : str   ← used directly for embedding
    """
    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)

    prepared = []
    skipped  = 0

    for item in raw:
        name = (item.get("name") or "").strip()
        url  = (item.get("url") or "").strip()

        # Skip records with no name or URL (should not happen, but be safe)
        if not name or not url:
            skipped += 1
            continue

        desc_cleaned = clean_description(name, item.get("description", ""))
        cleaned = {
            "name":                name,
            "url":                 url,
            "description":         desc_cleaned,
            "test_type":           (item.get("test_type") or "").strip(),
            "job_levels":          clean_job_levels(item.get("job_levels") or []),
            "duration_mins":       item.get("duration_mins"),
            "remote_testing":      bool(item.get("remote_testing", False)),
            "adaptive_irt":        bool(item.get("adaptive_irt", False)),
            # description_quality lets downstream steps know how rich this description is
            "description_quality": description_quality(desc_cleaned),
        }

        cleaned["composite_text"] = build_composite_text(cleaned)
        prepared.append(cleaned)

    return prepared

if __name__ == "__main__":
    assessments = load_and_prepare()
    total = len(assessments)

    print(f"Loaded and prepared: {total} assessments\n")

    # Show samples — short, medium, good description
    samples = (
        [a for a in assessments if len(a["description"]) < 60][:2]
        + [a for a in assessments if 60 <= len(a["description"]) < 200][:2]
        + [a for a in assessments if len(a["description"]) >= 200][:2]
    )

    print("── Sample composite texts ──────────────────────────────────────")
    for a in samples:
        print(f"\n[{len(a['description'])} chars desc] {a['name']}")
        print(f"  {a['composite_text']}")

    # Quality check
    empty_composite = sum(1 for a in assessments if len(a["composite_text"]) < 30)
    avg_len         = sum(len(a["composite_text"]) for a in assessments) // total
    print(f"\n── Stats ───────────────────────────────────────────────────────")
    print(f"  Total prepared        : {total}")
    print(f"  Avg composite length  : {avg_len} chars")
    print(f"  Suspiciously short    : {empty_composite}")
