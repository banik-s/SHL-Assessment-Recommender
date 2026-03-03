import sys, os
os.environ.setdefault("OPENAI_API_KEY", open(".env").read().split("OPENAI_API_KEY=")[1].split("\n")[0].strip() if os.path.exists(".env") else "")
sys.path.insert(0, ".")

from pipeline.retrieve import Retriever, resolve_query

r = Retriever()

# All expected slugs across all 10 train queries
all_expected = {
    # Java
    "automata-fix-new", "core-java-entry-level-new", "java-8-new", "core-java-advanced-level-new", "interpersonal-communications",
    # Sales
    "entry-level-sales-7-1", "entry-level-sales-sift-out-7-1", "entry-level-sales-solution", "sales-representative-solution",
    "business-communication-adaptive", "technical-sales-associate-solution", "svar-spoken-english-indian-accent-new", "english-comprehension-new",
    # COO
    "enterprise-leadership-report", "occupational-personality-questionnaire-opq32r", "opq-leadership-report",
    "opq-team-types-and-leadership-styles-report", "enterprise-leadership-report-2-0", "global-skills-assessment",
    # QA
    "automata-selenium", "professional-7-1-solution", "javascript-new", "htmlcss-new", "css3-new", "selenium-new", "sql-server-new", "automata-sql-new", "manual-testing-new",
    # Consultant
    "shl-verify-interactive-numerical-calculation", "administrative-professional-short-form", "verify-verbal-ability-next-generation", "professional-7-1-solution",
    # Data Analyst
    "tableau-new", "microsoft-excel-365-new", "microsoft-excel-365-essentials-new", "professional-7-0-solution-3958", "data-warehousing-concepts",
}

import json
data = json.load(open("data/vector_store/metadata.json"))
indexed_slugs = {d["url"].rstrip("/").split("/")[-1] for d in data}

print("=== MISSING from index (not scrapeable) ===")
for slug in sorted(all_expected):
    if slug not in indexed_slugs:
        print(f"  MISSING: {slug}")

print(f"\nTotal expected unique: {len(all_expected)}")
print(f"Total in index: {len(indexed_slugs)}")
found = len([s for s in all_expected if s in indexed_slugs])
print(f"Found: {found}/{len(all_expected)}")
