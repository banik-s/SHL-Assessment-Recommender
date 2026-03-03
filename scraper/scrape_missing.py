import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MISSING_URLS = [
    "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-7-1/",
    "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-sift-out-7-1/",
    "https://www.shl.com/solutions/products/product-catalog/view/financial-professional-short-form/",
    "https://www.shl.com/solutions/products/product-catalog/view/general-entry-level-data-entry-7-0-solution/",
    "https://www.shl.com/solutions/products/product-catalog/view/manager-8-0-jfa-4310/",
    "https://www.shl.com/solutions/products/product-catalog/view/professional-7-0-solution-3958/",
    "https://www.shl.com/solutions/products/product-catalog/view/professional-7-1-solution/",
    "https://www.shl.com/solutions/products/product-catalog/view/sales-representative-solution/",
    "https://www.shl.com/solutions/products/product-catalog/view/technical-sales-associate-solution/",
]

# Known test-type mapping (from scraper/context)
TEST_TYPE_MAPPING = {
    "A": "A",  # Ability & Aptitude
    "B": "B",  # Biodata & Situational Judgment
    "C": "C",  # Competencies
    "D": "D",  # Development & 360
    "E": "E",  # Assessment Exercises
    "K": "K",  # Knowledge & Skills
    "P": "P",  # Personality & Behavior
    "S": "S",  # Simulations
}


def parse_product_page(html: str, url: str) -> dict:
    """Parse an individual SHL product page to extract assessment metadata."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # --- Name ---
    name = ""
    h1 = soup.select_one("h1")
    if h1:
        name = h1.get_text(strip=True)
    if not name:
        # Try title tag
        title = soup.select_one("title")
        if title:
            name = title.get_text(strip=True).split("|")[0].strip()

    # --- Description ---
    description = ""
    # Try product description block
    desc_selectors = [
        ".product-catalogue__description",
        ".product-description",
        "[class*='description']",
        "section p",
        "article p",
        ".content p",
    ]
    for sel in desc_selectors:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(strip=True)
            if len(text) > 40:
                description = text[:1000]
                break

    if not description:
        # Grab first substantial paragraph
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 80:
                description = text[:1000]
                break

    # --- Test types from badge-like spans ---
    test_type = ""
    type_badges = soup.select("span.catalogue__badge, span[class*='badge'], span[class*='type']")
    types_found = []
    for b in type_badges:
        t = b.get_text(strip=True).upper()
        if t in TEST_TYPE_MAPPING:
            types_found.append(t)
    if types_found:
        test_type = types_found[0]  # Use the first/primary type

    # --- Duration ---
    duration_mins = None
    duration_text = soup.get_text()
    dur_match = re.search(r"(\d+)\s*(?:min(?:ute)?s?|mins?)", duration_text, re.IGNORECASE)
    if dur_match:
        val = int(dur_match.group(1))
        if 5 <= val <= 240:  # sanity check
            duration_mins = val

    # --- Remote testing & Adaptive ---
    remote_testing = False
    adaptive_irt = False
    page_text = soup.get_text().lower()
    if "remote" in page_text:
        remote_testing = True
    if "adaptive" in page_text or "irt" in page_text:
        adaptive_irt = True

    # --- Job levels ---
    job_levels = []
    level_keywords = {
        "entry": "Entry-Level",
        "graduate": "Entry-Level",
        "professional": "Professional",
        "manager": "Manager",
        "director": "Director",
        "executive": "Executive",
        "mid-level": "Mid-Professional",
        "general population": "General Population",
    }
    for kw, level in level_keywords.items():
        if kw in page_text and level not in job_levels:
            job_levels.append(level)

    # Fallback: infer from URL slug
    slug = url.rstrip("/").split("/")[-1].lower()
    if not job_levels:
        if "entry" in slug or "graduate" in slug or "7-0" in slug or "7-1" in slug:
            job_levels = ["Entry-Level"]
        elif "manager" in slug:
            job_levels = ["Manager", "Mid-Professional"]
        elif "professional" in slug:
            job_levels = ["Professional", "Mid-Professional"]

    # Fallback: infer test_type from URL slug
    if not test_type:
        if "solution" in slug:
            test_type = "B"  # Job-focused / situational = Biodata
        elif "short-form" in slug:
            test_type = "B"
        elif "simulation" in slug:
            test_type = "S"
        else:
            test_type = "B"  # Safe default for job-focused solutions

    return {
        "name": name or slug.replace("-", " ").title(),
        "url": url,
        "test_type": test_type,
        "description": description,
        "duration_mins": duration_mins,
        "remote_testing": remote_testing,
        "adaptive_irt": adaptive_irt,
        "job_levels": job_levels,
    }


def fetch_with_playwright(url: str) -> str:
    """Fetch a page using synchronous Playwright."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
            ))
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"  [Playwright error] {e}")
        return ""


def fetch_with_requests(url: str) -> str:
    """Simpler fallback using requests + trafilatura."""
    import requests
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36"
        })
        return resp.text
    except Exception as e:
        print(f"  [requests error] {e}")
        return ""


def main():
    print("=" * 60)
    print("Scraping 9 missing assessments from SHL product pages")
    print("=" * 60)

    # Load current shl_assessments.json
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "shl_assessments.json")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    existing_slugs = {d["url"].rstrip("/").split("/")[-1] for d in data}
    print(f"Current assessment count: {len(data)}")
    print()

    new_items = []
    for url in MISSING_URLS:
        slug = url.rstrip("/").split("/")[-1]
        if slug in existing_slugs:
            print(f"  [SKIP] Already indexed: {slug}")
            continue

        print(f"  Fetching: {slug}")

        # Try Playwright first, then requests
        html = fetch_with_playwright(url)
        if not html:
            print(f"    Playwright failed, trying requests...")
            html = fetch_with_requests(url)

        if not html:
            print(f"    [FAIL] Could not fetch {url}")
            # Add a minimal entry so the slug is indexed
            item = {
                "name": slug.replace("-", " ").title(),
                "url": url,
                "test_type": "B",
                "description": f"Assessment: {slug.replace('-', ' ')}",
                "duration_mins": None,
                "remote_testing": True,
                "adaptive_irt": False,
                "job_levels": [],
            }
        else:
            item = parse_product_page(html, url)

        print(f"    -> name='{item['name']}' type={item['test_type']} dur={item['duration_mins']} desc_len={len(item['description'])}")
        new_items.append(item)
        time.sleep(1)

    if new_items:
        data.extend(new_items)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nAdded {len(new_items)} new assessments")
        print(f"Total now: {len(data)} assessments")
        print(f"Saved to: {json_path}")
    else:
        print("\nNo new items to add.")


if __name__ == "__main__":
    main()
