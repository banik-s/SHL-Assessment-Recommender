import asyncio
import json
import os
import re
import time
from bs4 import BeautifulSoup

# ── crawl4ai ──────────────────────────────────────────────────────────────────
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    print("[WARN] crawl4ai not installed.")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shl_assessments.json")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

BATCH_SIZE  = 5    # concurrent pages per batch
SAVE_EVERY  = 25   # save progress every N items
DELAY_SEC   = 0.5  # polite delay between batches


def parse_detail_page(html: str, url: str) -> dict:
    result = {
        "description":   "",
        "job_levels":    [],
        "languages":     [],
        "duration_mins": None,
        "test_type":     "",
    }

    if not html:
        return result

    soup = BeautifulSoup(html, "lxml")

    def get_section_text(label: str) -> str:
        for h4 in soup.find_all("h4"):
            if label.lower() in h4.get_text(strip=True).lower():
                # Collect all text from sibling elements until next h4
                texts = []
                for sib in h4.next_siblings:
                    if sib.name == "h4":
                        break
                    text = sib.get_text(separator=" ", strip=True) if hasattr(sib, "get_text") else str(sib).strip()
                    if text:
                        texts.append(text)
                return " ".join(texts).strip()
        return ""

    desc = get_section_text("Description")
    if desc:
        result["description"] = desc

    job_levels_text = get_section_text("Job levels")
    if job_levels_text:
        result["job_levels"] = [j.strip() for j in job_levels_text.split(",") if j.strip()]

    lang_text = get_section_text("Languages")
    if lang_text:
        result["languages"] = [l.strip() for l in lang_text.split(",") if l.strip()]

    # ── Duration: "Approximate Completion Time in minutes = N" ───────────────
    full_text = soup.get_text(" ", strip=True)

    dur_match = re.search(r"Approximate Completion Time in minutes\s*=\s*(\d+)", full_text)
    if dur_match:
        result["duration_mins"] = int(dur_match.group(1))

    # ── Test Type: "Test Type: X" ────────────────────────────────────────────
    type_match = re.search(r"Test Type:\s*([A-Z])", full_text)
    if type_match:
        result["test_type"] = type_match.group(1)

    # ── Strategy 2: og:description meta fallback ─────────────────────────────
    if not result["description"]:
        og_desc = soup.find("meta", property="og:description")
        if og_desc:
            result["description"] = og_desc.get("content", "").strip()

    return result


# Fetch with crawl4ai (async)


async def fetch_batch(urls: list[str], crawler: "AsyncWebCrawler") -> dict[str, str]:
    
    run_cfg = CrawlerRunConfig(
        wait_for="js:() => document.querySelector('h4') !== null",
        delay_before_return_html=2.0,
        page_timeout=25000,
    )

    tasks = [crawler.arun(url=url, config=run_cfg) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    html_map = {}
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            print(f"  [WARN] crawl4ai failed for {url}: {result}")
            html_map[url] = ""
        elif result.success:
            html_map[url] = result.html
        else:
            print(f"  [WARN] crawl4ai error for {url}: {result.error_message}")
            html_map[url] = ""

    return html_map

def fetch_single_playwright(url: str) -> str:
    """Fetch a single detail page using playwright sync API."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, timeout=25000)
            try:
                page.wait_for_selector("h4", timeout=10000)
            except Exception:
                pass  # Continue even if selector not found
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"    [ERROR] Playwright fallback also failed: {e}")
        return ""


def load_assessments() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_assessments(data: list[dict]) -> None:
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def enrich():
    assessments = load_assessments()
    total       = len(assessments)

    # Resume: skip already enriched items
    to_enrich = [
        (i, a) for i, a in enumerate(assessments)
        if not a.get("description")
    ]
    print(f"\nTotal assessments : {total}")
    print(f"Already enriched  : {total - len(to_enrich)}")
    print(f"To enrich         : {len(to_enrich)}\n")

    if not to_enrich:
        print("[OK] All assessments already have descriptions.")
        return

    enriched_count = 0

    if CRAWL4AI_AVAILABLE:
        browser_cfg = BrowserConfig(headless=True, user_agent=USER_AGENT)

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            # Process in batches for concurrency
            for batch_start in range(0, len(to_enrich), BATCH_SIZE):
                batch = to_enrich[batch_start: batch_start + BATCH_SIZE]
                urls  = [a["url"] for _, a in batch]

                batch_num = batch_start // BATCH_SIZE + 1
                total_batches = (len(to_enrich) + BATCH_SIZE - 1) // BATCH_SIZE
                print(f"[Batch {batch_num}/{total_batches}] Fetching {len(batch)} pages...")

                html_map = await fetch_batch(urls, crawler)

                for (orig_idx, assessment), url in zip(batch, urls):
                    html = html_map.get(url, "")

                    # Fallback if crawl4ai returned empty
                    if not html:
                        print(f"  Fallback → {assessment['name']}")
                        html = fetch_single_playwright(url)

                    enrichment = parse_detail_page(html, url)
                    assessments[orig_idx].update(enrichment)
                    enriched_count += 1

                    desc_preview = (enrichment["description"] or "N/A")[:60]
                    print(f"  [{enriched_count:>3}] {assessment['name'][:45]:<45} | {desc_preview}")

                # Save progress periodically
                if enriched_count % SAVE_EVERY == 0:
                    save_assessments(assessments)
                    print(f"  [SAVED] Progress saved ({enriched_count} enriched)")

                await asyncio.sleep(DELAY_SEC)

    else:
        # Pure playwright fallback
        for enriched_count, (orig_idx, assessment) in enumerate(to_enrich, 1):
            print(f"[{enriched_count:>3}/{len(to_enrich)}] {assessment['name'][:50]}")
            html = fetch_single_playwright(assessment["url"])
            enrichment = parse_detail_page(html, assessment["url"])
            assessments[orig_idx].update(enrichment)

            if enriched_count % SAVE_EVERY == 0:
                save_assessments(assessments)
                print(f"  [SAVED] Progress saved")

            time.sleep(DELAY_SEC)

    # Final save
    save_assessments(assessments)

    # Summary
    with_desc = sum(1 for a in assessments if a.get("description"))
    with_dur  = sum(1 for a in assessments if a.get("duration_mins"))
    with_type = sum(1 for a in assessments if a.get("test_type"))

    print(f"\n{'='*55}")
    print(f"Enrichment complete!")
    print(f"  Assessments with description  : {with_desc}/{total}")
    print(f"  Assessments with duration     : {with_dur}/{total}")
    print(f"  Assessments with test_type    : {with_type}/{total}")
    print(f"  Saved to: {DATA_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    asyncio.run(enrich())
