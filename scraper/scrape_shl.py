import asyncio
import json
import os
import time
import requests
from bs4 import BeautifulSoup

# ── crawl4ai ─────────────────────────────────────────────────────────────────
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    print("[WARN] crawl4ai not installed — using fallback.")

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL     = "https://www.shl.com"
CATALOG_BASE = "https://www.shl.com/solutions/products/product-catalog/"
PAGE_SIZE    = 12
MAX_PAGES    = 35      # 32 confirmed pages + buffer
USER_AGENT   = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "shl_assessments.json"
)



def build_url(start: int) -> str:
    return f"{CATALOG_BASE}?start={start}&type=1"



def parse_html(html: str) -> list[dict]:
    
    soup = BeautifulSoup(html, "lxml")
    assessments = []

    # Verified: the catalog table uses class 'custom__table-catalog'
    table = soup.select_one("table.custom__table-catalog")
    if not table:
        # Attempt broader match if class varies
        table = soup.select_one("table")

    if not table:
        return []

    rows = table.select("tbody tr")

    for row in rows:
        # Name & URL — verified selector
        name_cell = row.select_one("td.custom__table-heading__title a")
        if not name_cell:
            continue

        name = name_cell.get_text(strip=True)
        href = name_cell.get("href", "")
        url  = href if href.startswith("http") else BASE_URL + href

        # General columns (remote, adaptive, test types)
        general_cols = row.select("td.custom__table-heading__general")

        # Col 0 (index 1 in full row) Remote Testing
        remote_testing = False
        if len(general_cols) > 0:
            remote_testing = bool(general_cols[0].select_one("span.catalogue__circle"))

        # Col 1 (index 2 in full row) Adaptive/IRT
        adaptive_irt = False
        if len(general_cols) > 1:
            adaptive_irt = bool(general_cols[1].select_one("span.catalogue__circle"))

        # Col 2+ Test Type badges (A=Ability, B=Biodata, C=Competency, P=Personality, S=Simulation, E=Assessment Exercise, K=Knowledge)
        test_types = []
        for col in general_cols[2:]:
            badges = col.select("span.catalogue__badge")
            for badge in badges:
                label = badge.get_text(strip=True)
                if label:
                    test_types.append(label)

        assessments.append({
            "name":           name,
            "url":            url,
            "remote_testing": remote_testing,
            "adaptive_irt":   adaptive_irt,
            "test_types":     test_types,
            "description":    "",
        })

    return assessments



async def fetch_page_crawl4ai(url: str, crawler: "AsyncWebCrawler") -> str:
    
    run_cfg = CrawlerRunConfig(
        wait_for="js:() => document.querySelector('table') !== null",
        delay_before_return_html=3.0,      # give JS time to render
        page_timeout=30000,
        js_code="window.scrollTo(0, 300);",  # trigger lazy loading
    )
    result = await crawler.arun(url=url, config=run_cfg)
    if result.success:
        return result.html
    raise RuntimeError(f"crawl4ai: {result.error_message}")


async def scrape_with_crawl4ai() -> list[dict]:
    """Paginate through the SHL catalog using crawl4ai."""
    browser_cfg = BrowserConfig(headless=True, user_agent=USER_AGENT)
    all_assessments = []
    seen_urls       = set()

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for page_num in range(MAX_PAGES):
            start = page_num * PAGE_SIZE
            url   = build_url(start)
            print(f"[crawl4ai] Page {page_num + 1:>2} | offset {start:>4} {url}")

            try:
                html   = await fetch_page_crawl4ai(url, crawler)
                items  = parse_html(html)
            except Exception as e:
                print(f"  [WARN] crawl4ai error: {e} — skipping page")
                items = []

            new_items = [a for a in items if a["url"] not in seen_urls]
            for a in new_items:
                seen_urls.add(a["url"])
            all_assessments.extend(new_items)

            print(f"   {len(new_items)} new items | running total: {len(all_assessments)}")

            if not new_items:
                print("  No new items found. Stopping pagination.")
                break

            await asyncio.sleep(0.8)

    return all_assessments



def fetch_page_playwright_sync(url: str) -> str:
    """Synchronous Playwright fetch for fallback use."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, timeout=30000)
            page.wait_for_selector("table.custom__table-catalog", timeout=15000)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"  [WARN] Playwright sync fallback failed: {e}")
        return ""


def scrape_with_playwright_fallback() -> list[dict]:
    """Fallback: paginate using synchronous Playwright."""
    all_assessments = []
    seen_urls       = set()

    for page_num in range(MAX_PAGES):
        start = page_num * PAGE_SIZE
        url   = build_url(start)
        print(f"[Playwright-sync] Page {page_num + 1:>2} | offset {start:>4}")

        html  = fetch_page_playwright_sync(url)
        items = parse_html(html) if html else []

        new_items = [a for a in items if a["url"] not in seen_urls]
        for a in new_items:
            seen_urls.add(a["url"])
        all_assessments.extend(new_items)

        print(f"  {len(new_items)} new items | running total: {len(all_assessments)}")

        if not new_items:
            print("  No new items found. Stopping.")
            break

        time.sleep(0.8)

    return all_assessments



def save(data: list[dict]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(data)} assessments  {OUTPUT_PATH}")


async def main():
    print("=" * 60)
    print("SHL Catalog Scraper — Individual Test Solutions")
    print("=" * 60)

    data = []

    if CRAWL4AI_AVAILABLE:
        print("\n[MODE] Primary: crawl4ai (Playwright)\n")
        try:
            data = await scrape_with_crawl4ai()
        except Exception as e:
            print(f"\n[WARN] crawl4ai pipeline failed entirely: {e}")
            print("[MODE] Fallback: Playwright sync\n")
            data = scrape_with_playwright_fallback()
    else:
        print("\n[MODE] Fallback: Playwright sync\n")
        data = scrape_with_playwright_fallback()

    print(f"\nTotal assessments collected: {len(data)}")

    if len(data) < 377:
        print(f"[WARN] Expected >=377 items, got {len(data)}.")
    else:
        print(f"[OK] Minimum count of 377 met.")

    save(data)


if __name__ == "__main__":
    asyncio.run(main())
