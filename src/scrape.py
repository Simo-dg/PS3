# --- CDOT Bid Tabs scraper (Hyland DocPop/PdfPop) ---

# ============================================================================
# 1) IMPORTS & CONFIGURATION
# ============================================================================

import os
import re
import asyncio
import base64
from typing import List, Tuple, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd
from playwright.async_api import async_playwright

# Pages to scrape
PAGES = [
    "https://www.codot.gov/business/bidding/bid-tab-archives/copy_of_bid-tabs-2023"
]

# Output directory - relative to script location
OUT_DIR = "../data"

# Browser user agent string
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

# Delay between requests
DELAY_SECONDS = 0.3

# Timeout for requests and page loads (in seconds)
REQUEST_TIMEOUT = 45

# ============================================================================
# 2) UTILITY FUNCTIONS
# ============================================================================

def slugify(text: str) -> str:
    """Convert text to a filesystem-safe filename."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\-\.\s\(\)]", "_", text)
    text = re.sub(r"\s", "_", text)
    return text[:120] or "file"

def get_out_dir() -> str:
    """Get the output directory path relative to the script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, OUT_DIR))

def scrape_page_for_links(page_url: str) -> List[Tuple[str, str]]:
    """
    Scrape a CDOT page to find all DocPop/PdfPop links.
    Returns list of tuples: (anchor_text, absolute_docpop_url)
    """
    sess = requests.Session()
    sess.headers.update({"User-Agent": BROWSER_UA})
    
    r = sess.get(page_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    
    soup = BeautifulSoup(r.text, "html.parser")
    anchors = [a for a in soup.find_all("a", href=True)
               if "hylandcloud.com" in a["href"] and "docpop" in a["href"]]
    
    seen, items = set(), []
    for a in anchors:
        full = urljoin(page_url, a["href"])
        if full in seen:
            continue
        seen.add(full)
        text = a.get_text(strip=True) or "Project_Bid_Tab"
        items.append((text, full))
    return items

def save_csv(rows: List[Tuple[str, str, str]], out_dir: str) -> str:
    """Save the index of discovered bid tabs to a CSV file."""
    csv_path = os.path.join(out_dir, "cdot_bid_tabs_index.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("page,anchor_text,docpop_url\n")
        for page, text, url in rows:
            page_csv = '"' + page.replace('"', '""') + '"'
            text_csv = '"' + text.replace('"', '""') + '"'
            url_csv = '"' + url.replace('"', '""') + '"'
            f.write(f"{page_csv},{text_csv},{url_csv}\n")
    return csv_path

# ============================================================================
# 3) BROWSER AUTOMATION
# ============================================================================

async def wait_for_pdf_url(page, timeout_ms: int = 20000):
    """Listen for network responses and capture the URL of the first PDF response."""
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()

    def handler(resp):
        try:
            headers = resp.headers
            ct = (headers.get("content-type") or "").lower()
        except Exception:
            try:
                ct = (resp.headers().get("content-type") or "").lower()
            except Exception:
                ct = ""
        
        if "application/pdf" in ct and not future.done():
            try:
                future.set_result(resp.url)
            except Exception:
                pass

    page.on("response", handler)
    try:
        return await asyncio.wait_for(future, timeout=timeout_ms/1000)
    except asyncio.TimeoutError:
        return None
    finally:
        try:
            page.off("response", handler)
        except Exception:
            pass

async def page_fetch_bytes(page, url: str) -> Optional[bytes]:
    """Fetch a URL from within the browser context using JavaScript."""
    js = """
    async (url) => {
      const res = await fetch(url, { credentials: 'include' });
      if (!res.ok) return null;
      
      const buf = await res.arrayBuffer();
      const bytes = new Uint8Array(buf);
      
      const chunk = 0x8000;
      let binary = '';
      for (let i = 0; i < bytes.length; i += chunk) {
        const sub = bytes.subarray(i, i + chunk);
        binary += String.fromCharCode.apply(null, Array.from(sub));
      }
      return btoa(binary);
    }
    """
    b64 = await page.evaluate(js, url)
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None

async def fetch_pdf_via_browser(context, docpop_url: str, out_path_pdf: str, out_path_html: str) -> bool:
    """
    Use Playwright to navigate to a DocPop URL and download the PDF.
    Listens for PDF responses during initial page load.
    """
    page = await context.new_page()
    try:
        # Listen for PDF responses during page load
        pdf_future = asyncio.create_task(wait_for_pdf_url(page, timeout_ms=20000))
        await page.goto(docpop_url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT * 1000)
        pdf_url = await pdf_future

        # If we have a PDF URL, fetch its content
        if pdf_url:
            pdf_bytes = await page_fetch_bytes(page, pdf_url)
            if pdf_bytes and (pdf_bytes.startswith(b"%PDF-") or len(pdf_bytes) > 1000):
                with open(out_path_pdf, "wb") as fp:
                    fp.write(pdf_bytes)
                return True

        # If no PDF found, save HTML for debugging
        html = await page.content()
        with open(out_path_html, "w", encoding="utf-8") as fp:
            fp.write(html)
        return False

    finally:
        await page.close()

# ============================================================================
# 4) MAIN SCRAPING ORCHESTRATION
# ============================================================================

async def run_cdot_scraper(
    pages: Optional[List[str]] = None,
    no_download: bool = False,
    delay: float = DELAY_SECONDS,
) -> pd.DataFrame:
    """
    Main function to scrape CDOT bid tabs and download PDFs.
    
    Returns DataFrame containing metadata for all discovered bid tabs.
    """
    pages = pages or PAGES
    out_dir = get_out_dir()
    pdf_dir = os.path.join(out_dir, "pdf")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    print(f"[INFO] Output folder: {out_dir}")
    print(f"[INFO] PDF folder: {pdf_dir}")

    all_items: List[Tuple[str, str, str]] = []

    # PHASE 1: Scrape all pages to find DocPop links
    for page_url in pages:
        try:
            items = scrape_page_for_links(page_url)
            print(f"[OK] {page_url}: found {len(items)} DocPop links")
            for text, docpop_url in items:
                all_items.append((page_url, text, docpop_url))
        except Exception as e:
            print(f"[ERROR] {page_url}: {e}")

    index_rows: List[Tuple[str, str, str]] = []
    rows: List[Dict[str, str]] = []

    # Initialize Playwright browser (only if downloading)
    pw = browser = context = None
    try:
        if not no_download and all_items:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=BROWSER_UA, accept_downloads=False)

        # PHASE 2: Process each discovered DocPop link
        for page_url, text, docpop_url in all_items:
            index_rows.append((page_url, text, docpop_url))

            saved_path = ""
            status = "skipped" if no_download else "html_saved"

            if not no_download:
                fname = f"{slugify(text)}.pdf"
                out_pdf = os.path.join(pdf_dir, fname)
                out_html = out_pdf.rsplit(".pdf", 1)[0] + ".html"

                ok = await fetch_pdf_via_browser(context, docpop_url, out_pdf, out_html)
                if ok:
                    saved_path = out_pdf
                    status = "saved"
                    print(f"[OK] {fname}")
                else:
                    print(f"[WARN] Not a PDF. Saved HTML at: {out_html}")
                
                await asyncio.sleep(delay)

            rows.append({
                "page": page_url,
                "anchor_text": text,
                "docpop_url": docpop_url,
                "saved_path": saved_path,
                "status": status,
            })
    finally:
        # PHASE 3: Save index CSV
        try:
            csv_path = save_csv(index_rows, out_dir)
            print(f"[OK] Index CSV: {csv_path}")
        except Exception as e:
            print(f"[WARN] Unable to write CSV: {e}")

        if context:
            await context.close()
        if browser:
            await browser.close()
        if pw:
            await pw.stop()

    df = pd.DataFrame(rows)
    return df

# ============================================================================
# 5) ENTRY POINT
# ============================================================================

async def main():
    """Entry point when running as a script."""
    df = await run_cdot_scraper(
        pages=PAGES,
        no_download=False,
        delay=DELAY_SECONDS,
    )
    return df

if __name__ == "__main__":
    asyncio.run(main())