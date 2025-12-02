import os
import re
import asyncio
import base64
import logging
from typing import List, Tuple, Dict, Optional
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import pandas as pd
from playwright.async_api import async_playwright

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Directories and settings
PAGES = [
    "https://www.codot.gov/business/bidding/bid-tab-archives/copy_of_bid-tabs-2023",
    "https://www.codot.gov/business/bidding/bid-tab-archives/2023-bid-tabs/bid-tabs-2023",
    "https://www.codot.gov/business/bidding/bid-tab-archives/2022-bid-tabs/copy_of_bid-tabs-2022",
    "https://www.codot.gov/business/bidding/bid-tab-archives"
]
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
DELAY_SECONDS = 0.3
REQUEST_TIMEOUT = 45
OUT_DIR = "../data"  # Output directory for PDFs and CSV


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe filename."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\-\.\s\(\)]", "_", text)
    text = re.sub(r"\s", "_", text)
    return text[:120] or "file"


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
        return await asyncio.wait_for(future, timeout=timeout_ms / 1000)
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

async def fetch_pdf_via_browser(context, docpop_url: str, out_path_pdf: str) -> bool:
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
    finally:
        await page.close()



async def run_cdot_scraper(pages:List[str]=None) -> pd.DataFrame:
    """
    Scrape CDOT bid tabs and download PDFs for all links in all pages in the list.
    Returns DataFrame containing metadata for all discovered bid tabs.
    """
    if pages is None:
        pages = PAGES
    pdf_dir = os.path.join(OUT_DIR, "pdf")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    logger.info(f"Output folder: {OUT_DIR}")
    logger.info(f"PDF folder: {pdf_dir}")

    all_links: List[Tuple[str, str, str]] = []
    for page_url in pages:
        items = scrape_page_for_links(page_url)
        logger.info(f"{page_url}: found {len(items)} DocPop links")
        for text, docpop_url in items:
            all_links.append((page_url, text, docpop_url))

    rows: List[Dict[str, str]] = []
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)
    context = await browser.new_context(user_agent=BROWSER_UA, accept_downloads=False)

    for page_url, text, docpop_url in all_links:
        saved_path = ""
        status = "html_saved"
        fname = f"{slugify(text)}.pdf"
        out_pdf = os.path.join(pdf_dir, fname)

        ok = await fetch_pdf_via_browser(context, docpop_url, out_pdf)
        if ok:
            saved_path = out_pdf
            status = "saved"
            logger.info(f"Successfully saved {fname}")
        else:
            logger.warning(f"Not a PDF: {docpop_url}")
        await asyncio.sleep(DELAY_SECONDS)
        rows.append({
            "page": page_url,
            "anchor_text": text,
            "docpop_url": docpop_url,
            "saved_path": saved_path,
            "status": status,
        })

    # Save index CSV
    csv_path = os.path.join(OUT_DIR, "cdot_bid_tabs_index.csv")
    df = pd.DataFrame(rows, columns=["page", "anchor_text", "docpop_url", "saved_path", "status"])
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")

    await context.close()
    await browser.close()
    await pw.stop()
    return df



async def main():
    df = await run_cdot_scraper()
    return df


if __name__ == "__main__":
    asyncio.run(main())
