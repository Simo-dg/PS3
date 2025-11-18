# --- CDOT Bid Tabs scraper (Hyland DocPop/PdfPop) ---
# This script scrapes PDF bid tabs from the Colorado Department of Transportation (CDOT) website.
# The PDFs are hosted on a Hyland cloud-based document management system (DocPop/PdfPop).
# Fix: avoid resp.body() protocol errors by fetching the PDF inside the page (window.fetch + base64)

# ============================================================================
# 1) IMPORTS & CONFIGURATION
# ============================================================================

# Standard library imports
import os, re, time, asyncio, base64
from typing import List, Tuple, Dict, Optional
from urllib.parse import urljoin, urlparse, parse_qs

# Third-party imports
import requests                                  # For HTTP requests
from bs4 import BeautifulSoup                   # For HTML parsing
import pandas as pd                             # For data manipulation
from playwright.async_api import async_playwright  # For browser automation

# Pages to scrape (add/remove as needed)
PAGES = [
    "https://www.codot.gov/business/bidding/bid-tab-archives/copy_of_bid-tabs-2023" # (contains 2024 items)
]

# Output directory - relative to script location
OUT_DIR = "../data"

# Browser user agent string for requests (mimics Chrome on macOS)
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
# Delay between requests to be polite to the server
POLITE_DELAY_SECONDS = 0.3
# Timeout for HTTP requests and page loads (in seconds)
REQUEST_TIMEOUT = 45  

# ============================================================================
# 2) UTILITY FUNCTIONS
# ============================================================================

def slugify(text: str) -> str:
    """
    Convert text to a filesystem-safe filename.
    
    Args:
        text: The text to convert
        
    Returns:
        A sanitized string suitable for use as a filename (max 120 chars)
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Replace special characters with underscores (keep alphanumeric, hyphens, dots, parens)
    text = re.sub(r"[^\w\-\.\s\(\)]", "_", text)
    # Replace remaining spaces with underscores
    text = re.sub(r"\s", "_", text)
    # Limit length to avoid filesystem issues
    return text[:120] or "file"

def get_out_dir() -> str:
    """
    Get the output directory path relative to the script location.
    
    Returns:
        Absolute path to ../data directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, OUT_DIR))

def extract_docid_from_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract document ID and client type from a Hyland DocPop URL.
    
    Args:
        url: The DocPop URL to parse
        
    Returns:
        Tuple of (docid, clienttype), either may be None if not present
    """
    # Parse query string parameters
    q = parse_qs(urlparse(url).query)
    docid = (q.get("docid") or [None])[0]
    clienttype = (q.get("clienttype") or [None])[0]
    return docid, clienttype

def direct_pdf_url(docid: str, clienttype: Optional[str] = None) -> str:
    """
    Construct a direct URL to the PDF viewer given a document ID.
    
    Args:
        docid: The document ID
        clienttype: Optional client type parameter
        
    Returns:
        Complete URL to the PdfPop viewer
    """
    base = "https://oitco.hylandcloud.com/cdotrmpop/docpop/PdfPop.aspx"
    if clienttype:
        return f"{base}?clienttype={clienttype}&docid={docid}"
    return f"{base}?docid={docid}"

def build_requests_session(user_agent: str = BROWSER_UA) -> requests.Session:
    """
    Create a requests Session with custom user agent.
    
    Args:
        user_agent: User agent string to use for requests
        
    Returns:
        Configured requests Session object
    """
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s

def scrape_page_for_links(session: requests.Session, page_url: str) -> List[Tuple[str, str]]:
    """
    Scrape a CDOT page to find all DocPop/PdfPop links.
    
    Args:
        session: Requests session to use
        page_url: URL of the CDOT bid tabs page to scrape
        
    Returns:
        List of tuples: (anchor_text, absolute_docpop_url)
    """
    # Fetch the page
    r = session.get(page_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(r.text, "html.parser")
    
    # Find all anchors pointing to hylandcloud.com DocPop URLs
    anchors = [a for a in soup.find_all("a", href=True)
               if "hylandcloud.com" in a["href"] and "docpop" in a["href"]]
    
    # Deduplicate and extract text and full URLs
    seen, items = set(), []
    for a in anchors:
        full = urljoin(page_url, a["href"])  # Convert to absolute URL
        if full in seen:
            continue
        seen.add(full)
        text = a.get_text(strip=True) or "Project_Bid_Tab"
        items.append((text, full))
    return items

def save_csv(rows: List[Tuple[str, str, str, str, str]], out_dir: str) -> str:
    """
    Save the index of discovered bid tabs to a CSV file.
    
    Args:
        rows: List of tuples (page, anchor_text, docid, clienttype, pdf_url)
        out_dir: Output directory for the CSV file
        
    Returns:
        Path to the created CSV file
    """
    csv_path = os.path.join(out_dir, "cdot_bid_tabs_index.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        # Write header
        f.write("page,anchor_text,docid,clienttype,pdf_url\n")
        # Write data rows with proper CSV escaping
        for page, text, docid, clienttype, pdf in rows:
            page_csv = '"' + page.replace('"', '""') + '"'
            text_csv = '"' + text.replace('"', '""') + '"'
            client_csv = clienttype or ""
            f.write(f"{page_csv},{text_csv},{docid},{client_csv},{pdf}\n")
    return csv_path

# ============================================================================
# 3) BROWSER AUTOMATION HELPERS (PLAYWRIGHT)
# ============================================================================

async def wait_for_pdf_url(page, timeout_ms: int = 20000):
    """
    Listen for network responses and capture the URL of the first PDF response.
    
    This function attaches a response event handler to intercept network traffic
    and identify when a PDF file is being loaded (by checking Content-Type headers).
    
    Args:
        page: Playwright page object
        timeout_ms: Maximum time to wait in milliseconds
        
    Returns:
        URL of the PDF response, or None if timeout or no PDF found
    """
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()

    def handler(resp):
        """Response handler to check if response is a PDF."""
        try:
            # Try accessing headers as dict attribute
            headers = resp.headers
            ct = (headers.get("content-type") or "").lower()
        except Exception:
            try:
                # Fallback: try calling headers() method
                ct = (resp.headers().get("content-type") or "").lower()
            except Exception:
                ct = ""
        # If this is a PDF and we haven't already resolved, capture the URL
        if "application/pdf" in ct and not future.done():
            try:
                future.set_result(resp.url)
            except Exception:
                pass

    # Attach the response listener
    page.on("response", handler)
    try:
        # Wait for the future to resolve or timeout
        return await asyncio.wait_for(future, timeout=timeout_ms/1000)
    except asyncio.TimeoutError:
        return None
    finally:
        # Clean up: remove the event handler
        try:
            page.off("response", handler)
        except Exception:
            pass

async def page_fetch_bytes(page, url: str) -> Optional[bytes]:
    """
    Fetch a URL from within the browser context to preserve cookies and referer.
    
    This approach avoids protocol errors that can occur when using response.body()
    directly. Instead, we use JavaScript's fetch() API inside the page, convert
    the response to base64, and decode it in Python.
    
    Args:
        page: Playwright page object
        url: URL to fetch
        
    Returns:
        Response body as bytes, or None if fetch failed
    """
    # JavaScript code to fetch URL and return as base64
    js = """
    async (url) => {
      // Fetch with credentials to maintain session
      const res = await fetch(url, { credentials: 'include' });
      if (!res.ok) return null;
      
      // Convert response to byte array
      const buf = await res.arrayBuffer();
      const bytes = new Uint8Array(buf);
      
      // Convert to base64 in chunks to avoid stack overflow on large files
      const chunk = 0x8000;  // 32KB chunks
      let binary = '';
      for (let i = 0; i < bytes.length; i += chunk) {
        const sub = bytes.subarray(i, i + chunk);
        binary += String.fromCharCode.apply(null, Array.from(sub));
      }
      return btoa(binary);  // base64 encode
    }
    """
    # Execute the JavaScript and get base64 result
    b64 = await page.evaluate(js, url)
    if not b64:
        return None
    try:
        # Decode base64 to bytes
        return base64.b64decode(b64)
    except Exception:
        return None

async def fetch_pdf_via_browser(context, docpop_url: str, out_path_pdf: str, out_path_html: str) -> bool:
    """
    Use Playwright to navigate to a DocPop URL and download the PDF.
    
    This function implements multiple fallback strategies:
    1. Listen for PDF responses during initial page load
    2. If no PDF found, try constructing direct PDF URL from docid
    3. If still no PDF, search page DOM for embedded PDF viewers
    4. Save HTML for debugging if PDF cannot be extracted
    
    Args:
        context: Playwright browser context
        docpop_url: URL of the DocPop document viewer page
        out_path_pdf: Where to save the PDF file
        out_path_html: Where to save HTML if PDF extraction fails
        
    Returns:
        True if PDF was successfully saved, False otherwise
    """
    page = await context.new_page()
    try:
        # STRATEGY 1: Listen for PDF responses during page load
        pdf_future = asyncio.create_task(wait_for_pdf_url(page, timeout_ms=20000))
        await page.goto(docpop_url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT * 1000)
        pdf_url = await pdf_future

        # STRATEGY 2: If no PDF captured, try direct PdfPop URL from docid
        if not pdf_url:
            docid, clienttype = extract_docid_from_url(docpop_url)
            if docid:
                # Construct direct PDF viewer URL
                pdf_url = direct_pdf_url(docid, clienttype)
                pdf_future2 = asyncio.create_task(wait_for_pdf_url(page, timeout_ms=15000))
                await page.goto(pdf_url, referer=docpop_url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT * 1000)
                # Check if the viewer made any internal PDF requests
                captured = await pdf_future2
                if captured:
                    pdf_url = captured

        # If we have a PDF URL, fetch its content
        if pdf_url:
            pdf_bytes = await page_fetch_bytes(page, pdf_url)
            # Validate it's actually a PDF (starts with %PDF- magic bytes or is substantial)
            if pdf_bytes and (pdf_bytes.startswith(b"%PDF-") or len(pdf_bytes) > 1000):
                with open(out_path_pdf, "wb") as fp:
                    fp.write(pdf_bytes)
                return True

        # STRATEGY 3: Search DOM for embedded PDF viewers
        # Look for embedded PDF viewers in the page DOM
        embed_src = await page.evaluate("""
        () => {
          // Find all embed, iframe, and object elements
          const sel = Array.from(document.querySelectorAll('embed,iframe,object'));
          // Extract their src/data attributes
          const cand = sel.map(e => e.src || e.data || '').filter(Boolean);
          // Prefer URLs pointing to PdfPop.aspx
          const m = cand.find(s => /PdfPop\\.aspx/i.test(s));
          // Return PdfPop URL if found, otherwise first candidate
          return m || (cand[0] || '');
        }
        """)
        if embed_src:
            pdf_bytes = await page_fetch_bytes(page, embed_src)
            if pdf_bytes and (pdf_bytes.startswith(b"%PDF-") or len(pdf_bytes) > 1000):
                with open(out_path_pdf, "wb") as fp:
                    fp.write(pdf_bytes)
                return True

        # If all strategies failed, save HTML for manual diagnosis
        html = await page.content()
        with open(out_path_html, "w", encoding="utf-8") as fp:
            fp.write(html)
        return False

    finally:
        await page.close()

# ============================================================================
# 6) MAIN SCRAPING ORCHESTRATION
# ============================================================================

async def run_cdot_in_notebook_async(
    pages: Optional[List[str]] = None,
    user_agent: str = BROWSER_UA,
    no_download: bool = False,
    polite_delay: float = POLITE_DELAY_SECONDS,
) -> pd.DataFrame:
    """
    Main function to scrape CDOT bid tabs and download PDFs.
    
    This function:
    1. Scrapes specified pages to find DocPop links
    2. Optionally downloads PDFs using browser automation
    3. Creates an index CSV with all discovered documents
    4. Returns a DataFrame with results
    
    Args:
        pages: List of CDOT pages to scrape
        user_agent: User agent string for requests
        no_download: If True, only build index without downloading PDFs
        polite_delay: Delay between requests in seconds
        
    Returns:
        DataFrame containing metadata for all discovered bid tabs
    """
    # Setup - output to ../data and ../data/pdf
    pages = pages or PAGES
    out_dir = get_out_dir()
    pdf_dir = os.path.join(out_dir, "pdf")
    print(f"[INFO] Output folder: {out_dir}")
    print(f"[INFO] PDF folder: {pdf_dir}")

    # Create HTTP session for scraping
    sess = build_requests_session(user_agent)
    all_items: List[Tuple[str, str, str]] = []  # (page_url, anchor_text, docpop_url)

    # PHASE 1: Scrape all pages to find DocPop links
    for page_url in pages:
        try:
            items = scrape_page_for_links(sess, page_url)
            print(f"[OK] {page_url}: found {len(items)} DocPop links")
            for text, docpop_url in items:
                all_items.append((page_url, text, docpop_url))
        except Exception as e:
            print(f"[ERROR] {page_url}: {e}")

    # Prepare data structures for results
    index_rows: List[Tuple[str, str, str, str, str]] = []  # For CSV
    rows: List[Dict[str, str]] = []  # For DataFrame

    # Initialize Playwright browser (only if downloading)
    pw = browser = context = None
    try:
        if not no_download and all_items:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True)  # Headless mode
            context = await browser.new_context(user_agent=user_agent, accept_downloads=False)

        # PHASE 2: Process each discovered DocPop link
        for page_url, text, docpop_url in all_items:
            # Extract document metadata from URL
            docid, clienttype = extract_docid_from_url(docpop_url)
            pdf_url_guess = direct_pdf_url(docid, clienttype) if docid else ""
            
            # Add to index (CSV will be created later)
            index_rows.append((page_url, text, docid or "", clienttype or "", pdf_url_guess))

            # Initialize result tracking
            saved_path = ""
            status = "skipped" if no_download else "html_saved"

            # Download PDF if requested
            if not no_download:
                # Create filename from anchor text and docid
                fname = f"{slugify(text)}__{docid or 'unknown'}.pdf"
                out_pdf = os.path.join(pdf_dir, fname)
                out_html = out_pdf.rsplit(".pdf", 1)[0] + ".html"

                # Attempt to fetch PDF via browser
                ok = await fetch_pdf_via_browser(context, docpop_url, out_pdf, out_html)
                if ok:
                    saved_path = out_pdf
                    status = "saved"
                    print(f"[OK] {fname}")
                else:
                    print(f"[WARN] Not a PDF. Saved HTML at: {out_html}")
                
                # Be polite: delay between requests
                await asyncio.sleep(polite_delay)

            # Add to results DataFrame
            rows.append({
                "page": page_url,
                "anchor_text": text,
                "docid": docid or "",
                "clienttype": clienttype or "",
                "pdf_url": pdf_url_guess,
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

        # Clean up browser resources
        if context:
            await context.close()
        if browser:
            await browser.close()
        if pw:
            await pw.stop()

    # Create DataFrame from results
    df = pd.DataFrame(rows)
    print("\n[INFO] First 20 rows:")
    print(df.head(20).to_string())
    return df

# ============================================================================
# 7) ENTRY POINT
# ============================================================================

async def main():
    """
    Entry point when running as a script.
    
    Executes the full scraping process with default settings.
    
    Returns:
        DataFrame with scraping results
    """
    df = await run_cdot_in_notebook_async(
        pages=PAGES,
        user_agent=BROWSER_UA,
        no_download=False,                     # Set True to only build index without downloading PDFs
        polite_delay=POLITE_DELAY_SECONDS,
    )
    return df

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())