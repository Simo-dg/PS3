# --- CDOT Bid Tabs scraper (Hyland DocPop/PdfPop) ---
# Fix: avoid resp.body() protocol errors by fetching the PDF inside the page (window.fetch + base64)

# 1) Imports & config
import os, re, time, tempfile, asyncio, base64
from typing import List, Tuple, Dict, Optional
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
import pandas as pd
from playwright.async_api import async_playwright

# Pages to scrape (add/remove as needed)
PAGES = [
    "https://www.codot.gov/business/bidding/bid-tab-archives/copy_of_bid-tabs-2023" # (contains some 2024 items)
]

# Preferred output directory (choose a writable folder)
OUT_DIR_PREFERRED = "../data"  # set None to auto-pick

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
POLITE_DELAY_SECONDS = 0.3
REQUEST_TIMEOUT = 45  # seconds

# 2) Utils
def slugify(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\-\.\s\(\)]", "_", text)
    text = re.sub(r"\s", "_", text)
    return text[:120] or "file"

def is_writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        testfile = os.path.join(path, ".writetest.tmp")
        with open(testfile, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(testfile)
        return True
    except Exception:
        return False

def pick_out_dir(preferred: Optional[str]) -> str:
    candidates: List[str] = []
    if preferred:
        # Se è un percorso relativo, calcolalo dalla posizione dello script
        if not os.path.isabs(preferred):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            preferred_abs = os.path.normpath(os.path.join(script_dir, preferred))
            candidates.append(preferred_abs)
        else:
            candidates.append(os.path.expanduser(preferred))
    # Percorso di default: ../data rispetto alla posizione dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.normpath(os.path.join(script_dir, "..", "data")))
    candidates.append(os.path.join(os.path.expanduser("~"), "Downloads", "cdot_bid_tabs"))
    candidates.append(os.path.join(tempfile.gettempdir(), "cdot_bid_tabs"))
    candidates.append(os.path.abspath("cdot_bid_tabs"))
    for p in candidates:
        if is_writable(p):
            return p
    raise OSError("No writable output directory found. Set OUT_DIR_PREFERRED to a valid path.")

def extract_docid_from_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    q = parse_qs(urlparse(url).query)
    docid = (q.get("docid") or [None])[0]
    clienttype = (q.get("clienttype") or [None])[0]
    return docid, clienttype

def direct_pdf_url(docid: str, clienttype: Optional[str] = None) -> str:
    base = "https://oitco.hylandcloud.com/cdotrmpop/docpop/PdfPop.aspx"
    if clienttype:
        return f"{base}?clienttype={clienttype}&docid={docid}"
    return f"{base}?docid={docid}"

def build_requests_session(user_agent: str = BROWSER_UA) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s

def scrape_page_for_links(session: requests.Session, page_url: str) -> List[Tuple[str, str]]:
    """Return (anchor_text, absolute DocPop url)."""
    r = session.get(page_url, timeout=REQUEST_TIMEOUT)
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

def save_csv(rows: List[Tuple[str, str, str, str, str]], out_dir: str) -> str:
    csv_path = os.path.join(out_dir, "cdot_bid_tabs_index.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("page,anchor_text,docid,clienttype,pdf_url\n")
        for page, text, docid, clienttype, pdf in rows:
            page_csv = '"' + page.replace('"', '""') + '"'
            text_csv = '"' + text.replace('"', '""') + '"'
            client_csv = clienttype or ""
            f.write(f"{page_csv},{text_csv},{docid},{client_csv},{pdf}\n")
    return csv_path

# 3) Event listener: capture first PDF response URL (not body)
async def wait_for_pdf_url(page, timeout_ms: int = 20000):
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()

    def handler(resp):
        try:
            headers = resp.headers  # dict
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

# 4) Fetch a URL inside the page to preserve cookies/referer; return bytes
async def page_fetch_bytes(page, url: str) -> Optional[bytes]:
    js = """
    async (url) => {
      const res = await fetch(url, { credentials: 'include' });
      if (!res.ok) return null;
      const buf = await res.arrayBuffer();
      const bytes = new Uint8Array(buf);
      // chunked base64 to avoid stack issues
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

# 5) Async Playwright: open DocPop and save PDF via in-page fetch
async def fetch_pdf_via_browser(context, docpop_url: str, out_path_pdf: str, out_path_html: str) -> bool:
    page = await context.new_page()
    try:
        # Listen before navigation
        pdf_future = asyncio.create_task(wait_for_pdf_url(page, timeout_ms=20000))
        await page.goto(docpop_url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT * 1000)
        pdf_url = await pdf_future

        # Fallback: if no PDF captured, try direct PdfPop from docid
        if not pdf_url:
            docid, clienttype = extract_docid_from_url(docpop_url)
            if docid:
                pdf_url = direct_pdf_url(docid, clienttype)
                pdf_future2 = asyncio.create_task(wait_for_pdf_url(page, timeout_ms=15000))
                await page.goto(pdf_url, referer=docpop_url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT * 1000)
                # If the viewer makes an internal request, we capture its URL; otherwise we still have pdf_url
                captured = await pdf_future2
                if captured:
                    pdf_url = captured

        # If we have a URL that should be the PDF, fetch bytes inside the page
        if pdf_url:
            pdf_bytes = await page_fetch_bytes(page, pdf_url)
            if pdf_bytes and (pdf_bytes.startswith(b"%PDF-") or len(pdf_bytes) > 1000):
                with open(out_path_pdf, "wb") as fp:
                    fp.write(pdf_bytes)
                return True

        # Last resort: look for an <embed|iframe|object> pointing to PdfPop.aspx and fetch that
        embed_src = await page.evaluate("""
        () => {
          const sel = Array.from(document.querySelectorAll('embed,iframe,object'));
          const cand = sel.map(e => e.src || e.data || '').filter(Boolean);
          const m = cand.find(s => /PdfPop\\.aspx/i.test(s));
          return m || (cand[0] || '');
        }
        """)
        if embed_src:
            pdf_bytes = await page_fetch_bytes(page, embed_src)
            if pdf_bytes and (pdf_bytes.startswith(b"%PDF-") or len(pdf_bytes) > 1000):
                with open(out_path_pdf, "wb") as fp:
                    fp.write(pdf_bytes)
                return True

        # Save HTML for diagnosis
        html = await page.content()
        with open(out_path_html, "w", encoding="utf-8") as fp:
            fp.write(html)
        return False

    finally:
        await page.close()

# 6) Main async runner
async def run_cdot_in_notebook_async(
    pages: Optional[List[str]] = None,
    out_dir_preferred: Optional[str] = OUT_DIR_PREFERRED,
    user_agent: str = BROWSER_UA,
    no_download: bool = False,
    polite_delay: float = POLITE_DELAY_SECONDS,
) -> pd.DataFrame:
    pages = pages or PAGES
    out_dir = pick_out_dir(out_dir_preferred)
    pdf_dir = os.path.join(out_dir, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    print(f"[INFO] Output folder: {out_dir}")
    print(f"[INFO] PDF folder: {pdf_dir}")

    sess = build_requests_session(user_agent)
    all_items: List[Tuple[str, str, str]] = []

    for page_url in pages:
        try:
            items = scrape_page_for_links(sess, page_url)
            print(f"[OK] {page_url}: found {len(items)} DocPop links")
            for text, docpop_url in items:
                all_items.append((page_url, text, docpop_url))
        except Exception as e:
            print(f"[ERROR] {page_url}: {e}")

    index_rows: List[Tuple[str, str, str, str, str]] = []
    rows: List[Dict[str, str]] = []

    pw = browser = context = None
    try:
        if not no_download and all_items:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=user_agent, accept_downloads=False)

        for page_url, text, docpop_url in all_items:
            docid, clienttype = extract_docid_from_url(docpop_url)
            pdf_url_guess = direct_pdf_url(docid, clienttype) if docid else ""
            index_rows.append((page_url, text, docid or "", clienttype or "", pdf_url_guess))

            saved_path = ""
            status = "skipped" if no_download else "html_saved"

            if not no_download:
                fname = f"{slugify(text)}__{docid or 'unknown'}.pdf"
                out_pdf = os.path.join(pdf_dir, fname)
                out_html = out_pdf.rsplit(".pdf", 1)[0] + ".html"

                ok = await fetch_pdf_via_browser(context, docpop_url, out_pdf, out_html)
                if ok:
                    saved_path = out_pdf
                    status = "saved"
                    print(f"[OK] {fname}")
                else:
                    print(f"[WARN] Not a PDF. Saved HTML at: {out_html}")
                await asyncio.sleep(polite_delay)

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
    print("\n[INFO] First 20 rows:")
    print(df.head(20).to_string())
    return df

# 7) Main function
async def main():
    df = await run_cdot_in_notebook_async(
        pages=PAGES,
        out_dir_preferred=OUT_DIR_PREFERRED,   # e.g., "~/Downloads/cdot_bid_tabs"
        user_agent=BROWSER_UA,
        no_download=False,                     # True = only build index, don't save PDFs
        polite_delay=POLITE_DELAY_SECONDS,
    )
    return df

if __name__ == "__main__":
    asyncio.run(main())