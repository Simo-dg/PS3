# src/parse_bid_tables.py
"""
Parse CDOT Bid Tab summary tables (Rank/Vendor/Total Bid) from PDFs in data/pdf/
and write CSV (file, num_bidders, lowest_bid) to data/bid_min_and_count.csv.

Usage:
  python -m src.parse_bid_tables --pdf-dir data/pdf --out data/bid_min_and_count.csv

Requires: pip install pdfplumber
"""

import re
import csv
import argparse
from pathlib import Path
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple

import pdfplumber

DEFAULT_PDF_DIR = "data/pdf"
DEFAULT_OUT = "data/bid_min_and_count.csv"

# ---------- helpers ----------

def _norm(s: str) -> str:
    return (s or "").strip()

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", _norm(s))

def _header_key(s: str) -> str:
    # "Percent Of Low Bid" -> "percentoflowbid"
    return re.sub(r"[^a-z]", "", (s or "").lower())

_money_rx = re.compile(r"\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
def _parse_money(s: str) -> Optional[Decimal]:
    if not s:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    if s in {"", "-", "—"}:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        m = re.search(r"([0-9]+(?:\.[0-9]{1,2})?)", s)
        return Decimal(m.group(1)) if m else None

def _try_int(x: str) -> Optional[int]:
    if x is None:
        return None
    x = str(x).strip()
    m = re.match(r"^\d+", x)
    return int(m.group(0)) if m else None

# ---------- WORD-BOUNDED PASS ----------

def _group_rows(words: List[dict], y_tol: float = 3.0) -> List[List[dict]]:
    rows: List[List[dict]] = []
    for w in sorted(words, key=lambda d: (d["top"], d["x0"])):
        if not rows:
            rows.append([w]); continue
        if abs(w["top"] - rows[-1][0]["top"]) <= y_tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return rows

def _find_phrase_span(row_words: List[dict], tokens: List[str]) -> Optional[Tuple[float, float]]:
    toks = [re.sub(r"[^a-z]", "", (w["text"] or "").lower()) for w in row_words]
    for i in range(len(toks)):
        if toks[i] != tokens[0]:
            continue
        j, k = i, 0
        x0 = row_words[i]["x0"]; x1 = row_words[i]["x1"]
        while j < len(toks) and k < len(tokens) and toks[j] == tokens[k]:
            x1 = row_words[j]["x1"]; j += 1; k += 1
        if k == len(tokens):
            return (x0, x1)
    return None

def _extract_summary_from_page_words(page) -> List[Tuple[int, Decimal]]:
    """
    Using word coordinates: return [(rank, total_bid)] with rank >= 1.
    Specifically looks for CDOT 'Vendor Ranking' pages.
    """
    bidders: List[Tuple[int, Decimal]] = []

    words = page.extract_words(use_text_flow=True, keep_blank_chars=False,
                               x_tolerance=2, y_tolerance=3)
    if not words:
        return bidders
    rows = _group_rows(words, y_tol=3)

    # Check if this is a Vendor Ranking page
    page_text = " ".join(_norm_ws(w["text"]) for row in rows[:20] for w in row).lower()
    if "vendor ranking" not in page_text:
        return bidders
    
    # Find the header row with columns: Rank, Vendor ID, Vendor Name, Total Bid, Percent Of Low Bid, Percent Of Estimate
    header_row = None
    for idx, row in enumerate(rows[:80]):
        joined = " ".join(_norm_ws(w["text"]) for w in row)
        key = _header_key(joined)
        # Look for a row that has "rank", "vendor", and "totalbid" or "percentof"
        if "rank" in key and "vendor" in key:
            header_row = row
            header_idx = idx
            break
    
    if header_row is None:
        return bidders

    # Find column positions
    rank_span = _find_phrase_span(header_row, ["rank"])
    vendor_span = _find_phrase_span(header_row, ["vendor", "name"]) or _find_phrase_span(header_row, ["vendor"])
    total_span = _find_phrase_span(header_row, ["total", "bid"]) or _find_phrase_span(header_row, ["bid", "total"])
    
    if not rank_span or not total_span:
        return bidders

    rank_left = rank_span[0] - 5
    rank_right = rank_span[1] + 15
    
    # Total Bid column is typically after Vendor Name
    total_left = total_span[0] - 10
    total_right = total_span[1] + 100

    # Parse data rows after header
    for row in rows[header_idx + 1:]:
        # Look for rank (should be a number 1, 2, 3, etc. or 0 for estimate)
        r_words = [w for w in row if rank_left <= w["x0"] <= rank_right]
        rank = None
        for w in r_words:
            rank = _try_int(w["text"])
            if rank is not None:
                break
        
        if rank is None or rank <= 0:
            # Skip row 0 (Engineer's Estimate) and invalid ranks
            continue

        # Look for total bid amount (should be a money value)
        t_words = [w for w in row if total_left <= w["x0"] <= total_right]
        cands = [w for w in t_words if _money_rx.search(_norm(w["text"]))]
        if not cands:
            continue
        
        amt = None
        for w in sorted(cands, key=lambda d: d["x0"]):
            amt = _parse_money(w["text"])
            if amt is not None and amt >= 1000:
                break
        
        if amt is None or amt < 1000:
            continue

        bidders.append((rank, amt))

    return bidders

# ---------- LINE-BASED PASS (fallback) ----------

# detect "Vendor Ranking" page and parse the table
# Format: Rank Vendor ID Vendor Name Total Bid Low Bid Estimate
_vendor_ranking_rx = re.compile(r"vendor\s+ranking", re.I)
_rank_vendor_line_rx = re.compile(r"^\s*(\d{1,2})\s+\S+\s+.*?(\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+[\d.]+%\s+[\d.]+%", re.I)

def _extract_summary_from_page_lines(page) -> List[Tuple[int, Decimal]]:
    bidders: List[Tuple[int, Decimal]] = []
    text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
    if not text:
        return bidders
    lines = [l for l in (t.strip("\x00") for t in text.splitlines()) if l is not None]

    # Check if this is a Vendor Ranking page
    is_vendor_ranking = any(_vendor_ranking_rx.search(ln) for ln in lines[:10])
    if not is_vendor_ranking:
        return bidders

    # Parse lines looking for pattern: rank vendorID vendorName $totalBid percentLow percentEst
    for ln in lines:
        # Skip row 0 (Engineer's Estimate) - starts with "0 -EST-"
        if re.match(r"^\s*0\s+", ln):
            continue
            
        m = _rank_vendor_line_rx.match(ln)
        if not m:
            continue
        
        rank = int(m.group(1))
        if rank <= 0:
            continue
        
        amt = _parse_money(m.group(2))
        if amt is None or amt < 1000:
            continue
        
        bidders.append((rank, amt))

    return bidders

# ---------- orchestration ----------

def _extract_two_metrics_from_pdf(pdf_path: Path) -> Tuple[int, Optional[Decimal]]:
    """
    Returns (num_bidders, lowest_bid) using word-bounded pass first,
    then falling back to the line-based pass. Checks first 4 pages.
    """
    num_bidders = 0
    lowest_bid: Optional[Decimal] = None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = min(len(pdf.pages), 4)
            for i in range(pages_to_check):
                page = pdf.pages[i]

                bidders = _extract_summary_from_page_words(page)
                if not bidders:
                    bidders = _extract_summary_from_page_lines(page)

                if bidders:
                    amounts = [amt for _, amt in bidders]
                    num_bidders = len(bidders)
                    lowest_bid = min(amounts) if amounts else None
                    break
    except Exception:
        pass

    return num_bidders, lowest_bid

def _write_csv(rows: List[Tuple[str, int, Optional[Decimal]]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "num_bidders", "lowest_bid"])
        for file_path, n, low in rows:
            w.writerow([file_path, n, f"{low:.2f}" if low is not None else ""])

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Parse #bidders and winner's bid from CDOT summary tables.")
    ap.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR, help="Folder containing PDFs (default: data/pdf)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="CSV output path (default: data/bid_min_and_count.csv)")
    ap.add_argument("--f", default=None, help="(ignored; for notebook kernels)")
    args, unknown = ap.parse_known_args()
    if unknown:
        print("[INFO] Ignoring unknown args:", unknown)

    pdfs = sorted(Path(args.pdf_dir).rglob("*.pdf"))
    results: List[Tuple[str, int, Optional[Decimal]]] = []
    for pdf in pdfs:
        n, low = _extract_two_metrics_from_pdf(pdf)
        results.append((str(pdf), n, low))
        print(f"[OK] {Path(pdf).name}: bidders={n} | lowest={f'{low:.2f}' if low is not None else 'N/A'}")

    _write_csv(results, args.out)
    print(f"[DONE] Wrote {args.out} with {len(results)} rows.")

if __name__ == "__main__":
    main()