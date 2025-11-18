# ============================================================================
# CDOT BID TAB PDF PARSER
# ============================================================================
"""
Parse CDOT Bid Tab summary tables (Rank/Vendor/Total Bid) from PDFs in data/pdf/
and write CSV (file, num_bidders, lowest_bid) to data/bid_min_and_count.csv.

This script extracts bidding information from CDOT (Colorado Department of Transportation)
PDF bid tabs. It uses two parsing strategies:
1. Word-bounded pass: Uses word coordinates to locate table columns precisely
2. Line-based pass: Uses regex patterns as a fallback for simpler table formats

The target is the "Vendor Ranking" page which contains:
- Rank (1, 2, 3, etc.)
- Vendor ID
- Vendor Name
- Total Bid amount
- Percent of Low Bid
- Percent of Estimate

Usage:
  python -m src.data --pdf-dir data/pdf --out data/bid_min_and_count.csv

Requires: pip install pdfplumber
"""

# ============================================================================
# IMPORTS
# ============================================================================

import re
import csv
import argparse
from pathlib import Path
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple

import pdfplumber

# Default paths
DEFAULT_PDF_DIR = "data/pdf"
DEFAULT_OUT = "data/bid_min_and_count.csv"

# ============================================================================
# HELPER FUNCTIONS - TEXT NORMALIZATION AND PARSING
# ============================================================================

def _norm(s: str) -> str:
    """Strip whitespace from string."""
    return (s or "").strip()

def _norm_ws(s: str) -> str:
    """Normalize whitespace: collapse multiple spaces into single space."""
    return re.sub(r"\s+", " ", _norm(s))

def _header_key(s: str) -> str:
    """
    Convert header text to normalized key for matching.
    Example: "Percent Of Low Bid" -> "percentoflowbid"
    
    Args:
        s: Header text
        
    Returns:
        Lowercase string with non-alphabetic characters removed
    """
    return re.sub(r"[^a-z]", "", (s or "").lower())

# Regex to match money amounts: $1,234.56 or 1234.56
_money_rx = re.compile(r"\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?")

def _parse_money(s: str) -> Optional[Decimal]:
    """
    Parse a money string into a Decimal.
    
    Handles formats like: $1,234.56, 1234.56, $1234, etc.
    
    Args:
        s: String containing a money amount
        
    Returns:
        Decimal value or None if parsing fails
    """
    if not s:
        return None
    # Remove dollar signs and commas
    s = s.replace("$", "").replace(",", "").strip()
    # Handle empty or placeholder values
    if s in {"", "-", "—"}:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        # Try extracting just the numeric part
        m = re.search(r"([0-9]+(?:\.[0-9]{1,2})?)", s)
        return Decimal(m.group(1)) if m else None

def _try_int(x: str) -> Optional[int]:
    """
    Try to parse the beginning of a string as an integer.
    
    Args:
        x: String to parse
        
    Returns:
        Integer value or None if no valid integer found
    """
    if x is None:
        return None
    x = str(x).strip()
    m = re.match(r"^\d+", x)
    return int(m.group(0)) if m else None

# ============================================================================
# WORD-BASED EXTRACTION STRATEGY
# ============================================================================
# This approach uses word coordinates from pdfplumber to locate table columns
# precisely, which is more robust than line-based regex matching.

def _group_rows(words: List[dict], y_tol: float = 3.0) -> List[List[dict]]:
    """
    Group words into rows based on their vertical position (top coordinate).
    
    Words with similar y-coordinates (within y_tol) are considered part of
    the same row.
    
    Args:
        words: List of word dictionaries from pdfplumber (with 'top', 'x0', 'text')
        y_tol: Vertical tolerance in points for grouping words into same row
        
    Returns:
        List of rows, where each row is a list of word dictionaries
    """
    rows: List[List[dict]] = []
    # Sort words by vertical position (top) first, then horizontal (x0)
    for w in sorted(words, key=lambda d: (d["top"], d["x0"])):
        if not rows:
            rows.append([w])
            continue
        # Check if word is on same row as previous (within tolerance)
        if abs(w["top"] - rows[-1][0]["top"]) <= y_tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return rows

def _find_phrase_span(row_words: List[dict], tokens: List[str]) -> Optional[Tuple[float, float]]:
    """
    Find the horizontal span (x0, x1) of a multi-word phrase in a row.
    
    This helps locate column headers like "Total Bid" or "Vendor Name".
    
    Args:
        row_words: Words in a single row
        tokens: List of normalized tokens to find (e.g., ["total", "bid"])
        
    Returns:
        Tuple of (left_x, right_x) coordinates, or None if phrase not found
    """
    # Normalize all words to lowercase without special chars
    toks = [re.sub(r"[^a-z]", "", (w["text"] or "").lower()) for w in row_words]
    
    # Search for sequence of tokens
    for i in range(len(toks)):
        if toks[i] != tokens[0]:
            continue
        j, k = i, 0
        x0 = row_words[i]["x0"]
        x1 = row_words[i]["x1"]
        # Try to match full sequence
        while j < len(toks) and k < len(tokens) and toks[j] == tokens[k]:
            x1 = row_words[j]["x1"]
            j += 1
            k += 1
        # If we matched all tokens, return the span
        if k == len(tokens):
            return (x0, x1)
    return None

def _extract_summary_from_page_words(page) -> List[Tuple[int, Decimal]]:
    """
    Extract bidder rankings and amounts using word coordinate analysis.
    
    This is the primary extraction method. It:
    1. Checks if page contains "Vendor Ranking" text
    2. Locates the header row with column names
    3. Finds x-coordinates for Rank and Total Bid columns
    4. Extracts data from rows below the header
    
    Args:
        page: pdfplumber page object
        
    Returns:
        List of tuples: (rank, total_bid_amount) for ranks >= 1
        Excludes rank 0 (Engineer's Estimate)
    """
    bidders: List[Tuple[int, Decimal]] = []

    # Extract words with their coordinates
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False,
                               x_tolerance=2, y_tolerance=3)
    if not words:
        return bidders
    
    # Group words into rows based on vertical position
    rows = _group_rows(words, y_tol=3)

    # STEP 1: Check if this is a Vendor Ranking page
    # Look in first 20 rows for "vendor ranking" text
    page_text = " ".join(_norm_ws(w["text"]) for row in rows[:20] for w in row).lower()
    if "vendor ranking" not in page_text:
        return bidders
    
    # STEP 2: Find the header row
    # Expected columns: Rank | Vendor ID | Vendor Name | Total Bid | Percent Of Low Bid | Percent Of Estimate
    header_row = None
    for idx, row in enumerate(rows[:80]):  # Check first 80 rows
        joined = " ".join(_norm_ws(w["text"]) for w in row)
        key = _header_key(joined)
        # Look for a row that contains both "rank" and "vendor" keywords
        if "rank" in key and "vendor" in key:
            header_row = row
            header_idx = idx
            break
    
    if header_row is None:
        return bidders

    # STEP 3: Find column positions using phrase matching
    # Locate "Rank" column
    rank_span = _find_phrase_span(header_row, ["rank"])
    # Locate "Vendor Name" column (try multiple patterns)
    vendor_span = _find_phrase_span(header_row, ["vendor", "name"]) or _find_phrase_span(header_row, ["vendor"])
    # Locate "Total Bid" column (try both orderings)
    total_span = _find_phrase_span(header_row, ["total", "bid"]) or _find_phrase_span(header_row, ["bid", "total"])
    
    if not rank_span or not total_span:
        return bidders

    # Define search boundaries for each column (with tolerance)
    rank_left = rank_span[0] - 5
    rank_right = rank_span[1] + 15
    
    # Total Bid column is typically after Vendor Name
    total_left = total_span[0] - 10
    total_right = total_span[1] + 100

    # STEP 4: Parse data rows after header
    for row in rows[header_idx + 1:]:
        # Extract rank from words in the Rank column area
        r_words = [w for w in row if rank_left <= w["x0"] <= rank_right]
        rank = None
        for w in r_words:
            rank = _try_int(w["text"])
            if rank is not None:
                break
        
        # Skip invalid ranks:
        # - rank 0 is Engineer's Estimate (not an actual bidder)
        # - None means no valid rank found
        if rank is None or rank <= 0:
            continue

        # Extract total bid amount from words in the Total Bid column area
        t_words = [w for w in row if total_left <= w["x0"] <= total_right]
        # Filter to words that look like money (contain dollar amounts)
        cands = [w for w in t_words if _money_rx.search(_norm(w["text"]))]
        if not cands:
            continue
        
        # Try to parse the money amount
        amt = None
        for w in sorted(cands, key=lambda d: d["x0"]):
            amt = _parse_money(w["text"])
            # Valid bids should be at least $1,000
            if amt is not None and amt >= 1000:
                break
        
        if amt is None or amt < 1000:
            continue

        bidders.append((rank, amt))

    return bidders

# ============================================================================
# LINE-BASED EXTRACTION STRATEGY (FALLBACK)
# ============================================================================
# This approach uses regex patterns on extracted text lines.
# Used as fallback when word-coordinate approach fails.

# Regex to detect "Vendor Ranking" page
_vendor_ranking_rx = re.compile(r"vendor\s+ranking", re.I)
# Regex to parse data lines: "1 VENDOR-ID Vendor Name $123,456.78 98.5% 95.2%"
_rank_vendor_line_rx = re.compile(r"^\s*(\d{1,2})\s+\S+\s+.*?(\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+[\d.]+%\s+[\d.]+%", re.I)

def _extract_summary_from_page_lines(page) -> List[Tuple[int, Decimal]]:
    """
    Extract bidder rankings and amounts using line-based regex matching.
    
    This is a simpler fallback approach that:
    1. Extracts text line by line
    2. Checks for "Vendor Ranking" page marker
    3. Uses regex to parse lines with format: rank vendorID name $amount %low %est
    
    Args:
        page: pdfplumber page object
        
    Returns:
        List of tuples: (rank, total_bid_amount) for ranks >= 1
    """
    bidders: List[Tuple[int, Decimal]] = []
    
    # Extract text from page
    text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
    if not text:
        return bidders
    # Split into lines and remove null characters
    lines = [l for l in (t.strip("\x00") for t in text.splitlines()) if l is not None]

    # Check if this is a Vendor Ranking page (look in first 10 lines)
    is_vendor_ranking = any(_vendor_ranking_rx.search(ln) for ln in lines[:10])
    if not is_vendor_ranking:
        return bidders

    # Parse lines looking for pattern: rank vendorID vendorName $totalBid percentLow percentEst
    for ln in lines:
        # Skip row 0 (Engineer's Estimate) - typically starts with "0 -EST-" or "0 EST"
        if re.match(r"^\s*0\s+", ln):
            continue
        
        # Try to match the expected line format
        m = _rank_vendor_line_rx.match(ln)
        if not m:
            continue
        
        # Extract rank (group 1)
        rank = int(m.group(1))
        if rank <= 0:
            continue
        
        # Extract and parse bid amount (group 2)
        amt = _parse_money(m.group(2))
        if amt is None or amt < 1000:
            continue
        
        bidders.append((rank, amt))

    return bidders

# ============================================================================
# ORCHESTRATION - PDF PROCESSING AND OUTPUT
# ============================================================================

def _extract_two_metrics_from_pdf(pdf_path: Path) -> Tuple[int, Optional[Decimal]]:
    """
    Extract number of bidders and lowest bid from a PDF file.
    
    Strategy:
    1. Check first 4 pages for "Vendor Ranking" table
    2. Try word-bounded extraction first (more accurate)
    3. Fall back to line-based extraction if word-bounded fails
    4. Return first successful extraction found
    
    Args:
        pdf_path: Path to PDF file to process
        
    Returns:
        Tuple of (num_bidders, lowest_bid_amount)
        Returns (0, None) if extraction fails
    """
    num_bidders = 0
    lowest_bid: Optional[Decimal] = None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Only check first 4 pages (vendor ranking is typically on page 1-2)
            pages_to_check = min(len(pdf.pages), 4)
            for i in range(pages_to_check):
                page = pdf.pages[i]

                # Try word-based extraction first
                bidders = _extract_summary_from_page_words(page)
                if not bidders:
                    # Fall back to line-based extraction
                    bidders = _extract_summary_from_page_lines(page)

                # If we found bidders, extract metrics and stop
                if bidders:
                    amounts = [amt for _, amt in bidders]
                    num_bidders = len(bidders)
                    lowest_bid = min(amounts) if amounts else None
                    break
    except Exception:
        # Silently handle PDF parsing errors
        pass

    return num_bidders, lowest_bid

def _write_csv(rows: List[Tuple[str, int, Optional[Decimal]]], out_path: str) -> None:
    """
    Write results to CSV file.
    
    Args:
        rows: List of (file_path, num_bidders, lowest_bid) tuples
        out_path: Output CSV file path
    """
    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Write header
        w.writerow(["file", "num_bidders", "lowest_bid"])
        # Write data rows
        for file_path, n, low in rows:
            # Format lowest_bid as currency with 2 decimal places, or empty if None
            w.writerow([file_path, n, f"{low:.2f}" if low is not None else ""])

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """
    Main entry point for command-line execution.
    
    Processes all PDFs in the specified directory and writes results to CSV.
    """
    # Parse command-line arguments
    ap = argparse.ArgumentParser(description="Parse #bidders and winner's bid from CDOT summary tables.")
    ap.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR, help="Folder containing PDFs (default: data/pdf)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="CSV output path (default: data/bid_min_and_count.csv)")
    ap.add_argument("--f", default=None, help="(ignored; for notebook kernels)")
    args, unknown = ap.parse_known_args()
    if unknown:
        print("[INFO] Ignoring unknown args:", unknown)

    # Find all PDF files in the specified directory (recursive)
    pdfs = sorted(Path(args.pdf_dir).rglob("*.pdf"))
    results: List[Tuple[str, int, Optional[Decimal]]] = []
    
    # Process each PDF
    for pdf in pdfs:
        n, low = _extract_two_metrics_from_pdf(pdf)
        results.append((str(pdf), n, low))
        # Print progress for each file
        print(f"[OK] {Path(pdf).name}: bidders={n} | lowest={f'{low:.2f}' if low is not None else 'N/A'}")

    # Write results to CSV
    _write_csv(results, args.out)
    print(f"[DONE] Wrote {args.out} with {len(results)} rows.")

if __name__ == "__main__":
    main()