import re
from pathlib import Path
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple
import pandas as pd
import pdfplumber
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default paths
DEFAULT_PDF_DIR = "data/pdf"
DEFAULT_OUT = "data/bid_min_and_count.csv"

# Regex to match money amounts: $1,234.56 or 1234.56
_money_rx = re.compile(r"\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
# Regex to detect "Vendor Ranking" page
_vendor_ranking_rx = re.compile(r"vendor\s+ranking", re.I)
# Regex to parse data lines: "1 VENDOR-ID Vendor Name $123,456.78 98.5% 95.2%"
_rank_vendor_line_rx = re.compile(r"^\s*(\d{1,2})\s+\S+\s+.*?(\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+[\d.]+%\s+[\d.]+%",re.I)


def _norm(s: str) -> str:
    """Strip whitespace from string."""
    return (s or "").strip()


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


# Uses regex patterns on extracted text lines to parse vendor ranking tables.

def _extract_summary_from_page_lines(page) -> List[Tuple[int, Decimal]]:
    """
    Extract bidder rankings and amounts using line-based regex matching.
    
    This method:
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


def _extract_two_metrics_from_pdf(pdf_path: Path) -> Tuple[int, Optional[Decimal]]:
    """
    Extract number of bidders and lowest bid from a PDF file.
    
    1. Check first 4 pages for "Vendor Ranking" table
    2. Use line-based regex extraction
    3. Return first successful extraction found
    
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

                # Extract using line-based method
                bidders = _extract_summary_from_page_lines(page)

                # If we found bidders, extract metrics and stop
                if bidders:
                    amounts = [amt for _, amt in bidders]
                    num_bidders = len(bidders)
                    lowest_bid = min(amounts) if amounts else None
                    break
    except Exception:
        logger.warning(f"Failed to process PDF: {pdf_path}", exc_info=True)

    return num_bidders, lowest_bid


def main(pdf_dir: str = DEFAULT_PDF_DIR, out_path: str = DEFAULT_OUT) -> pd.DataFrame:
    """
    Parses PDFs and return results as DataFrame.
    
    Args:
        pdf_dir: Directory containing PDF files (default: data/pdf)
        out_path: Output CSV path (default: data/bid_min_and_count.csv)

    Returns:
        DataFrame with columns: file, num_bidders, lowest_bid
    """
    # Find all PDF files in the specified directory
    pdfs = sorted(Path(pdf_dir).rglob("*.pdf"))
    results: List[Tuple[str, int, Optional[Decimal]]] = []

    # Process each PDF
    for pdf in pdfs:
        n, low = _extract_two_metrics_from_pdf(pdf)
        results.append((str(pdf), n, low))
        # Print progress for each file
        logger.debug(f"Saved {Path(pdf).name}: bidders={n} | lowest={f'{low:.2f}' if low is not None else 'N/A'}")

    # Create DataFrame
    df = pd.DataFrame(results, columns=["file", "num_bidders", "lowest_bid"])

    # Write to CSV
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Success: wrote {out_path} with {len(results)} rows.")
    # _write_csv(results, args.out)
    return df


if __name__ == "__main__":
    main()
