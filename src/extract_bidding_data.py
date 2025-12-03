import re
from pathlib import Path
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import logging

logger = logging.getLogger(__name__)

# Regex patterns
_vendor_ranking_rx = re.compile(r"vendor\s+ranking", re.I)
_rank_pattern = re.compile(
    r'^\s*(\d+)\s+'  # Rank
    r'[\$\w\-]+'  # Vendor ID
    r'\s+.*?'  # Vendor name
    r'\$?\s*([\d,\s]+(?:\.\d{2})?)'  # Amount
    r'\s+[\d.]+%',  # Percentage
    re.IGNORECASE
)


def _parse_money(s: str) -> Optional[Decimal]:
    """Parse money string to Decimal, handling spaces, commas, and dollar signs."""
    if not s:
        return None
    # Remove all formatting characters
    s = s.replace("$", "").replace(",", "").replace(" ", "").strip()
    if s in {"", "-", "—"}:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from first 2 pages, using OCR if CID-encoded."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = min(len(pdf.pages), 2)
            text_parts = []

            for i in range(pages_to_check):
                page = pdf.pages[i]
                text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""

                # Check if CID-encoded
                if "(cid:" in text:
                    logger.debug(f"CID-encoded PDF, using OCR: {pdf_path}")
                    pages = convert_from_path(pdf_path, dpi=300)
                    return "\n".join(
                        pytesseract.image_to_string(img)
                        for img in pages[:pages_to_check]
                    )

                text_parts.append(text)
            return "\n".join(text_parts)
    except Exception:
        logger.warning(f"Failed to extract text from PDF: {pdf_path}", exc_info=True)
        return ""


def extract_metrics_from_pdf(pdf_path: Path) -> Tuple[int, Optional[Decimal]]:
    """Extract number of bidders and lowest bid from PDF."""
    # Extract text
    text = _extract_text_from_pdf(pdf_path)
    if not text:
        return 0, None

    # Check for vendor ranking section (search entire text in case it spans lines)
    if not _vendor_ranking_rx.search(text):
        logger.info(f"No vendor ranking found in PDF: {pdf_path}")
        return 0, None

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Parse bidder lines
    bidders = []
    for ln in lines:
        # Skip rank 0 (engineer's estimate) and headers
        if re.match(r"^\s*0\s+", ln) or "Rank" in ln or "Vendor ID" in ln:
            continue

        m = _rank_pattern.match(ln)
        if not m:
            continue

        rank = int(m.group(1))
        if rank < 1:  # skip the engineer's estimate or invalid ranks
            continue

        amt = _parse_money(m.group(2))
        if amt is None:
            continue

        bidders.append((rank, amt))

    if not bidders:
        logger.info(f"No bidder data found in PDF: {pdf_path}")
        return 0, None

    num_bidders = len(bidders)-1 if len(bidders) > 1 else len(bidders)
    lowest_bid = min(amt for _, amt in bidders)

    return num_bidders, lowest_bid
