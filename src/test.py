import re
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

PDF_PATH = "data/pdf/Project_19771_Bid_Tab.pdf"


def looks_cid_garbled(text: str, threshold: float = 0.005) -> bool:
    """
    Heuristic: if we see (cid:XX) or lots of weird control chars,
    we treat the text as unusable.
    """
    if not text:
        return True

    cid_matches = len(re.findall(r"\(cid:\d+\)", text))
    if cid_matches > 0:
        return True

    # many non-printables → suspect garbage
    non_printable = sum(
        1 for ch in text
        if (ord(ch) < 32 and ch not in "\n\r\t") or ord(ch) == 127
    )
    return non_printable / len(text) > threshold


def ocr_first_pages(pdf_path: str, max_pages: int = 2) -> str:
    print("[INFO] Running OCR with Tesseract...")
    pages = convert_from_path(pdf_path, dpi=300)
    all_text = []
    for i, img in enumerate(pages[:max_pages]):
        print(f"  [OCR] Page {i+1}/{len(pages)}")
        txt = pytesseract.image_to_string(img)
        all_text.append(txt)
    return "\n".join(all_text)


def debug_print_page(pdf_path: str):
    print(f"--- ANALYSIS: {pdf_path} ---")

    # 1) Try pdfplumber for first 2 pages
    pdf_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:2]:
                t = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                pdf_text += t + "\n"
    except Exception as e:
        print(f"[WARN] pdfplumber failed: {e}")

    # 2) If pdfplumber output is garbage → OCR
    if looks_cid_garbled(pdf_text):
        print("[WARN] pdfplumber text looks garbled or CID-encoded. Using OCR fallback.\n")
        text = ocr_first_pages(pdf_path, max_pages=2)
    else:
        print("[INFO] pdfplumber text looks usable. Showing that.\n")
        text = pdf_text

    # 3) Print first lines of whatever text we ended up with
    lines = text.splitlines()
    for idx, line in enumerate(lines[:80]):
        print(f"{idx:03d}: {repr(line)}")


if __name__ == "__main__":
    debug_print_page(PDF_PATH)