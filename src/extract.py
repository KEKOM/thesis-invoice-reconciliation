"""
Stage 1: PDF invoice extraction.

Tries pdfplumber first for digital/text PDFs; falls back to pytesseract OCR
for scanned or image-only PDFs (no text layer detected).

Returns one DataFrame row per line item in the left_* schema expected by
the matcher (left_invoice_number, left_date, left_vendor, left_description,
left_quantity, left_unit_price, left_total).
"""
from __future__ import annotations
import re
from typing import Any

import pandas as pd

_DATE_RE = re.compile(
    r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|\d{4}[/\-]\d{1,2}[/\-]\d{1,2}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_INV_NUM_RE = re.compile(
    r"(?:invoice\s*(?:no\.?|number|#|num\.?)\s*[:#]?\s*)([A-Z0-9\-]+)",
    re.IGNORECASE,
)
_VENDOR_RE = re.compile(
    r"(?:from|vendor|supplier|bill(?:ed)?\s+(?:from|by))\s*[:\-]?\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"[\$£€]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")


def _extract_text_pdfplumber(pdf_path: str) -> tuple[str, list[Any]]:
    """Return (full_text, list_of_tables) using pdfplumber."""
    import pdfplumber  # optional dependency

    full_text: list[str] = []
    all_tables: list[Any] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            full_text.append(text)
            all_tables.extend(page.extract_tables() or [])
    return "\n".join(full_text), all_tables


def _extract_text_ocr(pdf_path: str) -> str:
    """Rasterise pages with pdf2image and OCR with pytesseract."""
    import pytesseract  # optional dependency
    from pdf2image import convert_from_path  # optional dependency

    images = convert_from_path(pdf_path, dpi=200)
    return "\n".join(pytesseract.image_to_string(img) for img in images)


def _parse_header_fields(text: str) -> dict[str, str]:
    inv_m = _INV_NUM_RE.search(text)
    date_m = _DATE_RE.search(text)
    vendor_m = _VENDOR_RE.search(text)
    return {
        "invoice_number": inv_m.group(1).strip() if inv_m else "",
        "date": date_m.group(0).strip() if date_m else "",
        "vendor": vendor_m.group(1).strip()[:80] if vendor_m else "",
    }


def _parse_line_items_from_tables(tables: list[Any]) -> list[dict]:
    """Extract line items from pdfplumber table structures."""
    rows: list[dict] = []
    for table in tables:
        if not table or len(table) < 2:
            continue
        header = [str(c).lower().strip() if c else "" for c in table[0]]
        col_map: dict[str, int] = {}
        for i, h in enumerate(header):
            if any(k in h for k in ("desc", "item", "product", "service", "particular")):
                col_map.setdefault("description", i)
            elif any(k in h for k in ("qty", "quantity", "units")):
                col_map.setdefault("quantity", i)
            elif any(k in h for k in ("unit price", "unit_price", "rate")):
                col_map.setdefault("unit_price", i)
            elif "price" in h and "unit" not in h:
                col_map.setdefault("unit_price", i)
            elif any(k in h for k in ("total", "amount", "subtotal")):
                col_map.setdefault("total", i)
        if not col_map:
            continue
        for data_row in table[1:]:
            if not any(data_row):
                continue
            item: dict[str, str] = {}
            for field, idx in col_map.items():
                item[field] = str(data_row[idx]).strip() if idx < len(data_row) and data_row[idx] else ""
            if item.get("description"):
                rows.append(item)
    return rows


def _parse_line_items_from_text(text: str) -> list[dict]:
    """
    Heuristic fallback: any line whose last two tokens look like amounts
    (unit_price, total) is treated as a line item.
    """
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        amounts = _AMOUNT_RE.findall(line.replace(",", ""))
        if len(amounts) >= 2:
            description = _AMOUNT_RE.sub("", line).strip(" $£€\t|:-")
            rows.append({
                "description": description,
                "quantity": "",
                "unit_price": amounts[-2],
                "total": amounts[-1],
            })
    return rows


def extract_invoice(pdf_path: str) -> pd.DataFrame:
    """
    Extract structured fields from a single invoice PDF.
    Returns one row per line item with left_* column names.
    Falls back to pytesseract OCR when pdfplumber finds no text layer.
    """
    # pdfplumber pass
    try:
        text, tables = _extract_text_pdfplumber(str(pdf_path))
    except ImportError:
        text, tables = "", []

    # OCR fallback
    if not text.strip():
        try:
            text = _extract_text_ocr(str(pdf_path))
        except ImportError:
            pass
        tables = []

    header = _parse_header_fields(text)

    items = _parse_line_items_from_tables(tables) if tables else []
    if not items:
        items = _parse_line_items_from_text(text)

    # Always return at least one placeholder row so the invoice isn't silently dropped
    if not items:
        items = [{"description": "", "quantity": "", "unit_price": "", "total": ""}]

    rows = [
        {
            "left_invoice_number": header["invoice_number"],
            "left_date": header["date"],
            "left_vendor": header["vendor"],
            "left_description": item.get("description", ""),
            "left_quantity": item.get("quantity", ""),
            "left_unit_price": item.get("unit_price", ""),
            "left_total": item.get("total", ""),
        }
        for item in items
    ]
    return pd.DataFrame(rows)


def extract_invoices(pdf_paths: list[str]) -> pd.DataFrame:
    """Extract from multiple PDFs and concatenate into one DataFrame."""
    frames = [extract_invoice(p) for p in pdf_paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
