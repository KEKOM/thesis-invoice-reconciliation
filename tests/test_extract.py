"""
Tests for src/extract.py.

Fast unit tests use plain text strings and a programmatically generated PDF
(no external fixtures needed).  Fixture-based tests are skipped when the
fixture PDF files haven't been placed in tests/fixtures/ yet.
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.extract import (
    _parse_header_fields,
    _parse_line_items_from_tables,
    _parse_line_items_from_text,
    extract_invoice,
    extract_invoices,
)

FIXTURES = Path(__file__).parent / "fixtures"

EXPECTED_COLS = {
    "left_invoice_number", "left_date", "left_vendor",
    "left_description", "left_quantity", "left_unit_price", "left_total",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_invoice_pdf(text_lines: list[str]) -> str:
    """
    Write a minimal but valid PDF containing the given text lines and return
    the path to a temporary file.  Uses only stdlib — no reportlab/fpdf2.
    """
    # Build content stream: each line is a separate Td move so pdfplumber
    # can extract them as distinct text runs.
    ops = ["BT /F1 12 Tf 50 750 Td"]
    for i, line in enumerate(text_lines):
        # Escape PDF string special chars: backslash, parentheses
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if i == 0:
            ops.append(f"({safe}) Tj")
        else:
            ops.append(f"0 -20 Td ({safe}) Tj")
    ops.append("ET")
    content = "\n".join(ops).encode()

    parts: list[bytes] = []
    offsets: dict[int, int] = {}

    parts.append(b"%PDF-1.4\n")
    offsets[1] = sum(len(p) for p in parts)
    parts.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    offsets[2] = sum(len(p) for p in parts)
    parts.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    offsets[3] = sum(len(p) for p in parts)
    parts.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    offsets[4] = sum(len(p) for p in parts)
    stream_hdr = f"4 0 obj\n<< /Length {len(content)} >>\nstream\n".encode()
    parts.append(stream_hdr + content + b"\nendstream\nendobj\n")
    offsets[5] = sum(len(p) for p in parts)
    parts.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    xref_pos = sum(len(p) for p in parts)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for i in range(1, 6):
        xref += f"{offsets[i]:010d} 00000 n \n".encode()
    parts.append(xref)
    parts.append(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode())

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(b"".join(parts))
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Unit tests: parsing helpers (no PDF required)
# ---------------------------------------------------------------------------

class TestParseHeaderFields:
    def test_invoice_number_extracted(self):
        text = "Invoice No. INV-2024-001\nDate: 01/15/2024\nFrom: ACME Corp"
        h = _parse_header_fields(text)
        assert h["invoice_number"] == "INV-2024-001"

    def test_date_extracted_slash_format(self):
        text = "Invoice # 42\nDate: 12/31/2023"
        h = _parse_header_fields(text)
        assert "12/31/2023" in h["date"]

    def test_date_extracted_iso_format(self):
        text = "Invoice number: 007\n2024-06-01"
        h = _parse_header_fields(text)
        assert "2024-06-01" in h["date"]

    def test_vendor_extracted(self):
        text = "From: Widget Supplies Ltd\nInvoice No. X-99"
        h = _parse_header_fields(text)
        assert "Widget Supplies" in h["vendor"]

    def test_missing_fields_are_empty_strings(self):
        h = _parse_header_fields("just some random text")
        assert h["invoice_number"] == ""
        assert h["date"] == ""
        assert h["vendor"] == ""


class TestParseLineItemsFromText:
    def test_detects_two_amount_line(self):
        text = "Widget A  25.00  50.00"
        items = _parse_line_items_from_text(text)
        assert len(items) == 1
        assert items[0]["unit_price"] == "25.00"
        assert items[0]["total"] == "50.00"

    def test_multiple_line_items(self):
        text = "ItemA  10.00  20.00\nItemB  5.00  15.00\n"
        items = _parse_line_items_from_text(text)
        assert len(items) == 2

    def test_line_with_single_amount_ignored(self):
        items = _parse_line_items_from_text("Total: 100.00")
        # Only one amount — shouldn't produce a line item
        assert len(items) == 0

    def test_returns_list_of_dicts(self):
        items = _parse_line_items_from_text("Service  30.00  60.00")
        assert isinstance(items, list)
        assert "unit_price" in items[0]
        assert "total" in items[0]


class TestParseLineItemsFromTables:
    def test_recognises_description_and_total_columns(self):
        table = [
            ["Description", "Qty", "Unit Price", "Total"],
            ["Widget A", "2", "25.00", "50.00"],
            ["Widget B", "1", "10.00", "10.00"],
        ]
        items = _parse_line_items_from_tables([table])
        assert len(items) == 2
        assert items[0]["description"] == "Widget A"
        assert items[0]["total"] == "50.00"

    def test_empty_table_ignored(self):
        assert _parse_line_items_from_tables([[]]) == []

    def test_table_without_known_header_ignored(self):
        table = [["Col1", "Col2"], ["A", "B"]]
        assert _parse_line_items_from_tables([table]) == []

    def test_skips_blank_data_rows(self):
        table = [
            ["Description", "Total"],
            [None, None],
            ["Widget", "50.00"],
        ]
        items = _parse_line_items_from_tables([table])
        assert len(items) == 1


# ---------------------------------------------------------------------------
# Integration tests: full extraction from a synthetic PDF
# ---------------------------------------------------------------------------

class TestExtractInvoice:
    def setup_method(self):
        pytest.importorskip("pdfplumber")

    def test_returns_dataframe_with_left_schema(self, tmp_path):
        pdf = _make_invoice_pdf([
            "Invoice No. INV-001",
            "Date: 01/15/2024",
            "From: ACME Corp",
            "Widget A  25.00  50.00",
        ])
        try:
            df = extract_invoice(pdf)
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert EXPECTED_COLS.issubset(set(df.columns))
        finally:
            os.unlink(pdf)

    def test_at_least_one_row_always_returned(self):
        """Even a blank PDF produces a placeholder row."""
        pdf = _make_invoice_pdf(["no amounts here"])
        try:
            df = extract_invoice(pdf)
            assert len(df) >= 1
            assert EXPECTED_COLS.issubset(set(df.columns))
        finally:
            os.unlink(pdf)

    def test_invoice_number_parsed(self):
        pdf = _make_invoice_pdf(["Invoice No. INV-2024-007", "Item  10.00  20.00"])
        try:
            df = extract_invoice(pdf)
            assert df["left_invoice_number"].iloc[0] == "INV-2024-007"
        finally:
            os.unlink(pdf)

    def test_vendor_parsed(self):
        pdf = _make_invoice_pdf(["From: ACME Corp", "Item  10.00  20.00"])
        try:
            df = extract_invoice(pdf)
            assert "ACME" in df["left_vendor"].iloc[0]
        finally:
            os.unlink(pdf)

    def test_ocr_fallback_called_when_no_text(self):
        """Verify pytesseract is invoked when pdfplumber returns empty text."""
        pdf = _make_invoice_pdf(["Invoice No. INV-001"])

        fake_ocr_text = "From: OCR Vendor\nInvoice No. INV-999\nItem  5.00  10.00"

        with patch("src.extract._extract_text_pdfplumber", return_value=("", [])):
            with patch("src.extract._extract_text_ocr", return_value=fake_ocr_text) as mock_ocr:
                df = extract_invoice(pdf)
                mock_ocr.assert_called_once()
                assert df["left_invoice_number"].iloc[0] == "INV-999"

        os.unlink(pdf)

    def test_extract_invoices_concatenates(self):
        pdfs = [
            _make_invoice_pdf(["Invoice No. INV-001", "ItemA  10.00  20.00"]),
            _make_invoice_pdf(["Invoice No. INV-002", "ItemB  5.00  10.00"]),
        ]
        try:
            df = extract_invoices(pdfs)
            assert len(df) >= 2
            assert EXPECTED_COLS.issubset(set(df.columns))
        finally:
            for p in pdfs:
                os.unlink(p)


# ---------------------------------------------------------------------------
# Fixture-based tests (skipped until the user adds PDFs to tests/fixtures/)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (FIXTURES / "sample_digital.pdf").exists(),
    reason="tests/fixtures/sample_digital.pdf not present",
)
def test_fixture_digital_pdf():
    df = extract_invoice(str(FIXTURES / "sample_digital.pdf"))
    assert not df.empty
    assert EXPECTED_COLS.issubset(set(df.columns))


@pytest.mark.skipif(
    not (FIXTURES / "sample_scanned.pdf").exists(),
    reason="tests/fixtures/sample_scanned.pdf not present",
)
def test_fixture_scanned_pdf():
    pytest.importorskip("pytesseract")
    pytest.importorskip("pdf2image")
    df = extract_invoice(str(FIXTURES / "sample_scanned.pdf"))
    assert not df.empty
    assert EXPECTED_COLS.issubset(set(df.columns))
