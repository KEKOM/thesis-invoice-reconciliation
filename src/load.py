"""
Data loading.

Supports two input formats:

1. ER-Magellan ".txt" format (default sample data):
   each line is  <record_left> \\t <record_right> \\t <label>
   where a record is serialised as  "COL <attr> VAL <value> COL <attr> VAL <value> ..."

2. Generic CSV: a single CSV where left-record columns are prefixed `left_`,
   right-record columns are prefixed `right_`, and there is a `label` column
   (1 = match / approve, 0 = no-match / flag). This is the format you will use
   when you plug in a Kaggle invoice dataset (see README).

Both formats are parsed into the SAME tidy DataFrame so the rest of the
pipeline does not care where the data came from.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

_COLVAL = re.compile(r"COL (.*?) VAL (.*?)(?= COL |$)")


def _parse_record(serialised: str) -> dict:
    """Turn 'COL title VAL x COL price VAL 9' into {'title': 'x', 'price': '9'}."""
    out = {}
    for attr, val in _COLVAL.findall(serialised.strip()):
        out[attr.strip()] = val.strip()
    return out


def load_ermagellan(path: str) -> pd.DataFrame:
    """Load one ER-Magellan split into a tidy pair DataFrame."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            left, right, label = parts
            lrec = _parse_record(left)
            rrec = _parse_record(right)
            row = {f"left_{k}": v for k, v in lrec.items()}
            row.update({f"right_{k}": v for k, v in rrec.items()})
            row["label"] = int(label)
            rows.append(row)
    return pd.DataFrame(rows)


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a pairs CSV into a tidy left_*/right_*/label DataFrame.

    Handles two layouts:
    - Pre-joined: columns already named left_<attr>, right_<attr>, label.
    - Magellan pairs table: columns ltable_id, rtable_id, label.
      tableA.csv and tableB.csv must exist in the same directory; the function
      joins them automatically and prefixes columns with left_/right_.
    """
    p = Path(path)
    df = pd.read_csv(p)
    if "label" not in df.columns:
        raise ValueError(f"CSV must contain a 'label' column: {path}")

    if "ltable_id" in df.columns and "rtable_id" in df.columns:
        table_a = pd.read_csv(p.parent / "tableA.csv").add_prefix("left_")
        table_b = pd.read_csv(p.parent / "tableB.csv").add_prefix("right_")
        table_a = table_a.rename(columns={"left_id": "ltable_id"})
        table_b = table_b.rename(columns={"right_id": "rtable_id"})
        df = (
            df.merge(table_a, on="ltable_id")
              .merge(table_b, on="rtable_id")
              .drop(columns=["ltable_id", "rtable_id"])
        )

    return df


def load_split(path: str) -> pd.DataFrame:
    if path.endswith(".txt"):
        return load_ermagellan(path)
    return load_csv(path)
