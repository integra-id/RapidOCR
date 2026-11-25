# -*- encoding: utf-8 -*-
"""Lightweight reading-order layout from OCR line boxes.

This is not a document model. It only groups boxes that already exist:
taller short lines become headings, bullet-like lines become lists, and
rows that share a column count become a simple table. Merged cells, ruling
lines, and nested columns are not recovered.
"""

from __future__ import annotations

import csv
import html
import io
import re
from typing import Any, Optional

import numpy as np

_LIST_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+\S")


def _props(box) -> dict[str, float]:
    array = np.asarray(box, dtype=float)
    xs = array[:, 0]
    ys = array[:, 1]
    top = float(np.min(ys))
    bottom = float(np.max(ys))
    left = float(np.min(xs))
    right = float(np.max(xs))
    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "height": max(1.0, bottom - top),
        "width": max(1.0, right - left),
        "center_y": (top + bottom) / 2,
    }


def _rows(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for line in lines:
        text = (line.get("text") or "").strip()
        box = line.get("box")
        if not text or box is None:
            continue
        items.append({"text": text, "props": _props(box)})
    items.sort(key=lambda item: (item["props"]["center_y"], item["props"]["left"]))
    rows: list[dict[str, Any]] = []
    for item in items:
        placed = False
        for row in rows:
            limit = max(item["props"]["height"], row["height"]) * 0.6
            if abs(item["props"]["center_y"] - row["center"]) <= limit:
                row["items"].append(item)
                row["center"] = sum(
                    entry["props"]["center_y"] for entry in row["items"]
                ) / len(row["items"])
                row["height"] = max(entry["props"]["height"] for entry in row["items"])
                placed = True
                break
        if not placed:
            rows.append(
                {
                    "items": [item],
                    "center": item["props"]["center_y"],
                    "height": item["props"]["height"],
                }
            )
    for row in rows:
        row["items"].sort(key=lambda item: item["props"]["left"])
    return rows


def detect_tables(lines: list[dict[str, Any]]) -> list[list[list[str]]]:
    """Return simple tables as rows of cell strings."""
    rows = _rows(lines)
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        width = len(row["items"])
        if width >= 2 and (not current or width == len(current[0]["items"])):
            current.append(row)
            continue
        if len(current) >= 2:
            groups.append(current)
        current = [row] if width >= 2 else []
    if len(current) >= 2:
        groups.append(current)
    tables = []
    for group in groups:
        if not _columns_align(group):
            continue
        tables.append([[item["text"] for item in row["items"]] for row in group])
    return tables


def _columns_align(group: list[dict[str, Any]]) -> bool:
    first = [item["props"]["left"] for item in group[0]["items"]]
    tolerance = max(item["props"]["width"] for item in group[0]["items"]) * 0.75
    for row in group[1:]:
        for item, origin in zip(row["items"], first):
            if abs(item["props"]["left"] - origin) > tolerance:
                return False
    return True


def tables_to_csv(tables: list[list[list[str]]]) -> str:
    if not tables:
        return "# no aligned table found\n"
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    for index, table in enumerate(tables):
        if index:
            writer.writerow([])
        for row in table:
            writer.writerow(row)
    text = buffer.getvalue()
    if not text.endswith("\n"):
        text += "\n"
    return text


def layout_markdown(lines: list[dict[str, Any]]) -> str:
    if not any(line.get("box") for line in lines):
        return "\n".join(line["text"] for line in lines if line.get("text"))
    return "\n\n".join(_blocks(lines, html_mode=False))


def layout_html(lines: list[dict[str, Any]]) -> str:
    if not lines:
        return ""
    if not any(line.get("box") for line in lines):
        return "".join(
            f"<p>{html.escape(line['text'])}</p>" for line in lines if line.get("text")
        )
    return "".join(_blocks(lines, html_mode=True))


def _blocks(lines: list[dict[str, Any]], html_mode: bool) -> list[str]:
    tables = detect_tables(lines)
    table_by_header = {_row_key(table[0]): table for table in tables}
    emitted: set[tuple[str, ...]] = set()
    rows = _rows(lines)
    median = _median_height(rows)
    blocks: list[str] = []
    for row in rows:
        key = _row_key([item["text"] for item in row["items"]])
        if key in emitted:
            continue
        table = table_by_header.get(key)
        if table is not None:
            blocks.append(
                _html_table(table)
                if html_mode
                else _markdown_table(table[0], table[1:])
            )
            for table_row in table:
                emitted.add(_row_key(table_row))
            continue
        text = " ".join(item["text"] for item in row["items"]).strip()
        if not text:
            continue
        if _LIST_RE.match(text):
            blocks.append(f"<li>{html.escape(text)}</li>" if html_mode else text)
        elif len(text) <= 80 and row["height"] > median * 1.45:
            blocks.append(
                f"<h2>{html.escape(text)}</h2>" if html_mode else f"## {text}"
            )
        else:
            blocks.append(f"<p>{html.escape(text)}</p>" if html_mode else text)
    return blocks


def _markdown_table(header: list[str], rest: list[list[str]]) -> str:
    cells = [_escape_cell(cell) for cell in header]
    lines = [
        "| " + " | ".join(cells) + " |",
        "| " + " | ".join("---" for _ in cells) + " |",
    ]
    for row in rest:
        lines.append("| " + " | ".join(_escape_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _html_table(table: list[list[str]]) -> str:
    rows = []
    for index, row in enumerate(table):
        tag = "th" if index == 0 else "td"
        body = "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in row)
        rows.append(f"<tr>{body}</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _row_key(row: list[str]) -> tuple[str, ...]:
    return tuple(row)


def _median_height(rows: list[dict[str, Any]]) -> float:
    heights = sorted(row["height"] for row in rows) or [1.0]
    return heights[len(heights) // 2]


def render_csv(pages: list[dict[str, Any]]) -> str:
    tables: list[list[list[str]]] = []
    for page in pages:
        for table in page.get("tables") or []:
            tables.append(table)
    return tables_to_csv(tables)
