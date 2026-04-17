"""Rule-based metrics for the widesearch PARL v2 environment.

Two reward heads, picked per-sample by reward.py:

- ``row_f1_from_markdown(response, answer, unique_columns)``: the main signal
  on WideSeek-R1 / WideSearch data. Both ``response`` and ``answer`` are
  expected to contain a markdown table; we extract the ``unique_columns``
  projection from each, normalize, and compute F1 over the set of row
  identity tuples. Falls back to 0.0 when either side lacks a parseable table.
  This is stricter than paper's item-F1 (which also judges non-key cells via
  LLM), but fully rule-based — directional gradient, zero external deps.

- ``em_score(response, answer)``: normalized exact-match used on the
  ASearcher QA benchmarks (HotpotQA / 2Wiki / NQ / …). The model's final
  answer is expected either inside ``\\boxed{}`` or inside a
  ``<answer>…</answer>`` block; we search both, then EM against the GT.

Both helpers are sync / pure / side-effect-free so they can run inside the
reward hot path without async plumbing.
"""

from __future__ import annotations

import re
import string
import unicodedata

_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$", re.MULTILINE)
_SEPARATOR_ROW_RE = re.compile(r"^\s*:?-{2,}:?\s*$")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _norm_column(s: str) -> str:
    """Strip + lowercase + collapse whitespace, for column-name alignment."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_cell(s: str) -> str:
    """Strip + lowercase + drop surrounding punctuation for cell comparison.

    Kept simple on purpose — too aggressive a normalization (e.g. stripping
    all punctuation) would collapse semantically distinct values like
    "2020" vs "$2020". Instead we only strip trailing punctuation and
    collapse internal whitespace.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.strip("`*_ ").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _extract_markdown_tables(text: str) -> list[list[dict[str, str]]]:
    """Return every markdown table found in ``text`` as a list of row-dicts.

    A table = consecutive lines each matching ``| a | b | ...|``. The second
    row is allowed to be a separator (``| --- | --- |``) which we skip.
    Tables without at least one data row are dropped.
    """
    tables: list[list[dict[str, str]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _TABLE_ROW_RE.match(lines[i])
        if not m:
            i += 1
            continue
        # collect consecutive pipe-rows
        block = []
        while i < len(lines):
            bm = _TABLE_ROW_RE.match(lines[i])
            if not bm:
                break
            block.append([c.strip() for c in bm.group(1).split("|")])
            i += 1
        if len(block) < 2:
            continue
        header = [_norm_column(c) for c in block[0]]
        data_rows = block[1:]
        if data_rows and all(_SEPARATOR_ROW_RE.match(c) for c in data_rows[0] if c):
            data_rows = data_rows[1:]
        rows: list[dict[str, str]] = []
        for row in data_rows:
            if len(row) != len(header):
                # truncate or pad so keys align; drop malformed silently
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                else:
                    row = row[: len(header)]
            rows.append({h: v for h, v in zip(header, row, strict=False) if h})
        if rows:
            tables.append(rows)
    return tables


def _pick_table_with_columns(tables: list[list[dict[str, str]]], required: list[str]) -> list[dict[str, str]] | None:
    """Return the first table that contains every required column, else None."""
    required_set = set(required)
    for t in tables:
        if t and required_set.issubset(t[0].keys()):
            return t
    return None


def _row_key(row: dict[str, str], unique_cols: list[str]) -> tuple[str, ...]:
    return tuple(_norm_cell(row.get(c, "")) for c in unique_cols)


def row_f1_from_markdown(response: str, answer: str, unique_columns: list[str] | tuple[str, ...]) -> float:
    """Rule-based row-F1 on the ``unique_columns`` projection.

    Returns 0.0 on any parse failure (no table, missing columns, etc.)
    instead of raising — we don't want bad formatting to crash a training
    step.
    """
    if not unique_columns:
        return 0.0
    uc = [_norm_column(c) for c in unique_columns]
    gt_tables = _extract_markdown_tables(answer or "")
    pred_tables = _extract_markdown_tables(response or "")
    gt = _pick_table_with_columns(gt_tables, uc)
    pred = _pick_table_with_columns(pred_tables, uc)
    if gt is None or pred is None:
        return 0.0

    gt_keys = {_row_key(r, uc) for r in gt if any(r.get(c) for c in uc)}
    pred_keys = {_row_key(r, uc) for r in pred if any(r.get(c) for c in uc)}
    if not gt_keys or not pred_keys:
        return 0.0
    tp = len(gt_keys & pred_keys)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_keys)
    recall = tp / len(gt_keys)
    return 2 * precision * recall / (precision + recall)


def _extract_final_answer(response: str) -> str:
    """Pull the model's final answer string out of a response.

    Priority: last ``\\boxed{...}`` match, then last ``<answer>…</answer>``
    block, then fall back to the response tail (last 200 chars) stripped of
    think-trace noise.
    """
    if not response:
        return ""
    boxes = _BOXED_RE.findall(response)
    if boxes:
        return boxes[-1].strip()
    tags = _ANSWER_TAG_RE.findall(response)
    if tags:
        return tags[-1].strip()
    return response[-200:].strip()


def _normalize_em(s: str) -> str:
    """QA-style normalization: lowercase, strip articles + punctuation + ws."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def em_score(response: str, answer: str | list[str]) -> float:
    """Normalized exact-match against GT (which may be a list of aliases)."""
    pred = _normalize_em(_extract_final_answer(response))
    if not pred:
        return 0.0
    if isinstance(answer, (list, tuple)):
        gts = [_normalize_em(str(a)) for a in answer]
    else:
        gts = [_normalize_em(str(answer))]
    return 1.0 if any(g and g == pred for g in gts) else 0.0
