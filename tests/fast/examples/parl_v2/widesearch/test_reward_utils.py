"""Unit tests for the widesearch rule-based reward helpers.

Focus: make sure the Tier-1 upgrade (row-F1 → item-F1 + URL canonicalize +
<br>/newline split) changes behavior in the intended direction and does not
introduce high-FP shortcuts.
"""

from __future__ import annotations

import pytest

from examples.parl_v2.widesearch.reward_utils import (
    _canonicalize_url,
    _cell_set,
    cell_equal,
    em_score,
    item_f1_from_markdown,
)


class TestCanonicalizeUrl:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://www.harvard.edu/", "https://www.harvard.edu"),
            ("https://www.harvard.edu", "https://www.harvard.edu"),
            ("https://www.Harvard.EDU/", "https://www.harvard.edu"),
            ("HTTPS://example.com/path/", "https://example.com/path"),
            (
                "https://example.com/a/b/?q=1#frag",
                "https://example.com/a/b?q=1#frag",
            ),
            ("http://x.y/z", "http://x.y/z"),
        ],
    )
    def test_canonicalize(self, raw, expected):
        assert _canonicalize_url(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "not a url",
            "ftp://example.com/",
            "January 1",
            "$85",
            None,
        ],
    )
    def test_non_url_returns_none(self, raw):
        assert _canonicalize_url(raw) is None


class TestCellSet:
    def test_empty(self):
        assert _cell_set("") == frozenset()
        assert _cell_set(None) == frozenset()

    def test_single_value(self):
        assert _cell_set("Harvard") == frozenset({"harvard"})

    def test_br_split(self):
        assert _cell_set("A<br>B<br>C") == frozenset({"a", "b", "c"})

    def test_br_order_invariant(self):
        assert _cell_set("A<br>B") == _cell_set("B<br>A")

    def test_newline_split(self):
        assert _cell_set("A\nB") == frozenset({"a", "b"})

    def test_br_and_newline_mixed(self):
        assert _cell_set("A<br>B\nC") == frozenset({"a", "b", "c"})

    def test_whitespace_fragments_dropped(self):
        # trailing <br> / blank lines must not introduce empty set elements
        assert _cell_set("A<br>") == frozenset({"a"})
        assert _cell_set("A\n\nB") == frozenset({"a", "b"})


class TestCellEqual:
    @pytest.mark.parametrize(
        "pred,gt",
        [
            ("Harvard University", "Harvard University"),
            ("Harvard University", "harvard university"),  # case insensitive
            ("  Harvard  ", "Harvard"),  # whitespace
            ("https://www.harvard.edu/", "https://www.harvard.edu"),  # URL slash
            ("https://Harvard.edu", "https://harvard.edu"),  # URL host case
            ("A<br>B<br>C", "C<br>B<br>A"),  # multi-value order
            ("A<br>B", "A\nB"),  # <br> vs newline
            ("40.0%", "40.0%"),
            ("", ""),
        ],
    )
    def test_equal(self, pred, gt):
        assert cell_equal(pred, gt), f"expected {pred!r} == {gt!r}"

    @pytest.mark.parametrize(
        "pred,gt",
        [
            ("Harvard", "Yale"),  # different entity
            ("https://harvard.edu", "https://yale.edu"),  # different URL
            ("https://harvard.edu/a", "https://harvard.edu/b"),  # path differs
            ("A<br>B", "A<br>B<br>C"),  # subset != set equal
            ("A<br>B", "A"),  # multi vs single
            ("40.0%", "40%"),  # percent format not canonicalized (Tier 2)
            ("$85", "85"),  # currency symbol matters
            ("8 miles", "8 km"),  # unit matters (Tier 2)
            ("January 1", "Jan 1"),  # date format not canonicalized (Tier 2)
        ],
    )
    def test_not_equal(self, pred, gt):
        assert not cell_equal(pred, gt), f"expected {pred!r} != {gt!r}"


def _md_table(rows: list[list[str]]) -> str:
    """Helper: build a markdown table string from a list of rows (including header)."""
    sep = ["---"] * len(rows[0])
    lines = [rows[0], sep] + rows[1:]
    return "\n".join("| " + " | ".join(r) + " |" for r in lines)


class TestItemF1:
    def test_perfect_match(self):
        table = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard University", "1"],
                ["Arts", "Oxford", "2"],
            ]
        )
        # required defaults to full GT header, all cells match.
        assert item_f1_from_markdown(table, table, ["subject", "university"]) == 1.0

    def test_one_cell_wrong(self):
        gt = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
                ["Arts", "Oxford", "2"],
            ]
        )
        pred = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "99"],  # wrong rank
                ["Arts", "Oxford", "2"],
            ]
        )
        # 2 rows × 3 cols = 6 total cells; 5 TPs (one wrong rank).
        # precision = recall = 5/6 → f1 = 5/6.
        result = item_f1_from_markdown(pred, gt, ["subject", "university"])
        assert abs(result - 5 / 6) < 1e-9

    def test_missing_row(self):
        gt = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
                ["Arts", "Oxford", "2"],
            ]
        )
        pred = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
            ]
        )
        # matched: 1 row × 3 cols = 3 TPs; pred has 1 row, gt has 2 rows.
        # precision = 3/3 = 1.0; recall = 3/6 = 0.5 → f1 = 2/3.
        result = item_f1_from_markdown(pred, gt, ["subject", "university"])
        assert abs(result - 2 / 3) < 1e-9

    def test_extra_row_hurts_precision(self):
        gt = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
            ]
        )
        pred = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
                ["Arts", "Oxford", "2"],  # fabricated row
            ]
        )
        # matched rows: 1; pred has 2 rows; gt has 1.
        # precision = 3/6 = 0.5; recall = 3/3 = 1.0 → f1 = 2/3.
        result = item_f1_from_markdown(pred, gt, ["subject", "university"])
        assert abs(result - 2 / 3) < 1e-9

    def test_url_trailing_slash_counts_as_match(self):
        gt = _md_table(
            [
                ["Subject", "University", "Home Page"],
                ["Arts", "Harvard", "https://www.harvard.edu/"],
            ]
        )
        pred = _md_table(
            [
                ["Subject", "University", "Home Page"],
                ["Arts", "Harvard", "https://www.harvard.edu"],
            ]
        )
        # All 3 cells now match thanks to URL canonicalization.
        assert item_f1_from_markdown(pred, gt, ["subject", "university"]) == 1.0

    def test_multivalue_cell_order_invariant(self):
        gt = _md_table(
            [
                ["Date", "Event", "Top 3"],
                ["2018-02", "Halfpipe", "Hansen<br>Irving<br>Gu"],
            ]
        )
        pred = _md_table(
            [
                ["Date", "Event", "Top 3"],
                ["2018-02", "Halfpipe", "Gu<br>Irving<br>Hansen"],
            ]
        )
        assert item_f1_from_markdown(pred, gt, ["date", "event"]) == 1.0

    def test_required_columns_restricts_denominator(self):
        gt = _md_table(
            [
                ["Subject", "University", "Rank", "Notes"],
                ["Arts", "Harvard", "1", "foo"],
            ]
        )
        pred = _md_table(
            [
                ["Subject", "University", "Rank", "Notes"],
                ["Arts", "Harvard", "1", "bar"],  # notes differ
            ]
        )
        # Without required_columns, denominator is 4 cols, 1 wrong → f1 = 3/4 × 2 / (3/4 + 3/4) = 0.75
        assert item_f1_from_markdown(pred, gt, ["subject", "university"]) == 0.75
        # With required_columns excluding `notes`, all 3 cells match → f1 = 1.0
        assert (
            item_f1_from_markdown(
                pred, gt, ["subject", "university"], required_columns=["subject", "university", "rank"]
            )
            == 1.0
        )

    def test_no_table_returns_zero(self):
        pred = "no markdown table here"
        gt = _md_table([["A", "B"], ["x", "y"]])
        assert item_f1_from_markdown(pred, gt, ["a"]) == 0.0

    def test_missing_unique_columns_returns_zero(self):
        gt = _md_table([["Subject", "University"], ["Arts", "Harvard"]])
        pred = _md_table([["Subject", "University"], ["Arts", "Harvard"]])
        assert item_f1_from_markdown(pred, gt, []) == 0.0

    def test_pred_missing_required_column_drops_score(self):
        gt = _md_table(
            [
                ["Subject", "University", "Rank"],
                ["Arts", "Harvard", "1"],
            ]
        )
        # pred is missing the Rank column entirely.
        pred = _md_table(
            [
                ["Subject", "University"],
                ["Arts", "Harvard"],
            ]
        )
        # GT table still picked via unique columns; pred table also picked
        # (subset covers unique cols). Missing Rank cell → `""` which won't
        # match "1". 2 TPs (subject+university) out of 3 req cols for matched
        # row. precision = 2/3, recall = 2/3 → f1 = 2/3.
        result = item_f1_from_markdown(pred, gt, ["subject", "university"])
        assert abs(result - 2 / 3) < 1e-9


class TestEmScore:
    @pytest.mark.parametrize(
        "response,gt,expected",
        [
            ("The answer is \\boxed{Paris}.", "Paris", 1.0),
            ("\\boxed{Paris}", "paris", 1.0),
            ("<answer>Paris</answer>", "Paris", 1.0),
            ("\\boxed{A Soldier's Oath}", "A Soldier'S Oath", 1.0),  # case corruption
            ("\\boxed{yes}", "yes", 1.0),
            ("\\boxed{Paris}", "London", 0.0),
            ("\\boxed{Paris, France}", "Paris", 0.0),  # must be strict
            # alias list
            ("\\boxed{Paris}", ["Paris", "Paris, France"], 1.0),
            # empty prediction
            ("", "anything", 0.0),
        ],
    )
    def test_em_score(self, response, gt, expected):
        assert em_score(response, gt) == expected
