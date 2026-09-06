"""Temporal parsing + earliest-ranking regression (Task 2A).

Date semantics (inclusive start, exclusive end):
- Before March 2023: < 2023-03-01
- After March 2023: >= 2023-04-01
- During March 2023: [2023-03-01, 2023-04-01)
- Since March 2023: >= 2023-03-01
- Equivalent for years / explicit days.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.retrieval.retrieval import (
    AtomHit,
    parse_reference_date,
    parse_temporal_query,
    search_atoms_with_stages,
    temporal_atom_boost,
)
from src.storage.db import Database

REF = "2023/06/01 (Thu) 00:00"


def atom_ts(year: int, month: int, day: int, hour: int = 10, minute: int = 0) -> str:
    # Haystack format parsed by parse_reference_date.
    dt = datetime(year, month, day, hour, minute, tzinfo=UTC)
    return dt.strftime("%Y/%m/%d (%a) %H:%M")


# ---------------------------------------------------------------------------
# Interval semantics
# ---------------------------------------------------------------------------


def test_before_march_uses_start_not_midpoint():
    tq = parse_temporal_query("What happened before March 2023?", REF)
    assert tq.intent == "before"
    assert tq.date_range_start == datetime(2023, 3, 1, tzinfo=UTC)
    assert tq.date_range_end == datetime(2023, 4, 1, tzinfo=UTC)
    assert tq.target_date == datetime(2023, 3, 1, tzinfo=UTC)
    # March 10 must NOT be boosted (old midpoint March 16 incorrectly boosted it).
    assert temporal_atom_boost(atom_ts(2023, 3, 10), tq) < 0
    assert temporal_atom_boost(atom_ts(2023, 2, 28), tq) > 0
    # Boundary: exactly March 1 is not "before".
    assert temporal_atom_boost("2023-03-01T00:00:00+00:00", tq) < 0


def test_after_march_uses_end():
    tq = parse_temporal_query("What happened after March 2023?", REF)
    assert tq.intent == "after"
    assert tq.target_date == datetime(2023, 4, 1, tzinfo=UTC)
    assert temporal_atom_boost(atom_ts(2023, 4, 1), tq) > 0
    assert temporal_atom_boost(atom_ts(2023, 3, 31), tq) < 0
    # March 15 is inside March, not after it.
    assert temporal_atom_boost(atom_ts(2023, 3, 15), tq) < 0


def test_during_march_interval():
    tq = parse_temporal_query("What happened during March 2023?", REF)
    assert tq.intent == "around"
    assert tq.date_range_start == datetime(2023, 3, 1, tzinfo=UTC)
    assert tq.date_range_end == datetime(2023, 4, 1, tzinfo=UTC)
    assert temporal_atom_boost(atom_ts(2023, 3, 1), tq) > 0
    assert temporal_atom_boost(atom_ts(2023, 3, 31), tq) > 0
    # In-range outranks the exclusive-end boundary (near-miss decay may still
    # give partial credit, but must be strictly lower than in-range).
    assert temporal_atom_boost(atom_ts(2023, 3, 15), tq) > temporal_atom_boost(atom_ts(2023, 4, 1), tq)


def test_since_march_uses_start_and_differs_from_after():
    since_q = parse_temporal_query("What happened since March 2023?", REF)
    after_q = parse_temporal_query("What happened after March 2023?", REF)
    assert since_q.intent == "since"
    assert after_q.intent == "after"
    assert since_q.intent != after_q.intent
    assert since_q.target_date == datetime(2023, 3, 1, tzinfo=UTC)
    assert after_q.target_date == datetime(2023, 4, 1, tzinfo=UTC)
    # March 15 counts for since, not for after.
    assert temporal_atom_boost(atom_ts(2023, 3, 15), since_q) > 0
    assert temporal_atom_boost(atom_ts(2023, 3, 15), after_q) < 0
    assert temporal_atom_boost(atom_ts(2023, 3, 1), since_q) > 0
    assert temporal_atom_boost(atom_ts(2023, 2, 28), since_q) < 0


def test_december_year_transition():
    before_dec = parse_temporal_query("What happened before December 2023?", REF)
    assert before_dec.date_range_start == datetime(2023, 12, 1, tzinfo=UTC)
    assert before_dec.date_range_end == datetime(2024, 1, 1, tzinfo=UTC)
    assert before_dec.target_date == datetime(2023, 12, 1, tzinfo=UTC)
    assert temporal_atom_boost(atom_ts(2023, 11, 30), before_dec) > 0
    assert temporal_atom_boost(atom_ts(2023, 12, 15), before_dec) < 0

    after_dec = parse_temporal_query("What happened after December 2023?", REF)
    assert after_dec.target_date == datetime(2024, 1, 1, tzinfo=UTC)
    assert temporal_atom_boost(atom_ts(2024, 1, 1), after_dec) > 0
    assert temporal_atom_boost(atom_ts(2023, 12, 31), after_dec) < 0

    during_dec = parse_temporal_query("What happened during December 2023?", REF)
    assert temporal_atom_boost(atom_ts(2023, 12, 31), during_dec) > 0
    assert temporal_atom_boost(atom_ts(2023, 12, 15), during_dec) > temporal_atom_boost(atom_ts(2024, 1, 1), during_dec)


def test_year_boundaries():
    before_y = parse_temporal_query("What happened before 2023?", REF)
    assert before_y.intent == "before"
    assert before_y.date_range_start == datetime(2023, 1, 1, tzinfo=UTC)
    assert before_y.date_range_end == datetime(2024, 1, 1, tzinfo=UTC)
    assert before_y.target_date == datetime(2023, 1, 1, tzinfo=UTC)
    assert temporal_atom_boost(atom_ts(2022, 12, 31), before_y) > 0
    assert temporal_atom_boost(atom_ts(2023, 6, 15), before_y) < 0

    after_y = parse_temporal_query("What happened after 2023?", REF)
    assert after_y.target_date == datetime(2024, 1, 1, tzinfo=UTC)
    assert temporal_atom_boost(atom_ts(2024, 1, 1), after_y) > 0
    assert temporal_atom_boost(atom_ts(2023, 12, 31), after_y) < 0

    since_y = parse_temporal_query("What happened since 2023?", REF)
    assert since_y.intent == "since"
    assert since_y.target_date == datetime(2023, 1, 1, tzinfo=UTC)


def test_abbreviated_month_year_matches_full_names():
    cases = [
        ("Mar 2023", datetime(2023, 3, 1, tzinfo=UTC), datetime(2023, 4, 1, tzinfo=UTC)),
        ("Sep 2023", datetime(2023, 9, 1, tzinfo=UTC), datetime(2023, 10, 1, tzinfo=UTC)),
        ("Sept 2023", datetime(2023, 9, 1, tzinfo=UTC), datetime(2023, 10, 1, tzinfo=UTC)),
    ]
    for label, start, end in cases:
        tq = parse_temporal_query(f"What happened during {label}?", REF)
        assert tq.date_range_start == start, label
        assert tq.date_range_end == end, label

    # Full-name equivalence.
    assert (
        parse_temporal_query("What happened during Mar 2023?", REF).date_range_start
        == parse_temporal_query("What happened during March 2023?", REF).date_range_start
    )
    assert (
        parse_temporal_query("What happened during Sep 2023?", REF).date_range_start
        == parse_temporal_query("What happened during September 2023?", REF).date_range_start
    )
    assert (
        parse_temporal_query("What happened during Sept 2023?", REF).date_range_start
        == parse_temporal_query("What happened during September 2023?", REF).date_range_start
    )


def test_abbreviated_month_year_boundaries():
    before_mar = parse_temporal_query("What happened before Mar 2023?", REF)
    assert before_mar.intent == "before"
    assert before_mar.date_range_start == datetime(2023, 3, 1, tzinfo=UTC)
    assert before_mar.target_date == datetime(2023, 3, 1, tzinfo=UTC)
    assert temporal_atom_boost(atom_ts(2023, 2, 28), before_mar) > 0
    assert temporal_atom_boost(atom_ts(2023, 3, 10), before_mar) < 0

    after_sep = parse_temporal_query("What happened after Sep 2023?", REF)
    assert after_sep.intent == "after"
    assert after_sep.target_date == datetime(2023, 10, 1, tzinfo=UTC)
    assert temporal_atom_boost(atom_ts(2023, 10, 1), after_sep) > 0
    assert temporal_atom_boost(atom_ts(2023, 9, 30), after_sep) < 0

    since_sept = parse_temporal_query("What happened since Sept 2023?", REF)
    assert since_sept.intent == "since"
    assert since_sept.target_date == datetime(2023, 9, 1, tzinfo=UTC)
    # Same-month contrast: Sep 15 counts for since-Sept but not after-Sept.
    after_sept_same = parse_temporal_query("What happened after Sept 2023?", REF)
    assert after_sept_same.intent == "after"
    assert after_sept_same.target_date == datetime(2023, 10, 1, tzinfo=UTC)
    assert temporal_atom_boost(atom_ts(2023, 9, 15), since_sept) > 0
    assert temporal_atom_boost(atom_ts(2023, 9, 15), after_sept_same) < 0


def test_explicit_day_iso_and_english():
    for query in ("What happened before March 10, 2023?", "What happened before 2023-03-10?"):
        tq = parse_temporal_query(query, REF)
        assert tq.intent == "before", query
        assert tq.date_range_start == datetime(2023, 3, 10, tzinfo=UTC), query
        assert tq.date_range_end == datetime(2023, 3, 11, tzinfo=UTC), query
        assert tq.target_date == datetime(2023, 3, 10, tzinfo=UTC), query
        assert temporal_atom_boost("2023-03-09T23:59:00+00:00", tq) > 0
        assert temporal_atom_boost("2023-03-10T00:00:00+00:00", tq) < 0

    after_iso = parse_temporal_query("What happened after 2023-03-10?", REF)
    assert after_iso.target_date == datetime(2023, 3, 11, tzinfo=UTC)
    assert temporal_atom_boost("2023-03-11T00:00:00+00:00", after_iso) > 0
    assert temporal_atom_boost("2023-03-10T12:00:00+00:00", after_iso) < 0

    since_day = parse_temporal_query("What happened since March 10, 2023?", REF)
    assert since_day.intent == "since"
    assert since_day.target_date == datetime(2023, 3, 10, tzinfo=UTC)
    assert temporal_atom_boost("2023-03-10T00:00:00+00:00", since_day) > 0

    during_day = parse_temporal_query("What happened on 2023-03-10?", REF)
    assert during_day.date_range_start == datetime(2023, 3, 10, tzinfo=UTC)
    assert during_day.date_range_end == datetime(2023, 3, 11, tzinfo=UTC)


def test_leap_day_valid_and_invalid():
    valid = parse_temporal_query("What happened on February 29, 2024?", REF)
    assert valid.date_range_start == datetime(2024, 2, 29, tzinfo=UTC)
    assert valid.date_range_end == datetime(2024, 3, 1, tzinfo=UTC)
    assert temporal_atom_boost("2024-02-29T12:00:00+00:00", valid) > 0

    invalid = parse_temporal_query("What happened on February 29, 2023?", REF)
    assert invalid.date_range_start is None
    assert invalid.date_range_end is None
    assert invalid.target_date is None

    invalid_iso = parse_temporal_query("What happened before 2023-02-30?", REF)
    assert invalid_iso.date_range_start is None
    assert invalid_iso.target_date is None


def test_missing_and_invalid_dates_explicit():
    no_date = parse_temporal_query("What happened before?", REF)
    assert no_date.intent == "before"
    assert no_date.target_date is None
    assert no_date.date_range_start is None
    # Explicit: no boost, no invented midpoint, no crash.
    assert temporal_atom_boost(atom_ts(2023, 3, 10), no_date) == 0.0
    assert temporal_atom_boost(None, no_date) == 0.0

    assert parse_reference_date("") is None
    assert parse_reference_date(None) is None
    assert parse_reference_date("not-a-date") is None
    assert temporal_atom_boost(None, parse_temporal_query("What is SQLite?", REF)) == 0.0


def test_first_name_not_earliest():
    assert parse_temporal_query("What is my first name?", REF).intent != "earliest"
    assert parse_temporal_query("My first name is Alice.", REF).intent != "earliest"
    # Genuine earliest requests still classify.
    assert parse_temporal_query("What was my first job?", REF).intent == "earliest"
    assert parse_temporal_query("What was the earliest event?", REF).intent == "earliest"


# ---------------------------------------------------------------------------
# Earliest ranking: relevance primary
# ---------------------------------------------------------------------------


def _seed_atoms(db: Database) -> None:
    db.execute("INSERT OR IGNORE INTO namespaces(id, org_id, created_at) VALUES ('n1','b',datetime('now'))")
    db.connection.commit()
    rows = [
        ("a_old", "2020/01/01 (Wed) 00:00", "old irrelevant fact xyz"),
        ("a_new", "2023/01/01 (Sun) 00:00", "new relevant fact xyz"),
    ]
    for atom_id, ts, fact in rows:
        db.execute(
            "INSERT INTO atoms(atom_id, session_id, fact, timestamp, source_role, created_at, namespace_id) VALUES (?,?,?,?,?,?,?)",
            (atom_id, "s1", fact, ts, "user", ts, "n1"),
        )
        db.execute(
            "INSERT INTO atoms_fts(rowid, fact, timestamp, atom_id) VALUES ((SELECT rowid FROM atoms WHERE atom_id=?),?,?,?)",
            (atom_id, fact, ts, atom_id),
        )
    db.connection.commit()


def test_earliest_preserves_relevance_over_timestamp(tmp_path: Path):
    """Old irrelevant candidate must not outrank newer relevant one."""
    db = Database(tmp_path / "earliest.sqlite")
    try:
        _seed_atoms(db)

        def vector_search(query: str, n: int) -> list[AtomHit]:
            # Newer hit is more relevant (ranked first).
            return [
                AtomHit("a_new", "s1", "new relevant fact xyz", "2023/01/01 (Sun) 00:00", "user", 0.9),
                AtomHit("a_old", "s1", "old irrelevant fact xyz", "2020/01/01 (Wed) 00:00", "user", 0.1),
            ]

        hits, _ = search_atoms_with_stages(db, "earliest event", 10, vector_search=vector_search, namespace_id="n1", reference_date=REF)
        assert [h.atom_id for h in hits] == ["a_new", "a_old"]
    finally:
        db.close()


def test_earliest_equal_scores_sort_oldest_first_unknown_last(tmp_path: Path, monkeypatch) -> None:
    """Equal merged scores through search: oldest known, newer known, unknown.

    Stubs ``rrf_merge`` to supply controlled equal-score candidates in
    deliberately wrong input order. Without the timestamp tie-break
    (score-and-ID only), alphabetical order would be
    ``[a_new, a_none, a_old]``; the correct earliest order is
    ``[a_old, a_new, a_none]``.
    """
    import src.retrieval.retrieval as retrieval_mod

    db = Database(tmp_path / "earliest-tie.sqlite")
    try:
        db.execute("INSERT OR IGNORE INTO namespaces(id, org_id, created_at) VALUES ('n1','b',datetime('now'))")
        db.connection.commit()
        # Wrong input order on purpose: unknown, newer, older.
        stubbed = [
            AtomHit("a_none", "s1", "tie fact none", None, "user", 1.0),
            AtomHit("a_new", "s1", "tie fact a_new", "2023/01/01 (Sun) 00:00", "user", 1.0),
            AtomHit("a_old", "s1", "tie fact a_old", "2020/01/01 (Wed) 00:00", "user", 1.0),
        ]
        monkeypatch.setattr(retrieval_mod, "rrf_merge", lambda lists, k=60: list(stubbed))
        hits, _ = search_atoms_with_stages(db, "What was the earliest event?", 10, vector_search=lambda q, n: [], namespace_id="n1", reference_date=REF)
        assert [h.atom_id for h in hits] == ["a_old", "a_new", "a_none"]
    finally:
        db.close()
