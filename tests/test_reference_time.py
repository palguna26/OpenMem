"""Reference-time consistency for production retrieval (Task 2B).

- Ordinary evaluates ``[valid_from, valid_until)`` at one resolved reference.
- Lifecycle (deleted/invalidated) always excluded; historical permits expired.
- Explicit invalid reference raises; absent captures one now(UTC).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src import OpenMem
from src.models import TemporalQuery as ModelsTemporalQuery
from src.models import temporal_recency_score, temporal_valid_at_score
from src.retrieval.embedding import pack_embedding
from src.retrieval.retrieval import TemporalQuery as RetrievalTemporalQuery
from src.retrieval.retrieval import parse_temporal_query, temporal_atom_boost


class ConstantEmbedding:
    name = "test-ref-v1"
    dimensions = 2

    def embed(self, value: str) -> list[float]:
        return [1.0, 0.0]

    def embed_many(self, values: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in values]


def make_db(tmp_path: Path, name: str = "t.sqlite") -> OpenMem:
    return OpenMem(tmp_path / name, embedding_provider=ConstantEmbedding())


def insert_memory(
    db: OpenMem,
    namespace: str,
    key: str,
    statement: str,
    valid_from: str,
    valid_until: str | None,
) -> tuple[str, str]:
    """Insert an active memory with explicit real-world validity.

    ``valid_from``/``valid_until`` are ISO strings; ``recorded_at`` uses
    ``valid_from`` for determinism (DB-history field, not event date).
    """
    memory_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ref:{namespace}:{key}"))
    version_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"refver:{namespace}:{key}"))
    db.repository.ensure_namespace(namespace)
    with db.database.connection:
        db.database.execute(
            "INSERT OR IGNORE INTO memories(id, namespace_id, kind, subject_key, status, confidence, current_version_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (memory_id, namespace, "fact", f"subject-{key}", "active", 0.9, version_id, valid_from),
        )
        db.database.execute(
            "UPDATE memories SET current_version_id=?, status='active' WHERE id=? AND namespace_id=?",
            (version_id, memory_id, namespace),
        )
        db.database.execute(
            """INSERT OR REPLACE INTO memory_versions
            (id, memory_id, namespace_id, version, statement, valid_from, valid_to, valid_until,
             recorded_at, status, reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                version_id,
                memory_id,
                namespace,
                1,
                statement,
                valid_from,
                None,
                valid_until,
                valid_from,
                "active",
                "test",
            ),
        )
        db.database.execute(
            "INSERT OR REPLACE INTO memory_fts(memory_version_id, namespace_id, statement, evidence_text) VALUES (?,?,?,?)",
            (version_id, namespace, statement, statement),
        )
        db.database.execute(
            "INSERT OR REPLACE INTO memory_embeddings(memory_version_id, namespace_id, provider, dimensions, vector) VALUES (?,?,?,?,?)",
            (version_id, namespace, ConstantEmbedding.name, ConstantEmbedding.dimensions, pack_embedding([1.0, 0.0])),
        )
    return memory_id, version_id


def test_shared_temporal_query_unified():
    assert ModelsTemporalQuery is RetrievalTemporalQuery
    q = ModelsTemporalQuery(reference_date=datetime(2023, 6, 1, tzinfo=UTC), intent="since")
    assert q.intent == "since"


def test_shared_scoring_supports_since():
    ref = datetime(2023, 6, 1, tzinfo=UTC)
    since_q = ModelsTemporalQuery(reference_date=ref, intent="since", target_date=datetime(2023, 3, 1, tzinfo=UTC))
    after_q = ModelsTemporalQuery(reference_date=ref, intent="after", target_date=datetime(2023, 4, 1, tzinfo=UTC))
    assert temporal_valid_at_score(datetime(2023, 3, 15, tzinfo=UTC), None, None, since_q) > 0
    assert temporal_valid_at_score(datetime(2023, 3, 15, tzinfo=UTC), None, None, after_q) < 0
    assert temporal_valid_at_score(datetime(2023, 4, 1, tzinfo=UTC), None, None, after_q) > 0


def test_valid_at_reference_but_expired_today_eligible(tmp_path: Path):
    db = make_db(tmp_path)
    try:
        insert_memory(db, "n1", "k1", "Reference eligible alpha beacon", "2023-01-01T00:00:00+00:00", "2023-06-01T00:00:00+00:00")
        # Valid at March reference, expired today (2026) -> eligible with ref.
        assert db.search("n1", "alpha beacon", reference_date="2023-03-01T00:00:00+00:00")
        # No reference (captured now, expired) -> excluded.
        assert db.search("n1", "alpha beacon") == []
        # Historical permits expired.
        assert db.repository.search("n1", "alpha beacon", 10, historical=True)
    finally:
        db.close()


def test_fallback_vector_uses_resolved_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = make_db(tmp_path)
    try:
        db.repository.vector_index.available = False
        monkeypatch.setattr(db.repository, "_chunk_hits", lambda *a, **k: {})
        _, version_id = insert_memory(db, "n1", "k1", "Fallback vector gamma beacon", "2023-01-01T00:00:00+00:00", "2023-06-01T00:00:00+00:00")
        # Non-lexical query isolates dense fallback (lexical cannot match).
        assert db.search("n1", "persistence layer choice", reference_date="2023-03-01T00:00:00+00:00")
        assert db.search("n1", "persistence layer choice") == []
        assert version_id
    finally:
        db.close()


def test_indexed_vector_uses_resolved_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = make_db(tmp_path)
    try:
        _, version_id = insert_memory(db, "n1", "k1", "Indexed vector delta beacon", "2023-01-01T00:00:00+00:00", "2023-06-01T00:00:00+00:00")
        monkeypatch.setattr(db.repository, "_chunk_hits", lambda *a, **k: {})
        monkeypatch.setattr(db.repository.vector_index, "search", lambda ns, qv, lim: [(version_id, 0.99)])
        assert db.search("n1", "persistence layer choice", reference_date="2023-03-01T00:00:00+00:00")
        assert db.search("n1", "persistence layer choice") == []
    finally:
        db.close()


def test_starting_after_reference_excluded(tmp_path: Path):
    db = make_db(tmp_path)
    try:
        insert_memory(db, "n1", "k1", "Future start epsilon beacon", "2023-06-01T00:00:00+00:00", None)
        assert db.search("n1", "epsilon beacon", reference_date="2023-03-01T00:00:00+00:00") == []
        assert db.search("n1", "epsilon beacon", reference_date="2023-07-01T00:00:00+00:00")
    finally:
        db.close()


def test_start_inclusive_expiry_exclusive(tmp_path: Path):
    db = make_db(tmp_path)
    try:
        insert_memory(db, "n1", "k1", "Boundary zeta beacon", "2023-03-01T00:00:00+00:00", "2023-04-01T00:00:00+00:00")
        assert db.search("n1", "zeta beacon", reference_date="2023-03-01T00:00:00+00:00")
        assert db.search("n1", "zeta beacon", reference_date="2023-04-01T00:00:00+00:00") == []
    finally:
        db.close()


def test_timezone_offsets_equivalent(tmp_path: Path):
    db = make_db(tmp_path)
    try:
        # Same instant (00:00Z), different offsets.
        insert_memory(db, "n1", "ka", "Timezone theta beacon", "2023-01-01T00:00:00+00:00", None)
        insert_memory(db, "n1", "kb", "Timezone iota beacon", "2023-01-01T02:00:00+02:00", None)
        at_start = db.search("n1", "beacon", reference_date="2023-01-01T00:00:00+00:00", limit=10)
        assert {r.statement for r in at_start} >= {"Timezone theta beacon", "Timezone iota beacon"}
        before = db.search("n1", "beacon", reference_date="2022-12-31T23:59:00+00:00", limit=10)
        assert before == []
    finally:
        db.close()


def test_explicit_reference_stable_across_clock_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = make_db(tmp_path)
    try:
        # valid_from 10 days before ref -> recency in (0, 0.02]; detects clock reads.
        insert_memory(db, "n1", "k1", "Stable kappa beacon", "2023-05-22T00:00:00+00:00", "2023-12-31T00:00:00+00:00")
        ref = "2023-06-01T00:00:00+00:00"
        first = db.search("n1", "kappa beacon", reference_date=ref)
        assert first
        assert first[0].component_scores["recency"] > 0
        scores_first = [(str(r.memory_version_id), r.score) for r in first]
        # Patch the actual clock source used by resolve_reference_time().
        monkeypatch.setattr("src.retrieval.retrieval._utc_now", lambda: datetime(2099, 1, 1, tzinfo=UTC))
        monkeypatch.setattr("src.models.utc_now", lambda: datetime(2099, 1, 1, tzinfo=UTC))
        second = db.search("n1", "kappa beacon", reference_date=ref)
        assert [(str(r.memory_version_id), r.score) for r in second] == scores_first
    finally:
        db.close()


def test_absent_reference_single_clock_read_controls_eligibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Without a reference, exactly one clock read decides validity at boundaries."""
    import src.retrieval.retrieval as retrieval_mod

    db = make_db(tmp_path)
    try:
        # Validity ends exactly at June 1 (exclusive); recency fixture 10 days old.
        insert_memory(db, "n1", "k1", "Clock sigma beacon", "2023-01-01T00:00:00+00:00", "2023-06-01T00:00:00+00:00")
        calls: list[datetime] = []
        just_before = datetime(2023, 5, 31, 23, 59, tzinfo=UTC)

        def fake_now_before() -> datetime:
            calls.append(just_before)
            return just_before

        monkeypatch.setattr(retrieval_mod, "_utc_now", fake_now_before)
        assert db.search("n1", "sigma beacon")  # 1 min before expiry -> eligible
        assert len(calls) == 1

        calls.clear()
        at_expiry = datetime(2023, 6, 1, 0, 0, tzinfo=UTC)

        def fake_now_at() -> datetime:
            calls.append(at_expiry)
            return at_expiry

        monkeypatch.setattr(retrieval_mod, "_utc_now", fake_now_at)
        assert db.search("n1", "sigma beacon") == []  # at expiry -> excluded
        assert len(calls) == 1
    finally:
        db.close()


def test_invalid_reference_raises_absent_uses_default(tmp_path: Path):
    db = make_db(tmp_path)
    try:
        insert_memory(db, "n1", "k1", "Default lambda beacon", "2020-01-01T00:00:00+00:00", None)
        with pytest.raises(ValueError):
            db.search("n1", "lambda beacon", reference_date="not-a-date")
        with pytest.raises(ValueError):
            db.search("n1", "lambda beacon", reference_date="2023-02-30")
        # Absent (None/blank) captures now -> currently-valid memory eligible.
        assert db.search("n1", "lambda beacon", reference_date=None)
        assert db.search("n1", "lambda beacon", reference_date="")
    finally:
        db.close()


def test_context_apis_forward_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = make_db(tmp_path)
    try:
        insert_memory(db, "n1", "k1", "Context mu beacon", "2023-01-01T00:00:00+00:00", "2023-06-01T00:00:00+00:00")
        ref = "2023-03-01T00:00:00+00:00"
        ctx = db.search_context("n1", "mu beacon", limit=5, reference_date=ref)
        assert any("mu beacon" in m.statement for m in ctx["memories"])
        ctx_now = db.search_context("n1", "mu beacon", limit=5)
        assert ctx_now["memories"] == []
        # build_answer_context must forward the reference to search (spy, since
        # direct-SQL fixtures have no chunks for packing).
        seen: dict[str, object] = {}
        orig_search = db.search

        def spy_search(namespace_id: str, query: str, limit: int = 10, historical: bool = False, **kwargs):  # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return orig_search(namespace_id, query, limit, historical, **kwargs)

        monkeypatch.setattr(db, "search", spy_search)
        db.build_answer_context("n1", "mu beacon", limit=5, reference_date=ref)
        assert seen.get("reference_date") == ref
        db.build_answer_context("n1", "mu beacon", limit=5)
        assert seen.get("reference_date") is None
    finally:
        db.close()


def test_equivalent_representations_identical_scores():
    """Same instant in UTC / offset / naive forms scores identically."""
    ref = datetime(2023, 6, 1, tzinfo=UTC)
    utc = datetime(2023, 3, 1, 0, 0, tzinfo=UTC)
    offset = datetime.fromisoformat("2023-03-01T02:00:00+02:00")
    naive = datetime(2023, 3, 1, 0, 0)
    assert offset == utc  # same instant sanity check
    # Shared scoring: before / after / since with equivalent anchors.
    before_q = ModelsTemporalQuery(reference_date=ref, intent="before", target_date=utc)
    assert temporal_valid_at_score(utc, None, None, before_q) == temporal_valid_at_score(offset, None, None, before_q)
    assert temporal_valid_at_score(naive, None, None, before_q) == temporal_valid_at_score(utc, None, None, before_q)
    after_q = ModelsTemporalQuery(reference_date=ref, intent="after", target_date=utc)
    since_q = ModelsTemporalQuery(reference_date=ref, intent="since", target_date=utc)
    for q in (after_q, since_q):
        assert temporal_valid_at_score(utc, None, None, q) == temporal_valid_at_score(offset, None, None, q)
        assert temporal_valid_at_score(naive, None, None, q) == temporal_valid_at_score(utc, None, None, q)
    around_q = ModelsTemporalQuery(
        reference_date=ref,
        intent="around",
        date_range_start=datetime(2023, 3, 1, tzinfo=UTC),
        date_range_end=datetime(2023, 4, 1, tzinfo=UTC),
    )
    assert temporal_valid_at_score(utc, None, None, around_q) == temporal_valid_at_score(offset, None, None, around_q)
    # Atom scoring agrees across representations.
    atom_before = parse_temporal_query("What happened before March 2023?", "2023/06/01 (Thu) 00:00")
    assert temporal_atom_boost("2023-03-01T00:00:00+00:00", atom_before) == temporal_atom_boost("2023-03-01T02:00:00+02:00", atom_before)
    # Recency: naive-as-UTC matches aware UTC.
    assert temporal_recency_score(naive, now=ref) == temporal_recency_score(utc, now=ref)
    assert temporal_recency_score(offset, now=ref) == temporal_recency_score(utc, now=ref)


def test_february_march_boundary_with_offset():
    """Reproduced stripping bug: 00:00+02:00 is Feb 28 22:00Z, before March."""
    before_mar = parse_temporal_query("What happened before March 2023?", "2023/06/01 (Thu) 00:00")
    # Same instant, different representations must agree (before -> +0.05).
    assert temporal_atom_boost("2023-02-28T22:00:00+00:00", before_mar) > 0
    assert temporal_atom_boost("2023-03-01T00:00:00+02:00", before_mar) == temporal_atom_boost("2023-02-28T22:00:00+00:00", before_mar)
    # Shared model agrees.
    ref = datetime(2023, 6, 1, tzinfo=UTC)
    q = ModelsTemporalQuery(reference_date=ref, intent="before", target_date=datetime(2023, 3, 1, tzinfo=UTC))
    assert temporal_valid_at_score(datetime.fromisoformat("2023-03-01T00:00:00+02:00"), None, None, q) > 0
    assert temporal_valid_at_score(datetime.fromisoformat("2023-03-01T00:00:00+02:00"), None, None, q) == temporal_valid_at_score(
        datetime.fromisoformat("2023-02-28T22:00:00+00:00"), None, None, q
    )


def test_production_temporal_boost_identical_across_offsets(tmp_path: Path):
    """Production component_scores['temporal_boost'] matches for same instant."""
    db = make_db(tmp_path)
    try:
        insert_memory(db, "n1", "ka", "Offset theta beacon", "2023-02-28T22:00:00+00:00", None)
        insert_memory(db, "n1", "kb", "Offset iota beacon", "2023-03-01T00:00:00+02:00", None)
        results = db.search("n1", "What happened before March 2023?", limit=10)
        boosts = {r.statement: r.component_scores["temporal_boost"] for r in results}
        assert boosts["Offset theta beacon"] == boosts["Offset iota beacon"]
    finally:
        db.close()
