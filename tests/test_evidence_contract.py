"""Evidence-contract consistency: direct vs queued processing (Task 3A).

Both paths must reject evidence-free, malformed, and out-of-scope candidates
with explicit diagnostics; transport failures stay retryable.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src import OpenMem
from src.memory.provider import ProviderError, ProviderResult
from src.models import EvidenceSpan, ExtractionCandidate, ExtractionResponse


class RecordingEmbedding:
    name = "evidence-v1"
    dimensions = 2

    def embed(self, value):
        return [1.0, 0.0]

    def embed_many(self, values):
        return [[1.0, 0.0] for _ in values]


def _result(candidates, name="t", model="t-v1"):
    return ProviderResult(
        response=ExtractionResponse(schema_version="extraction-v1", prompt_version="simple-v1", candidates=candidates),
        provider_name=name,
        model_name=model,
        prompt_version="simple-v1",
        raw_response_hash="test",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        stage="facts",
    )


def _valid_candidate(request, statement=None):
    assert request.events, "test setup: request has no extractable events"
    eid = request.events[0]
    assert eid in request.evidence_text, "test setup: extractable event missing from evidence_text"
    text = request.evidence_text[eid]
    excerpt = text[:60]
    return ExtractionCandidate(
        kind="fact",
        subject="user preference",
        statement=statement or text[:120],
        evidence=[EvidenceSpan(event_id=eid, start_offset=0, end_offset=len(excerpt), excerpt=excerpt)],
        confidence=0.9,
        importance=0.5,
        durability="permanent",
    )


class MissingProvider:
    name = "missing"
    model = "missing-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        return _result(
            [
                ExtractionCandidate(
                    kind="fact",
                    subject="user preference",
                    statement="User prefers SQLite for local storage.",
                    evidence=[],
                    confidence=0.9,
                    importance=0.5,
                    durability="permanent",
                )
            ],
            self.name,
            self.model,
        )


class BadExcerptProvider:
    name = "bad-excerpt"
    model = "bad-excerpt-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        assert request.events, "test setup: no extractable events"
        eid = request.events[0]
        text = request.evidence_text[eid]
        assert text[:5] != "WRONG", "test setup: fixture collides with corruption marker"
        return _result(
            [
                ExtractionCandidate(
                    kind="fact",
                    subject="user preference",
                    statement=text[:120],
                    evidence=[EvidenceSpan(event_id=eid, start_offset=0, end_offset=5, excerpt="WRONG")],
                    confidence=0.9,
                    importance=0.5,
                    durability="permanent",
                )
            ],
            self.name,
            self.model,
        )


class BadOffsetProvider:
    name = "bad-offset"
    model = "bad-offset-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        assert request.events, "test setup: no extractable events"
        eid = request.events[0]
        return _result(
            [
                ExtractionCandidate(
                    kind="fact",
                    subject="user preference",
                    statement=request.evidence_text[eid][:120],
                    evidence=[EvidenceSpan(event_id=eid, start_offset=9999, end_offset=10005, excerpt="xxxxx")],
                    confidence=0.9,
                    importance=0.5,
                    durability="permanent",
                )
            ],
            self.name,
            self.model,
        )


class UnknownSourceProvider:
    name = "unknown-source"
    model = "unknown-source-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        ghost = uuid4()
        assert ghost not in request.evidence_text, "test setup: ghost id collides"
        return _result(
            [
                ExtractionCandidate(
                    kind="fact",
                    subject="user preference",
                    statement="User prefers SQLite for local storage.",
                    evidence=[EvidenceSpan(event_id=ghost, start_offset=0, end_offset=5, excerpt="hello")],
                    confidence=0.9,
                    importance=0.5,
                    durability="permanent",
                )
            ],
            self.name,
            self.model,
        )


class CrossNamespaceProvider:
    """Cite a real n2 event (with its correct excerpt) from an n1 extraction."""

    name = "cross-ns"
    model = "cross-ns-v1"

    def __init__(self, foreign_event_id: UUID, foreign_text: str):
        self.foreign_event_id = foreign_event_id
        self.foreign_text = foreign_text

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        excerpt = self.foreign_text[:5]
        return _result(
            [
                ExtractionCandidate(
                    kind="fact",
                    subject="user preference",
                    statement=self.foreign_text[:120],
                    evidence=[EvidenceSpan(event_id=self.foreign_event_id, start_offset=0, end_offset=len(excerpt), excerpt=excerpt)],
                    confidence=0.9,
                    importance=0.5,
                    durability="permanent",
                )
            ],
            self.name,
            self.model,
        )


class UnknownChunkProvider:
    name = "unknown-chunk"
    model = "unknown-chunk-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        cand = _valid_candidate(request)
        cand = cand.model_copy(update={"source_chunk_ids": ["chunk_nonexistent_999"]})
        return _result([cand], self.name, self.model)


class ContextOnlyProvider:
    """Cite a context (non-extractable) event: in input but outside the batch."""

    name = "context-only"
    model = "context-only-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        extractable = set(request.events) | set(getattr(request, "extractable_event_ids", []) or [])
        context_ids = list(getattr(request, "context_event_ids", []) or [])
        targets = [eid for eid in context_ids if eid in request.evidence_text]
        if not targets:
            targets = [eid for eid in request.evidence_text if eid not in extractable]
        if not targets:
            raise AssertionError("test setup: no context event in request")
        eid = targets[0]
        text = request.evidence_text[eid]
        excerpt = text[:60]
        return _result(
            [
                ExtractionCandidate(
                    kind="fact",
                    subject="user preference",
                    statement=text[:120],
                    evidence=[EvidenceSpan(event_id=eid, start_offset=0, end_offset=len(excerpt), excerpt=excerpt)],
                    confidence=0.9,
                    importance=0.5,
                    durability="permanent",
                )
            ],
            self.name,
            self.model,
        )


class ValidProvider:
    name = "valid"
    model = "valid-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        return _result([_valid_candidate(request)], self.name, self.model)


class FailingProvider:
    name = "failing"
    model = "failing-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        raise ProviderError("provider unavailable", retryable=True, error_class="transport_error")


def _v3_candidate(statement, *, labels):
    assert labels, "test setup: v3 candidate needs source labels"
    return ExtractionCandidate(
        kind="fact",
        subject="user profile",
        statement=statement,
        evidence=[],
        confidence=0.9,
        importance=0.8,
        durability="permanent",
        v3_type="fact",
        v3_lifecycle="stable",
        v3_source_labels=list(labels),
        v3_importance_int=4,
    )


class V3Provider:
    name = "v3-test"
    model = "v3-test-model"

    def __init__(self, factory):
        self.factory = factory

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        return ProviderResult(
            response=ExtractionResponse(
                schema_version="extraction-v1",
                prompt_version="extraction-v3-test",
                candidates=[self.factory(request)],
            ),
            provider_name=self.name,
            model_name=self.model,
            prompt_version="extraction-v3-test",
            raw_response_hash="test",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            stage="facts",
        )


def _event(ns, key, text, stream_id=None):
    ev = {"namespace_id": ns, "idempotency_key": key, "type": "conversation", "payload": {"text": text}}
    if stream_id is not None:
        ev["stream_id"] = stream_id
    return ev


def run_direct(tmp_path: Path, provider, events, name="t.sqlite"):
    db = OpenMem(tmp_path / name, extraction_provider=provider, embedding_provider=RecordingEmbedding())
    result = db.ingest_batch(events)
    return db, result


def run_queued(tmp_path: Path, provider, events, name="t.sqlite"):
    """Exercise process_namespace via a failing direct ingest then provider swap."""
    db = OpenMem(tmp_path / name, extraction_provider=FailingProvider(), embedding_provider=RecordingEmbedding())
    for ev in events:
        with pytest.raises(ProviderError):
            db.ingest(ev)
    assert db.memories(events[0]["namespace_id"]) == []
    db.processor.provider = provider
    resp = db.process(events[0]["namespace_id"])
    return db, resp


def _rejection_reasons(db, namespace):
    rows = db.database.execute(
        "SELECT rejection_reason FROM extraction_decisions WHERE namespace_id=? AND validation_status='rejected'",
        (namespace,),
    ).fetchall()
    return {r[0] for r in rows}


def _citations(db, namespace):
    rows = db.database.execute(
        """SELECT r.event_id AS event_id, r.start_offset AS start_offset, r.end_offset AS end_offset,
                  r.excerpt AS excerpt, e.namespace_id AS ns
           FROM evidence_refs r JOIN events e ON e.id=r.event_id
           WHERE r.namespace_id=? ORDER BY r.rowid""",
        (namespace,),
    ).fetchall()
    return [dict(r) for r in rows]


def _event_id(db, namespace, key):
    # Stable UUID derivation matches repository.stable_uuid.
    from src.storage.repository import stable_uuid

    return stable_uuid(namespace, key)


def test_missing_evidence_rejected_by_both(tmp_path: Path):
    db, result = run_direct(tmp_path, MissingProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "d1.sqlite")
    try:
        assert result.accepted == 0 and result.rejected == 1
        assert db.memories("n1") == []
        assert "missing_source_evidence" in _rejection_reasons(db, "n1")
    finally:
        db.close()
    db2, resp = run_queued(tmp_path, MissingProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "q1.sqlite")
    try:
        assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0
        assert resp.accepted == 0 and resp.rejected == 1
        assert db2.memories("n1") == []
        assert "missing_source_evidence" in _rejection_reasons(db2, "n1")
    finally:
        db2.close()


def test_incorrect_excerpt_or_offsets_rejected_by_both(tmp_path: Path):
    for provider, name in ((BadExcerptProvider(), "excerpt"), (BadOffsetProvider(), "offset")):
        db, result = run_direct(tmp_path, provider, [_event("n1", f"k-{name}", "I prefer SQLite for local storage.")], f"d-{name}.sqlite")
        try:
            assert result.accepted == 0 and result.rejected == 1, name
            assert db.memories("n1") == [], name
            reasons = _rejection_reasons(db, "n1")
            assert reasons & {"evidence_excerpt_mismatch", "invalid_evidence_span"}, (name, reasons)
        finally:
            db.close()
        db2, resp = run_queued(tmp_path, provider.__class__(), [_event("n1", f"k-{name}", "I prefer SQLite for local storage.")], f"q-{name}.sqlite")
        try:
            assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0, name
            assert resp.accepted == 0 and resp.rejected == 1, name
            assert db2.memories("n1") == [], name
            reasons = _rejection_reasons(db2, "n1")
            assert reasons & {"evidence_excerpt_mismatch", "invalid_evidence_span"}, (name, reasons)
        finally:
            db2.close()


def test_unknown_source_rejected_by_both(tmp_path: Path):
    db, result = run_direct(tmp_path, UnknownSourceProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "d-unk.sqlite")
    try:
        assert result.accepted == 0 and result.rejected == 1
        assert db.memories("n1") == []
        assert "evidence_not_in_extraction_input" in _rejection_reasons(db, "n1")
    finally:
        db.close()
    db2, resp = run_queued(tmp_path, UnknownSourceProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "q-unk.sqlite")
    try:
        assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0
        assert resp.accepted == 0 and resp.rejected == 1
        assert db2.memories("n1") == []
        reasons = _rejection_reasons(db2, "n1")
        assert reasons & {"evidence_not_in_extraction_input", "evidence_not_in_ingestion_batch"}, reasons
    finally:
        db2.close()


def test_cross_namespace_real_event_rejected_by_both(tmp_path: Path):
    """Same database: n2 event exists; n1 extraction cites it with correct excerpt."""
    n2_text = "Decision: use SQLite in another namespace project work."
    n1_text = "I prefer SQLite for local storage."
    foreign_excerpt = n2_text[:5]

    # Direct: seed n2 (valid), then swap to cross-namespace provider for n1.
    db = OpenMem(tmp_path / "d-cross.sqlite", extraction_provider=ValidProvider(), embedding_provider=RecordingEmbedding())
    try:
        db.ingest(_event("n2", "k1", n2_text))
        foreign_id = UUID(_event_id(db, "n2", "k1"))
        db.processor.provider = CrossNamespaceProvider(foreign_id, n2_text)
        result = db.ingest(_event("n1", "k1", n1_text))
        assert result.accepted == 0 and result.rejected == 1
        assert db.memories("n1") == []
        assert "evidence_not_in_extraction_input" in _rejection_reasons(db, "n1")
        # n2 memory untouched.
        assert len(db.memories("n2")) == 1
    finally:
        db.close()
    # Queued: same database, n1 via failing ingest then retry with cross-namespace cite.
    db2 = OpenMem(tmp_path / "q-cross.sqlite", extraction_provider=ValidProvider(), embedding_provider=RecordingEmbedding())
    try:
        db2.ingest(_event("n2", "k1", n2_text))
        foreign_id2 = UUID(_event_id(db2, "n2", "k1"))
        db2.processor.provider = FailingProvider()
        with pytest.raises(ProviderError):
            db2.ingest(_event("n1", "k1", n1_text))
        assert db2.memories("n1") == []
        db2.processor.provider = CrossNamespaceProvider(foreign_id2, n2_text)
        resp = db2.process("n1")
        assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0
        assert resp.accepted == 0 and resp.rejected == 1
        assert db2.memories("n1") == []
        reasons = _rejection_reasons(db2, "n1")
        assert "evidence_not_in_extraction_input" in reasons, reasons
        assert foreign_excerpt == n2_text[:5]
    finally:
        db2.close()


def test_context_only_source_rejected_by_both(tmp_path: Path):
    """Evidence in input but outside the extractable batch must not fall back to batch[0]."""
    prior = {**_event("n1", "k1", "I prefer SQLite for local storage."), "stream_id": "s1"}
    current = {**_event("n1", "k2", "Constraint: deploy the SQLite service in India region."), "stream_id": "s1"}

    db = OpenMem(tmp_path / "ctx-seed-d.sqlite", extraction_provider=ValidProvider(), embedding_provider=RecordingEmbedding())
    try:
        db.ingest(prior)
        db.processor.provider = ContextOnlyProvider()
        result = db.ingest(current)
        assert result.accepted == 0 and result.rejected == 1
        assert len(db.memories("n1")) == 1
        assert "evidence_not_in_ingestion_batch" in _rejection_reasons(db, "n1")
    finally:
        db.close()
    db1 = OpenMem(tmp_path / "ctx-q.sqlite", extraction_provider=ValidProvider(), embedding_provider=RecordingEmbedding())
    try:
        db1.ingest(prior)
        db1.processor.provider = FailingProvider()
        with pytest.raises(ProviderError):
            db1.ingest(current)
        assert len(db1.memories("n1")) == 1
        db1.processor.provider = ContextOnlyProvider()
        resp = db1.process("n1")
        assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0
        assert resp.accepted == 0 and resp.rejected == 1
        assert len(db1.memories("n1")) == 1
        assert "evidence_not_in_ingestion_batch" in _rejection_reasons(db1, "n1")
    finally:
        db1.close()


def test_unknown_chunk_id_rejected_by_both(tmp_path: Path):
    db, result = run_direct(tmp_path, UnknownChunkProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "d-chunk.sqlite")
    try:
        assert result.accepted == 0 and result.rejected == 1
        assert db.memories("n1") == []
        assert "unknown_source_chunk_id" in _rejection_reasons(db, "n1")
    finally:
        db.close()
    db2, resp = run_queued(tmp_path, UnknownChunkProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "q-chunk.sqlite")
    try:
        assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0
        assert resp.accepted == 0 and resp.rejected == 1
        assert db2.memories("n1") == []
        assert "unknown_source_chunk_id" in _rejection_reasons(db2, "n1")
    finally:
        db2.close()


def test_valid_evidence_accepted_with_citations_by_both(tmp_path: Path):
    text = "I prefer SQLite for local storage."
    expected_excerpt = text[:60]
    db, result = run_direct(tmp_path, ValidProvider(), [_event("n1", "k1", text)], "d-valid.sqlite")
    try:
        assert result.accepted == 1 and result.rejected == 0
        assert len(db.memories("n1")) == 1
        cites = _citations(db, "n1")
        assert len(cites) == 1
        assert cites[0]["event_id"] == _event_id(db, "n1", "k1")
        assert cites[0]["ns"] == "n1"
        assert (cites[0]["start_offset"], cites[0]["end_offset"]) == (0, len(expected_excerpt))
        assert cites[0]["excerpt"] == expected_excerpt
    finally:
        db.close()
    db2, resp = run_queued(tmp_path, ValidProvider(), [_event("n1", "k1", text)], "q-valid.sqlite")
    try:
        assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0
        assert resp.accepted == 1 and resp.rejected == 0
        assert len(db2.memories("n1")) == 1
        cites = _citations(db2, "n1")
        assert len(cites) == 1
        assert cites[0]["event_id"] == _event_id(db2, "n1", "k1")
        assert cites[0]["ns"] == "n1"
        assert (cites[0]["start_offset"], cites[0]["end_offset"]) == (0, len(expected_excerpt))
        assert cites[0]["excerpt"] == expected_excerpt
    finally:
        db2.close()


def test_provider_failure_then_queued_retry_succeeds_without_duplication(tmp_path: Path):
    text = "I prefer SQLite for local storage."
    expected_excerpt = text[:60]
    db = OpenMem(tmp_path / "retry.sqlite", extraction_provider=FailingProvider(), embedding_provider=RecordingEmbedding())
    try:
        with pytest.raises(ProviderError) as exc:
            db.ingest(_event("n1", "k1", text))
        assert exc.value.retryable is True
        assert db.memories("n1") == []
        db.processor.provider = ValidProvider()
        resp = db.process("n1")
        assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0
        assert resp.accepted == 1 and resp.rejected == 0
        assert len(db.memories("n1")) == 1
        assert db.database.execute("SELECT COUNT(*) FROM memory_versions WHERE namespace_id='n1'").fetchone()[0] == 1
        before = _citations(db, "n1")
        assert len(before) == 1
        assert before[0]["event_id"] == _event_id(db, "n1", "k1")
        assert before[0]["excerpt"] == expected_excerpt
        resp2 = db.process("n1")
        assert resp2.processed == 0
        assert len(db.memories("n1")) == 1
        assert db.database.execute("SELECT COUNT(*) FROM memory_versions WHERE namespace_id='n1'").fetchone()[0] == 1
        assert _citations(db, "n1") == before
    finally:
        db.close()


def test_retry_with_missing_evidence_rejected_without_memory(tmp_path: Path):
    db = OpenMem(tmp_path / "retrymiss.sqlite", extraction_provider=FailingProvider(), embedding_provider=RecordingEmbedding())
    try:
        with pytest.raises(ProviderError):
            db.ingest(_event("n1", "k1", "I prefer SQLite for local storage."))
        db.processor.provider = MissingProvider()
        resp = db.process("n1")
        assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0
        assert resp.accepted == 0 and resp.rejected == 1
        assert db.memories("n1") == []
        assert "missing_source_evidence" in _rejection_reasons(db, "n1")
    finally:
        db.close()


def _v3_factory_valid(request):
    labels = dict(getattr(request, "event_labels", {}) or {})
    if not labels:
        raise AssertionError("test setup: v3 request has no event labels")
    extractable = list(getattr(request, "extractable_event_ids", []) or [])
    if not extractable:
        raise AssertionError("test setup: v3 request has no extractable events")
    rev = {str(v): k for k, v in labels.items()}
    target_label = None
    for eid in extractable:
        if str(eid) in rev:
            target_label = rev[str(eid)]
            break
    if target_label is None:
        raise AssertionError("test setup: no extractable label in v3 request")
    from tests.test_direct_pipeline import _v3_candidate as _mk

    source = next(iter(request.evidence_text.values()))
    return _mk(source[:120] if source else "User prefers SQLite.", labels=[target_label])


def _v3_factory_context_only(request):
    labels = dict(getattr(request, "event_labels", {}) or {})
    if not labels:
        raise AssertionError("test setup: v3 request has no event labels")
    extractable = set(getattr(request, "extractable_event_ids", []) or [])
    context_labels = [lab for lab, eid in labels.items() if eid not in extractable]
    if not context_labels:
        raise AssertionError("test setup: v3 request has no context labels")
    from tests.test_direct_pipeline import _v3_candidate as _mk

    return _mk("User prefers SQLite for local storage.", labels=[context_labels[0]])


def _v3_factory_spanning(request):
    labels = dict(getattr(request, "event_labels", {}) or {})
    extractable = list(getattr(request, "extractable_event_ids", []) or [])
    context_ids = list(getattr(request, "context_event_ids", []) or [])
    if not extractable or not context_ids:
        raise AssertionError("test setup: v3 spanning needs one context and one extractable event")
    rev = {str(v): k for k, v in labels.items()}
    ctx_label = rev.get(str(context_ids[0]))
    ext_label = rev.get(str(extractable[0]))
    if ctx_label is None or ext_label is None:
        raise AssertionError("test setup: v3 labels do not cover context+extractable")
    from tests.test_direct_pipeline import _v3_candidate as _mk

    return _mk("User prefers SQLite for local storage.", labels=[ctx_label, ext_label])


def test_v3_context_only_rejected_both_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENMEM_EXTRACTION_SCHEMA", "v3")
    prior = {**_event("n1", "k1", "I prefer SQLite for local storage."), "stream_id": "s1"}
    current = {**_event("n1", "k2", "Constraint: deploy the SQLite service in India region."), "stream_id": "s1"}
    db = OpenMem(tmp_path / "v3-ctx-d.sqlite", extraction_provider=V3Provider(_v3_factory_valid), embedding_provider=RecordingEmbedding())
    try:
        seed = db.ingest(prior)
        assert seed.accepted == 1 and seed.rejected == 0
        db.processor.provider = V3Provider(_v3_factory_context_only)
        result = db.ingest(current)
        assert result.accepted == 0 and result.rejected == 1
        assert len(db.memories("n1")) == 1
        assert "context_only_source" in _rejection_reasons(db, "n1")
    finally:
        db.close()
    db2 = OpenMem(tmp_path / "v3-ctx-q.sqlite", extraction_provider=V3Provider(_v3_factory_valid), embedding_provider=RecordingEmbedding())
    try:
        seed = db2.ingest(prior)
        assert seed.accepted == 1
        db2.processor.provider = FailingProvider()
        with pytest.raises(ProviderError):
            db2.ingest(current)
        monkeypatch.setenv("OPENMEM_EXTRACTION_SCHEMA", "v3")
        db2.processor.provider = V3Provider(_v3_factory_context_only)
        resp = db2.process("n1")
        assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0
        assert resp.accepted == 0 and resp.rejected == 1
        assert len(db2.memories("n1")) == 1
        assert "context_only_source" in _rejection_reasons(db2, "n1")
    finally:
        db2.close()


def test_v3_spanning_selects_extractable_source_both_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spanning [context, extractable] labels: processor uses the extractable event.

    Storage policy (repository, stable lifecycle): persisted ``source_event_id``
    keeps the provider's first cited event (here the context prior), while
    ``observed_at`` tracks the latest source time and ``evidence_refs`` cite
    every event. Only ``current``-lifecycle rewrites primary provenance to the
    latest event. The test therefore asserts processor selection (spied
    ``reconcile_candidate`` event) == extractable current separately from the
    persisted ``source_event_id`` == first-cited prior.
    """
    monkeypatch.setenv("OPENMEM_EXTRACTION_SCHEMA", "v3")
    prior_text = "I prefer SQLite for local storage."
    current_text = "Constraint: deploy the SQLite service in India region."
    prior_occurred = "2023-04-01T00:00:00+00:00"
    current_occurred = "2023-05-01T00:00:00+00:00"
    prior = {**_event("n1", "k1", prior_text), "stream_id": "s1", "occurred_at": prior_occurred}
    current = {**_event("n1", "k2", current_text), "stream_id": "s1", "occurred_at": current_occurred}

    def _install_spy(db):
        seen: list[tuple[str, str]] = []
        orig = db.repository.reconcile_candidate

        def spy(namespace_id, event, candidate, run_id, embedding=None, **kwargs):
            try:
                seen.append((str(candidate.candidate.statement), str(event["id"])))
            except Exception:
                pass
            return orig(namespace_id, event, candidate, run_id, embedding, **kwargs)

        monkeypatch.setattr(db.repository, "reconcile_candidate", spy)
        return seen

    def _check_spanning(db, current_eid, prior_eid, seen):
        # Processor selection: spanning statement reconciled with extractable current.
        matches = [eid for stmt, eid in seen if stmt == "User prefers SQLite for local storage."]
        assert matches, "test setup: spanning candidate never reached reconcile"
        assert matches[-1] == current_eid
        ver = db.database.execute(
            "SELECT id, source_event_id, observed_at, source_event_ids_json FROM memory_versions"
            " WHERE namespace_id='n1' AND statement='User prefers SQLite for local storage.'"
        ).fetchone()
        assert ver is not None
        # Both citations with exact offsets/excerpts.
        refs = {
            r["event_id"]: r
            for r in db.database.execute(
                "SELECT event_id, start_offset, end_offset, excerpt FROM evidence_refs WHERE namespace_id=? AND memory_version_id=?",
                ("n1", ver["id"]),
            ).fetchall()
        }
        assert set(refs) == {prior_eid, current_eid}
        assert (refs[prior_eid]["start_offset"], refs[prior_eid]["end_offset"]) == (0, len(prior_text))
        assert refs[prior_eid]["excerpt"] == prior_text
        assert (refs[current_eid]["start_offset"], refs[current_eid]["end_offset"]) == (0, len(current_text))
        assert refs[current_eid]["excerpt"] == current_text
        # Persisted primary provenance keeps first-cited (context prior) for stable;
        # observed_at tracks the latest source time.
        assert str(ver["source_event_id"]) == prior_eid
        assert str(ver["observed_at"]) == current_occurred

    db = OpenMem(tmp_path / "v3-span-d.sqlite", extraction_provider=V3Provider(_v3_factory_valid), embedding_provider=RecordingEmbedding())
    try:
        seed = db.ingest(prior)
        assert seed.accepted == 1 and seed.rejected == 0
        db.processor.provider = V3Provider(_v3_factory_spanning)
        seen = _install_spy(db)
        result = db.ingest(current)
        assert result.accepted == 1 and result.rejected == 0
        _check_spanning(db, _event_id(db, "n1", "k2"), _event_id(db, "n1", "k1"), seen)
    finally:
        db.close()
    db2 = OpenMem(tmp_path / "v3-span-q.sqlite", extraction_provider=V3Provider(_v3_factory_valid), embedding_provider=RecordingEmbedding())
    try:
        seed = db2.ingest(prior)
        assert seed.accepted == 1
        db2.processor.provider = FailingProvider()
        with pytest.raises(ProviderError):
            db2.ingest(current)
        monkeypatch.setenv("OPENMEM_EXTRACTION_SCHEMA", "v3")
        db2.processor.provider = V3Provider(_v3_factory_spanning)
        seen2 = _install_spy(db2)
        resp = db2.process("n1")
        assert resp.processed == 1 and resp.failed == 0 and resp.dead_lettered == 0
        assert resp.accepted == 1 and resp.rejected == 0
        _check_spanning(db2, _event_id(db2, "n1", "k2"), _event_id(db2, "n1", "k1"), seen2)
    finally:
        db2.close()
