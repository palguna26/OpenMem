"""Evidence-contract consistency: direct vs queued processing (Task 3A).

Both paths must reject evidence-free, malformed, and out-of-scope candidates
with explicit diagnostics; transport failures stay retryable.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src import TermyteDB
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
    eid = request.events[0]
    text = request.evidence_text[eid]
    excerpt = text[:20]
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
        eid = request.events[0]
        text = request.evidence_text[eid]
        cand = _valid_candidate(request)
        bad = cand.model_copy(
            update={"evidence": [EvidenceSpan(event_id=eid, start_offset=0, end_offset=5, excerpt="WRONG")]},
        )
        assert text[:5] != "WRONG"
        return _result([bad], self.name, self.model)


class BadOffsetProvider:
    name = "bad-offset"
    model = "bad-offset-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
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
    name = "cross-ns"
    model = "cross-ns-v1"

    def __init__(self, foreign_event_id):
        self.foreign_event_id = foreign_event_id

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        return _result(
            [
                ExtractionCandidate(
                    kind="fact",
                    subject="user preference",
                    statement="User prefers SQLite for local storage.",
                    evidence=[EvidenceSpan(event_id=self.foreign_event_id, start_offset=0, end_offset=5, excerpt="hello")],
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
        # Prefer an explicit context id when the request splits them.
        context_ids = list(getattr(request, "context_event_ids", []) or [])
        targets = [eid for eid in context_ids if eid in request.evidence_text]
        if not targets:
            targets = [eid for eid in request.evidence_text if eid not in extractable]
        if not targets:
            # No context available (test setup error) — fall back to valid.
            return _result([_valid_candidate(request)], self.name, self.model)
        eid = targets[0]
        text = request.evidence_text[eid]
        excerpt = text[:20]
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


def _event(ns, key, text):
    return {"namespace_id": ns, "idempotency_key": key, "type": "conversation", "payload": {"text": text}}


def run_direct(tmp_path: Path, provider, events, name="t.sqlite"):
    db = TermyteDB(tmp_path / name, extraction_provider=provider, embedding_provider=RecordingEmbedding())
    try:
        result = db.ingest_batch(events)
        return db, result
    except Exception:
        return db, None


def run_queued(tmp_path: Path, provider, events, name="t.sqlite"):
    """Exercise process_namespace via a failing direct ingest then provider swap."""
    db = TermyteDB(tmp_path / name, extraction_provider=FailingProvider(), embedding_provider=RecordingEmbedding())
    try:
        for ev in events:
            with pytest.raises(ProviderError):
                db.ingest(ev)
        assert db.memories(events[0]["namespace_id"]) == []
        db.processor.provider = provider
        resp = db.process(events[0]["namespace_id"])
        return db, resp
    except Exception:
        return db, None


def _rejection_reasons(db, namespace):
    rows = db.database.execute(
        "SELECT rejection_reason FROM extraction_decisions WHERE namespace_id=? AND validation_status='rejected'",
        (namespace,),
    ).fetchall()
    return {r[0] for r in rows}


def test_missing_evidence_rejected_by_both(tmp_path: Path):
    db, result = run_direct(tmp_path, MissingProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "d1.sqlite")
    try:
        assert result is not None and result.accepted == 0 and result.rejected == 1
        assert db.memories("n1") == []
        assert "missing_source_evidence" in _rejection_reasons(db, "n1")
    finally:
        db.close()
    db2, resp = run_queued(tmp_path, MissingProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "q1.sqlite")
    try:
        assert resp is not None
        assert db2.memories("n1") == []
        assert "missing_source_evidence" in _rejection_reasons(db2, "n1")
    finally:
        db2.close()


def test_incorrect_excerpt_or_offsets_rejected_by_both(tmp_path: Path):
    for provider, name in ((BadExcerptProvider(), "excerpt"), (BadOffsetProvider(), "offset")):
        db, result = run_direct(tmp_path, provider, [_event("n1", f"k-{name}", "I prefer SQLite for local storage.")], f"d-{name}.sqlite")
        try:
            assert result is not None and result.accepted == 0, name
            assert db.memories("n1") == [], name
            reasons = _rejection_reasons(db, "n1")
            assert reasons & {"evidence_excerpt_mismatch", "invalid_evidence_span"}, (name, reasons)
        finally:
            db.close()
        db2, resp = run_queued(tmp_path, provider.__class__(), [_event("n1", f"k-{name}", "I prefer SQLite for local storage.")], f"q-{name}.sqlite")
        try:
            assert db2.memories("n1") == [], name
            reasons = _rejection_reasons(db2, "n1")
            assert reasons & {"evidence_excerpt_mismatch", "invalid_evidence_span"}, (name, reasons)
        finally:
            db2.close()


def test_unknown_and_cross_namespace_source_rejected_by_both(tmp_path: Path):
    # Unknown random event.
    db, result = run_direct(tmp_path, UnknownSourceProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "d-unk.sqlite")
    try:
        assert result is not None and result.accepted == 0
        assert db.memories("n1") == []
        assert "evidence_not_in_extraction_input" in _rejection_reasons(db, "n1")
    finally:
        db.close()
    db2, _ = run_queued(tmp_path, UnknownSourceProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "q-unk.sqlite")
    try:
        assert db2.memories("n1") == []
        reasons = _rejection_reasons(db2, "n1")
        assert reasons & {"evidence_not_in_extraction_input", "evidence_not_in_ingestion_batch"}, reasons
    finally:
        db2.close()
    # Cross-namespace: seed n2 event, then cite it from n1.
    seed = TermyteDB(tmp_path / "seed.sqlite", extraction_provider=ValidProvider(), embedding_provider=RecordingEmbedding())
    try:
        seed.ingest(_event("n2", "k1", "Decision: use SQLite in another namespace project work."))
        foreign_id = seed.database.execute("SELECT id FROM events WHERE namespace_id='n2'").fetchone()[0]
        from uuid import UUID as _UUID

        foreign_uuid = _UUID(str(foreign_id))
    finally:
        seed.close()
    db3 = TermyteDB(tmp_path / "d-cross.sqlite", extraction_provider=CrossNamespaceProvider(foreign_uuid), embedding_provider=RecordingEmbedding())
    try:
        result = db3.ingest(_event("n1", "k1", "I prefer SQLite for local storage."))
        assert result.accepted == 0
        assert db3.memories("n1") == []
        assert "evidence_not_in_extraction_input" in _rejection_reasons(db3, "n1")
    finally:
        db3.close()
    db4 = TermyteDB(tmp_path / "q-cross.sqlite", extraction_provider=FailingProvider(), embedding_provider=RecordingEmbedding())
    try:
        with pytest.raises(ProviderError):
            db4.ingest(_event("n1", "k1", "I prefer SQLite for local storage."))
        # Reuse the same foreign id (an n2 event id is unknown to n1's input).
        db4.processor.provider = CrossNamespaceProvider(foreign_uuid)
        db4.process("n1")
        assert db4.memories("n1") == []
        reasons = _rejection_reasons(db4, "n1")
        assert reasons & {"evidence_not_in_extraction_input", "evidence_not_in_ingestion_batch"}, reasons
    finally:
        db4.close()


def test_context_only_source_rejected_by_both(tmp_path: Path):
    """Evidence in input but outside the extractable batch must not fall back to batch[0]."""
    prior = {**_event("n1", "k1", "I prefer SQLite for local storage."), "stream_id": "s1"}
    current = {**_event("n1", "k2", "Constraint: deploy the SQLite service in India region."), "stream_id": "s1"}

    def _seed(db_path: str):
        db = TermyteDB(tmp_path / db_path, extraction_provider=ValidProvider(), embedding_provider=RecordingEmbedding())
        try:
            db.ingest(prior)
            return db
        except Exception:
            db.close()
            raise

    # Direct: prior exists, current cites only prior (context).
    db0 = _seed("ctx-seed-d.sqlite")
    try:
        db0.processor.provider = ContextOnlyProvider()
        result = db0.ingest(current)
        assert result.accepted == 0, result
        # Only the prior memory exists; no memory from context-only citation.
        assert len(db0.memories("n1")) == 1
        assert "evidence_not_in_ingestion_batch" in _rejection_reasons(db0, "n1")
    finally:
        db0.close()
    # Queued: same via failing ingest then retry.
    db1 = TermyteDB(tmp_path / "ctx-q.sqlite", extraction_provider=ValidProvider(), embedding_provider=RecordingEmbedding())
    try:
        db1.ingest(prior)
        db1.processor.provider = FailingProvider()
        with pytest.raises(ProviderError):
            db1.ingest(current)
        assert len(db1.memories("n1")) == 1
        db1.processor.provider = ContextOnlyProvider()
        db1.process("n1")
        assert len(db1.memories("n1")) == 1
        assert "evidence_not_in_ingestion_batch" in _rejection_reasons(db1, "n1")
    finally:
        db1.close()


def test_unknown_chunk_id_rejected_by_both(tmp_path: Path):
    db, result = run_direct(tmp_path, UnknownChunkProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "d-chunk.sqlite")
    try:
        assert result is not None and result.accepted == 0
        assert db.memories("n1") == []
        assert "unknown_source_chunk_id" in _rejection_reasons(db, "n1")
    finally:
        db.close()
    db2, _ = run_queued(tmp_path, UnknownChunkProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "q-chunk.sqlite")
    try:
        assert db2.memories("n1") == []
        assert "unknown_source_chunk_id" in _rejection_reasons(db2, "n1")
    finally:
        db2.close()


def test_valid_evidence_accepted_with_citations_by_both(tmp_path: Path):
    db, result = run_direct(tmp_path, ValidProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "d-valid.sqlite")
    try:
        assert result is not None and result.accepted == 1
        mems = db.memories("n1")
        assert len(mems) == 1
        refs = db.database.execute("SELECT COUNT(*) FROM evidence_refs WHERE namespace_id='n1'").fetchone()[0]
        assert refs >= 1
    finally:
        db.close()
    db2, resp = run_queued(tmp_path, ValidProvider(), [_event("n1", "k1", "I prefer SQLite for local storage.")], "q-valid.sqlite")
    try:
        mems = db2.memories("n1")
        assert len(mems) == 1
        refs = db2.database.execute("SELECT COUNT(*) FROM evidence_refs WHERE namespace_id='n1'").fetchone()[0]
        assert refs >= 1
    finally:
        db2.close()


def test_provider_failure_then_queued_retry_succeeds_without_duplication(tmp_path: Path):
    db = TermyteDB(tmp_path / "retry.sqlite", extraction_provider=FailingProvider(), embedding_provider=RecordingEmbedding())
    try:
        with pytest.raises(ProviderError) as exc:
            db.ingest(_event("n1", "k1", "I prefer SQLite for local storage."))
        assert exc.value.retryable is True
        assert db.memories("n1") == []
        db.processor.provider = ValidProvider()
        resp = db.process("n1")
        assert resp.accepted == 1
        assert len(db.memories("n1")) == 1
        assert db.database.execute("SELECT COUNT(*) FROM memory_versions WHERE namespace_id='n1'").fetchone()[0] == 1
        assert db.database.execute("SELECT COUNT(*) FROM evidence_refs WHERE namespace_id='n1'").fetchone()[0] >= 1
        # Second retry processes nothing and duplicates nothing.
        resp2 = db.process("n1")
        assert len(db.memories("n1")) == 1
        assert db.database.execute("SELECT COUNT(*) FROM memory_versions WHERE namespace_id='n1'").fetchone()[0] == 1
        assert resp2.processed == 0
    finally:
        db.close()


def test_retry_with_missing_evidence_rejected_without_memory(tmp_path: Path):
    db = TermyteDB(tmp_path / "retrymiss.sqlite", extraction_provider=FailingProvider(), embedding_provider=RecordingEmbedding())
    try:
        with pytest.raises(ProviderError):
            db.ingest(_event("n1", "k1", "I prefer SQLite for local storage."))
        db.processor.provider = MissingProvider()
        db.process("n1")
        assert db.memories("n1") == []
        assert "missing_source_evidence" in _rejection_reasons(db, "n1")
    finally:
        db.close()
