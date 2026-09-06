"""Regression coverage for forgotten/invalidated memory leakage in historical search.

Lifecycle rule:
- ``deleted`` / ``invalidated`` (on either memories or memory_versions)
  must never appear in ordinary or historical search.
- ``superseded`` remains retrievable via ``historical=True``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src import TermyteDB

from .conftest import event


class ConstantEmbedding:
    """Deterministic embedding: every text maps to the same unit vector.

    Guarantees fallback vector matches regardless of lexical overlap, so a
    non-lexical query isolates the dense channel from FTS.
    """

    name = "test-constant-v1"
    dimensions = 2

    def embed(self, value: str) -> list[float]:
        return [1.0, 0.0]

    def embed_many(self, values: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in values]


def make_db(tmp_path: Path, name: str = "t.sqlite") -> TermyteDB:
    return TermyteDB(tmp_path / name, embedding_provider=ConstantEmbedding())


def _memory_id(db: TermyteDB, namespace: str) -> str:
    memories = db.memories(namespace)
    assert memories, "expected at least one memory"
    return str(memories[0].memory_id)


def _version_id(db: TermyteDB, namespace: str, memory_id: str) -> str:
    row = db.database.execute(
        "SELECT current_version_id FROM memories WHERE id=? AND namespace_id=?",
        (memory_id, namespace),
    ).fetchone()
    assert row is not None
    return str(row["current_version_id"])


# The paraphrase shares no FTS terms with the stored statement, so lexical
# (FTS and LIKE) cannot match; only dense vector / chunk-dense can.
NON_LEXICAL_QUERY = "persistence layer choice"
LEXICAL_QUERY = "SQLite"


def test_historical_forgetting_regression(tmp_path: Path):
    """Core regression: forget must exclude from both search modes."""
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        memory_id = _memory_id(db, "n1")
        assert db.search("n1", LEXICAL_QUERY)
        assert db.repository.search("n1", LEXICAL_QUERY, 10, historical=True)
        assert db.forget("n1", memory_id, "user requested forgetting") is True
        assert db.search("n1", LEXICAL_QUERY) == []
        assert db.repository.search("n1", LEXICAL_QUERY, 10, historical=True) == []
        # Audit history retained.
        history = db.history("n1", memory_id)
        assert history is not None
        assert any(row["status"] == "deleted" for row in history)
    finally:
        db.close()


def test_historical_invalidation_excluded_both_modes(tmp_path: Path):
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        memory_id = _memory_id(db, "n1")
        assert db.invalidate("n1", memory_id, "bad source") is True
        assert db.search("n1", LEXICAL_QUERY) == []
        assert db.repository.search("n1", LEXICAL_QUERY, 10, historical=True) == []
        history = db.history("n1", memory_id)
        assert history is not None
        assert any(row["status"] == "invalidated" for row in history)
        # Invalidated memories are not restorable.
        assert db.restore("n1", memory_id) is False
        assert db.repository.search("n1", LEXICAL_QUERY, 10, historical=True) == []
    finally:
        db.close()


def test_historical_retrieves_superseded(tmp_path: Path):
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        old = db.search("n1", LEXICAL_QUERY)[0]
        db.ingest(event("n1", "two", "Decision: storage uses PostgreSQL."))
        db.process("n1")
        assert all(r.memory_version_id != old.memory_version_id for r in db.search("n1", LEXICAL_QUERY))
        historical = db.repository.search("n1", LEXICAL_QUERY, 10, historical=True)
        assert any(r.status == "superseded" and "SQLite" in r.statement for r in historical)
    finally:
        db.close()


def test_restore_forgotten_searchable(tmp_path: Path):
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        memory_id = _memory_id(db, "n1")
        db.forget("n1", memory_id, "tmp")
        assert db.search("n1", LEXICAL_QUERY) == []
        assert db.restore("n1", memory_id) is True
        ordinary = db.search("n1", LEXICAL_QUERY)
        assert len(ordinary) == 1 and str(ordinary[0].memory_id) == memory_id
        historical = db.repository.search("n1", LEXICAL_QUERY, 10, historical=True)
        assert any(str(r.memory_id) == memory_id for r in historical)
    finally:
        db.close()


def test_namespace_isolation_after_forget(tmp_path: Path):
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.ingest(event("n2", "two", "Decision: storage uses SQLite."))
        db.process("n1")
        db.process("n2")
        mid_n1 = str(db.search("n1", LEXICAL_QUERY)[0].memory_id)
        mid_n2 = str(db.search("n2", LEXICAL_QUERY)[0].memory_id)
        assert mid_n1 != mid_n2
        db.forget("n1", mid_n1, "tmp")
        assert db.search("n1", LEXICAL_QUERY) == []
        assert db.repository.search("n1", LEXICAL_QUERY, 10, historical=True) == []
        assert len(db.search("n2", LEXICAL_QUERY)) == 1
        assert len(db.repository.search("n2", LEXICAL_QUERY, 10, historical=True)) == 1
    finally:
        db.close()


def test_fallback_vector_excludes_deleted_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fallback dense path in isolation: lexical + chunk disabled, non-lexical query."""
    db = make_db(tmp_path)
    try:
        # Force fallback SQL (no sqlite-vec index in this environment anyway).
        db.repository.vector_index.available = False
        # Disable chunk channel so only fallback vector can produce hits.
        monkeypatch.setattr(db.repository, "_chunk_hits", lambda *a, **k: {})
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        memory_id = _memory_id(db, "n1")
        # Constant embedding matches even without lexical overlap.
        assert db.search("n1", NON_LEXICAL_QUERY), "fallback vector should match pre-forget"
        assert db.repository.search("n1", NON_LEXICAL_QUERY, 10, historical=True)
        db.forget("n1", memory_id, "tmp")
        assert db.search("n1", NON_LEXICAL_QUERY) == []
        assert db.repository.search("n1", NON_LEXICAL_QUERY, 10, historical=True) == []
    finally:
        db.close()


def test_fallback_vector_excludes_invalidated_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = make_db(tmp_path)
    try:
        db.repository.vector_index.available = False
        monkeypatch.setattr(db.repository, "_chunk_hits", lambda *a, **k: {})
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        memory_id = _memory_id(db, "n1")
        db.invalidate("n1", memory_id, "bad")
        assert db.search("n1", NON_LEXICAL_QUERY) == []
        assert db.repository.search("n1", NON_LEXICAL_QUERY, 10, historical=True) == []
    finally:
        db.close()


def test_indexed_vector_excludes_deleted_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Mocked indexed path in isolation: lexical miss + chunk disabled."""
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        memory_id = _memory_id(db, "n1")
        version_id = _version_id(db, "n1", memory_id)
        monkeypatch.setattr(db.repository, "_chunk_hits", lambda *a, **k: {})
        # Mock index to always return the target version.
        monkeypatch.setattr(
            db.repository.vector_index,
            "search",
            lambda namespace_id, query_vector, limit: [(version_id, 0.99)],
        )
        # Non-lexical query: lexical contributes nothing, so hits must come
        # from the mocked indexed channel (plus final filter).
        assert any(str(r.memory_version_id) == version_id for r in db.search("n1", NON_LEXICAL_QUERY)), "mocked index should match pre-forget"
        assert any(str(r.memory_version_id) == version_id for r in db.repository.search("n1", NON_LEXICAL_QUERY, 10, historical=True))
        db.forget("n1", memory_id, "tmp")
        assert db.search("n1", NON_LEXICAL_QUERY) == []
        assert db.repository.search("n1", NON_LEXICAL_QUERY, 10, historical=True) == []
    finally:
        db.close()


def test_indexed_vector_historical_includes_superseded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        old = db.search("n1", LEXICAL_QUERY)[0]
        old_vid = str(old.memory_version_id)
        db.ingest(event("n1", "two", "Decision: storage uses PostgreSQL."))
        db.process("n1")
        monkeypatch.setattr(db.repository, "_chunk_hits", lambda *a, **k: {})
        monkeypatch.setattr(
            db.repository.vector_index,
            "search",
            lambda namespace_id, query_vector, limit: [(old_vid, 0.99)],
        )
        assert all(str(r.memory_version_id) != old_vid for r in db.search("n1", NON_LEXICAL_QUERY))
        historical = db.repository.search("n1", NON_LEXICAL_QUERY, 10, historical=True)
        assert any(str(r.memory_version_id) == old_vid and r.status == "superseded" for r in historical)
    finally:
        db.close()


def test_chunk_channel_excludes_deleted(tmp_path: Path):
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        memory_id = _memory_id(db, "n1")
        query_vector = db.repository.embedding.embed(LEXICAL_QUERY)
        assert db.repository._chunk_hits("n1", LEXICAL_QUERY, query_vector, limit=10), "chunk channel should match pre-forget"
        db.forget("n1", memory_id, "tmp")
        assert db.repository._chunk_hits("n1", LEXICAL_QUERY, query_vector, limit=10) == {}
        db.restore("n1", memory_id)
        assert db.repository._chunk_hits("n1", LEXICAL_QUERY, query_vector, limit=10), "chunk channel should match after restore"
    finally:
        db.close()


@pytest.mark.parametrize(
    "table,status", [("memories", "deleted"), ("memory_versions", "deleted"), ("memories", "invalidated"), ("memory_versions", "invalidated")]
)
def test_parent_and_version_level_exclusions(tmp_path: Path, table: str, status: str):
    """Either parent or version tombstone must exclude from both search modes."""
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        memory_id = _memory_id(db, "n1")
        version_id = _version_id(db, "n1", memory_id)
        with db.database.connection:
            if table == "memories":
                db.database.execute("UPDATE memories SET status=? WHERE id=? AND namespace_id=?", (status, memory_id, "n1"))
            else:
                db.database.execute("UPDATE memory_versions SET status=? WHERE id=? AND namespace_id=?", (status, version_id, "n1"))
        assert db.search("n1", LEXICAL_QUERY) == [], f"{table}={status} leaked into ordinary search"
        assert db.repository.search("n1", LEXICAL_QUERY, 10, historical=True) == [], f"{table}={status} leaked into historical search"
    finally:
        db.close()


def test_lifecycle_filter_failure_discards_chunk_channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """On lifecycle-filter DB failure, chunk candidates are discarded with a diagnostic."""
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        real_execute = db.database.execute

        def failing_execute(sql: str, params: tuple = ()):  # type: ignore[no-untyped-def]
            if "NOT IN ('deleted', 'invalidated')" in sql and "FROM memory_versions v" in sql:
                raise RuntimeError("simulated lifecycle-filter failure")
            return real_execute(sql, params)

        monkeypatch.setattr(db.database, "execute", failing_execute)
        with caplog.at_level(logging.WARNING):
            hits = db.repository._chunk_hits("n1", LEXICAL_QUERY, [1.0, 0.0], limit=10)
        assert hits == {}, "unchecked chunk candidates must not be retained"
        assert any("lifecycle filter failed for chunk candidates" in r.message for r in caplog.records), "expected observable diagnostic"
    finally:
        db.close()


def test_lifecycle_filter_failure_discards_indexed_channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """On lifecycle-filter DB failure, indexed candidates are discarded with a diagnostic."""
    db = make_db(tmp_path)
    try:
        db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
        db.process("n1")
        memory_id = _memory_id(db, "n1")
        version_id = _version_id(db, "n1", memory_id)
        monkeypatch.setattr(db.repository, "_chunk_hits", lambda *a, **k: {})
        monkeypatch.setattr(
            db.repository.vector_index,
            "search",
            lambda namespace_id, query_vector, limit: [(version_id, 0.99)],
        )
        # Remove the fallback embedding so the mocked index is the sole dense
        # channel; otherwise fallback would legitimately return the active
        # memory and mask whether the indexed channel was discarded.
        with db.database.connection:
            db.database.execute("DELETE FROM memory_embeddings WHERE memory_version_id=?", (version_id,))
        real_execute = db.database.execute

        def failing_execute(sql: str, params: tuple = ()):  # type: ignore[no-untyped-def]
            # Fail only the indexed allow-list query; let final filtering run.
            if "SELECT v.id AS id FROM memory_versions v" in sql:
                raise RuntimeError("simulated indexed-filter failure")
            return real_execute(sql, params)

        monkeypatch.setattr(db.database, "execute", failing_execute)
        with caplog.at_level(logging.WARNING):
            # Non-lexical query isolates the indexed channel.
            ordinary = db.search("n1", NON_LEXICAL_QUERY)
        assert ordinary == [], "discarded indexed channel must leave no unchecked candidates"
        assert any("lifecycle filter failed for indexed vector candidates" in r.message for r in caplog.records), "expected observable diagnostic"
    finally:
        db.close()
