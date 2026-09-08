# API reference

The public entry point is `src.OpenMem`. Methods accept a namespace ID so one database file can hold isolated tenants or agents.

## Construction

```python
OpenMem(
    path: str | Path | None = None,
    *,
    database: Database | None = None,
    logger: logging.Logger | None = None,
    extraction_provider: ExtractionProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    summary_provider: SessionSummaryProvider | None = None,
)
```

Pass exactly one of `path` or `database`. An explicit extraction provider is recommended for production and tests.

## Core operations

| Method | Purpose |
| --- | --- |
| `ingest(event)` | Validate, redact, persist, index, and process one event. |
| `ingest_batch(events)` | Ingest multiple events from one namespace. |
| `search(namespace_id, query, limit=10, historical=False, reference_date=None)` | Return ranked `SearchResult` memories. |
| `get_memory(namespace_id, memory_id)` | Read one memory or return `None`. |
| `memories(namespace_id, limit=100, offset=0)` | List memories. |
| `history(namespace_id, memory_id)` | Return memory version history. |
| `search_context(namespace_id, query, limit=20, reference_date=None)` | Return memories, source chunks, packed text, token count, and session fallbacks. |
| `build_answer_context(namespace_id, query, limit=6, token_budget=1200, reference_date=None)` | Pack evidence for answer generation within a token budget. |
| `process(namespace_id, limit=100, lease_seconds=180)` | Retry queued processing jobs. |

## Memory lifecycle

| Method | Effect |
| --- | --- |
| `update_memory(...)` | Update a statement and optional confidence, kind, source, or evidence. |
| `invalidate(namespace_id, memory_id, reason)` | Mark a memory invalid with a reason. |
| `forget(namespace_id, memory_id, reason)` | Remove a memory from active use while retaining lifecycle history. |
| `restore(namespace_id, memory_id)` | Restore a forgotten or invalidated memory. |
| `delete_memory(namespace_id, memory_id)` | Delete a memory record. Use carefully. |

## Operations and maintenance

`events`, `event`, `evidence`, `jobs`, `extraction_runs`, `extraction_decisions`, `feedback_rows`, `metrics`, and `encoding_decisions` expose stored operational data. `export_namespace` and `import_namespace` support namespace migration. `backup(destination)` creates a SQLite backup and `checkpoint()` checkpoints the database.

`close()` closes the database and should be called when the engine is no longer needed.

## Event model

`EventInput` uses `protocol_version="event-v1"` by default. `type` is 1–100 characters. `artifacts` accepts up to 20 items; each artifact has a `sha256:` content hash and a size from 0 to 100,000,000 bytes. The redacted JSON payload is limited to 1 MiB at ingestion.

## Provider interfaces

- `FakeExtractionProvider` — deterministic, offline provider for tests and examples.
- `OpenRouterExtractionProvider` — OpenAI-compatible network provider.
- `HttpExtractionProvider` — compatibility provider for `OPENMEM_EXTRACTION_URL`.
- `OpenRouterSessionSummaryProvider` — optional session summary provider.

See [configuration](configuration.md) for environment variables and defaults.
