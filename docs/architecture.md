# Architecture

OpenMem is a local, event-first memory pipeline. The event is the durable source record; extracted memories are derived records that retain links back to that source.

```text
Event input
    |
    v
Redaction + validation
    |
    v
SQLite event store + idempotency
    |
    +--> processing job
    +--> source chunks --> FTS / vector index
    |
    v
Extraction provider
    |
    v
Evidence-backed memory versions
    |
    v
Hybrid retrieval --> temporal / preference ranking --> caller context
```

## Events and namespaces

`EventInput` is the ingestion contract. It requires `namespace_id`, `idempotency_key`, `type`, and a JSON-like `payload`. Optional timestamps and actor, agent, session, stream, and source identifiers preserve context.

All events in one `ingest_batch` call must use the same namespace. Namespaces isolate reads, writes, search, jobs, and exports.

## Durability and retries

OpenMem writes a redacted event before extraction. It then creates a processing job and indexes source chunks. Provider failures do not erase the event. The job records the error and can be retried with `process`.

This ordering makes the database useful even when an LLM provider is unavailable and makes failures inspectable through the job and extraction-run APIs.

## Extraction and evidence

Providers return candidate memories. The processor validates candidates, records evidence spans, and reconciles candidates with existing memories. A memory can be reinforced, updated, superseded, disputed, or ignored.

Search results include source event IDs and citations. Applications can use `search_context` or `build_answer_context` when they need source chunks packed into a token budget.

## Retrieval

Retrieval combines lexical search with vector similarity. Candidate results are rank-fused, optionally reranked, and adjusted for temporal intent and preference polarity. Non-historical search favors memories active at the relevant reference date; `historical=True` includes historical records.

The optional `reference_date` is important for evaluation and replay: it makes words such as “current” depend on the question’s time instead of the machine clock.

## Storage boundary

SQLite is the system of record. SQLite-vec stores vector representations, while local FastEmbed is the default embedding path when no custom embedding provider is supplied. The public facade is in `src/engine.py`; storage implementation details are under `src/storage/`.

## Trade-offs

- SQLite keeps deployment simple and data local, but this project is not a distributed database.
- Evidence and history use more storage than a single mutable row, but they make answers auditable.
- Hybrid retrieval is more robust across identifiers and concepts than one search mode, but it has more tuning knobs.
- Provider calls improve extraction quality, but the fake provider is the only zero-network path.
