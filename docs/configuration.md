# Configuration reference

OpenMem works offline with explicit providers. Environment variables are read when provider/settings objects are created.

## Extraction

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENMEM_EXTRACTION_MODEL` | none | Required model for `OpenRouterExtractionProvider`. |
| `OPENROUTER_API_KEY` | none | API key fallback for OpenRouter providers. |
| `OPENMEM_EXTRACTION_API_KEY` | none | Extraction-specific API key. |
| `OPENMEM_EXTRACTION_BASE_URL` | OpenRouter API URL | OpenAI-compatible base URL. |
| `OPENMEM_EXTRACTION_URL` | none | Endpoint for the compatibility HTTP provider. |
| `OPENMEM_EXTRACTION_TEMPERATURE` | `0` | Extraction temperature. |
| `OPENMEM_EXTRACTION_MAX_TOKENS` | `3500` for OpenRouter calls | Maximum extraction response tokens. |
| `OPENMEM_EXTRACTION_RETRIES` | provider-defined | Retry count for extraction requests. |
| `OPENMEM_EXTRACTION_STAGES` | `facts` | Comma-separated extraction stages. |
| `OPENMEM_RECONCILIATION_ENABLED` | enabled | Enable memory reconciliation. |
| `OPENMEM_RECONCILIATION_MODEL` | extraction model | Model used for reconciliation. |
| `OPENMEM_SUMMARY_ENABLED` | enabled | Enable session summaries. |
| `OPENMEM_SUMMARY_MODEL` | extraction model | Model used for summaries. |

Set boolean values to `0`, `false`, `no`, or `off` to disable a feature.

## Embeddings and reranking

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENMEM_EMBEDDING_MODEL` | empty | Remote OpenAI-compatible embedding model. |
| `OPENMEM_EMBEDDING_DIMENSIONS` | `1024` | Remote embedding vector dimensions. |
| `OPENMEM_EMBEDDING_BASE_URL` | OpenRouter API URL | Remote embedding base URL. |
| `OPENMEM_EMBEDDING_RETRIES` | `6` | Remote embedding retry count. |
| `OPENMEM_RERANKER_ENABLED` | enabled | Enable reranking. |
| `OPENMEM_RERANKER_MAX_CANDIDATES` | `30` | Maximum reranker candidates. |
| `OPENMEM_RERANKER_MAX_CHARS` | `600` | Maximum text length sent to reranking. |
| `OPENMEM_RERANKING_MODEL` | empty | Optional remote reranking model. |
| `OPENMEM_CHUNK_NO_DENSE` | enabled | Use chunk search when dense search is unavailable. |

The default local embedding model is `BAAI/bge-small-en-v1.5`, with 384 dimensions. The default local reranker is `ms-marco-MiniLM-L-12-v2`.

## Retrieval tuning

Temporal and multi-session weights can be changed with the `OPENMEM_TEMPORAL_*`, `OPENMEM_PREFERENCE_*`, and `OPENMEM_MULTI_SESSION_*` variables defined in `src/config/settings.py`. Keep these values versioned in deployment configuration so benchmark results remain reproducible.

## Security notes

Never put real keys in source, tests, benchmark fixtures, or committed `.env` files. OpenMem redacts common secrets from event payloads before storage, but applications should still avoid sending credentials as event content.
