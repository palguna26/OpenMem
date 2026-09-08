# OpenMem

OpenMem is an embedded memory engine for AI agents. It turns conversation events into durable, evidence-backed memories and retrieves them with hybrid lexical and vector search.

It runs in one Python process and stores data in SQLite. OpenMem owns memory storage and retrieval; your application decides how retrieved memories become model context.

> Early-stage project: the public API and storage format may change before the first stable release.

## What it does

- Stores raw events durably before running extraction.
- Redacts common secrets before persistence and extraction.
- Extracts facts, decisions, preferences, procedures, task state, and related memory kinds.
- Keeps source event IDs and evidence excerpts with memories.
- Handles updates, superseded memories, invalidation, forgetting, restoration, and history.
- Searches with SQLite FTS plus vector similarity, with temporal and preference-aware ranking.
- Supports offline tests through `FakeExtractionProvider`.
- Supports OpenAI-compatible extraction through OpenRouter or a custom HTTP endpoint.
- Keeps failed processing jobs retryable after provider failures.

## Install

Requires Python 3.11 or newer.

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

Install benchmark dependencies only when needed:

```powershell
python -m pip install -e ".[benchmark]"
```

## Quick start

The fake provider is deterministic and does not make network requests. It is the fastest way to try the engine.

```python
from src import OpenMem
from src.memory.provider import FakeExtractionProvider

db = OpenMem("memory.sqlite", extraction_provider=FakeExtractionProvider())

db.ingest({
    "namespace_id": "demo",
    "idempotency_key": "event-1",
    "type": "decision",
    "payload": {"text": "Decision: use SQLite for local storage."},
})

for memory in db.search("demo", "database choice", limit=5):
    print(memory.statement)

db.close()
```

Each event must have a namespace and an idempotency key. Repeating the same idempotency key in a namespace is safe and returns the original event receipt.

## Production extraction

OpenRouter extraction is explicit. Set a model and API key, then pass the provider to `OpenMem`:

```powershell
$env:OPENROUTER_API_KEY = "your-key"
$env:OPENMEM_EXTRACTION_MODEL = "your-provider/your-model"
```

```python
import os
from src import OpenMem
from src.memory.provider import OpenRouterExtractionProvider

db = OpenMem(
    "memory.sqlite",
    extraction_provider=OpenRouterExtractionProvider(
        model=os.environ["OPENMEM_EXTRACTION_MODEL"],
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
)
```

If a provider call fails, the event remains stored and a retryable job is kept. Retry pending work with `db.process("demo")`.

## Documentation

- [Getting started](docs/getting-started.md) — install, ingest, search, and configure a provider.
- [Architecture](docs/architecture.md) — storage, extraction, evidence, retrieval, and recovery.
- [API reference](docs/api-reference.md) — public `OpenMem` methods and event models.
- [Configuration reference](docs/configuration.md) — environment variables and defaults.
- [Benchmarking](docs/benchmarking.md) — run the LongMemEval-S and micro benchmarks.
- [Contributing](CONTRIBUTING.md) — local workflow and pull request expectations.
- [Security policy](SECURITY.md) — reporting vulnerabilities and handling secrets.

## LongMemEval-S

The benchmark runner is in `benchmarks/longmemeval`.

```powershell
python -m pip install -e ".[benchmark]"
python benchmarks/longmemeval/run_benchmark.py --mode retrieval --micro --confirm-benchmark
```

See [the benchmark guide](docs/benchmarking.md) for the full dataset workflow and cost notes.

## Project layout

```text
src/
  engine.py              Public OpenMem facade
  models.py              Pydantic input and response models
  memory/                Extraction, processing, reconciliation
  retrieval/             Chunking, embeddings, ranking, context packing
  storage/               SQLite stores, jobs, integrity, vector index
  config/                Provider and retrieval settings
tests/                   Unit, integration, reliability, and security tests
benchmarks/              LongMemEval-S runner and micro dataset tools
docs/                    User and contributor documentation
```

## License

OpenMem is available under the [MIT License](LICENSE).
