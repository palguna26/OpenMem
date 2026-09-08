# Getting started

This guide takes you from an empty directory to a working local OpenMem database.

## Prerequisites

- Python 3.11 or newer
- A virtual environment is recommended
- An API key and model are needed only for network-backed extraction

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Verify the installation:

```powershell
python -m pytest
```

## Create and search a database

```python
from src import OpenMem
from src.memory.provider import FakeExtractionProvider

db = OpenMem("memory.sqlite", extraction_provider=FakeExtractionProvider())
db.ingest({
    "namespace_id": "user-123",
    "idempotency_key": "message-001",
    "type": "preference",
    "payload": {"text": "I prefer compact mechanical keyboards."},
})

matches = db.search("user-123", "keyboard preferences", limit=5)
for match in matches:
    print(match.statement, match.confidence)

db.close()
```

The database file is created at `memory.sqlite`. A namespace is the isolation boundary for events and memories; always use a stable namespace ID for the user, agent, or tenant whose memory you are storing.

## Use a network provider

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

Do not commit keys or `.env` files. See [configuration](configuration.md) for provider options.

## Handle provider failures

Ingestion persists the event and creates a processing job before extraction. If extraction fails, call:

```python
response = db.process("user-123")
print(response.processed, response.failed, response.dead_lettered)
```

Use `db.jobs("user-123")` to inspect job state. A job can be retried when the failure is retryable.

## Next steps

- Read the [architecture](architecture.md) to understand the event-to-memory pipeline.
- Use the [API reference](api-reference.md) for lifecycle and export operations.
- Tune providers and retrieval with the [configuration reference](configuration.md).
