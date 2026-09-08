# Contributing to OpenMem

Thanks for helping improve OpenMem. Small, focused pull requests are easiest to review.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the checks before opening a pull request:

```powershell
python -m pytest
python -m ruff check src tests
python -m mypy src
```

## Pull requests

- Explain the user or maintainer problem being solved.
- Add or update tests for behavior changes.
- Update the README or `docs/` when public behavior changes.
- Keep provider keys, database files, benchmark outputs, and private data out of commits.
- Call out schema, storage, API, and benchmark changes clearly.

## Code style

Use type hints and keep public behavior explicit. Prefer a small change that preserves existing data and APIs. Do not change the SQLite schema or memory lifecycle rules without tests that cover migration, restart, and failure behavior.

## Commit messages

Use a short imperative subject, for example:

```text
docs: explain retryable processing jobs
fix: preserve evidence during memory updates
test: cover namespace export round trips
```

## Reporting bugs

Include Python version, operating system, OpenMem commit, a minimal reproduction, and the exact error. Remove API keys and personal data before sharing logs or database exports.
