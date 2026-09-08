# Benchmarking

OpenMem includes a LongMemEval-S runner under `benchmarks/longmemeval`.

## Install benchmark dependencies

```powershell
python -m pip install -e ".[benchmark]"
```

## Run the micro benchmark

The micro set has 30 stratified samples and is the fastest local check.

```powershell
python benchmarks/longmemeval/run_benchmark.py --mode retrieval --micro --confirm-benchmark
```

Retrieval mode does not call an LLM. End-to-end mode uses extraction and embedding providers and can incur network and model costs:

```powershell
python benchmarks/longmemeval/run_benchmark.py --mode end-to-end --micro --confirm-benchmark
```

## Run the full dataset

```powershell
python benchmarks/longmemeval/run_benchmark.py --mode end-to-end --confirm-benchmark
```

The runner reads the dataset from `benchmarks/longmemeval`. Large dataset files and generated run artifacts are ignored by Git; do not add them to commits.

## Regenerate the micro set

```powershell
python benchmarks/longmemeval/create_micro.py
```

The subset uses a deterministic seed of 42 and excludes abstention rows. See `python benchmarks/longmemeval/run_benchmark.py --help` for all modes and output options.

## Reproducibility

Record the commit, Python version, benchmark mode, dataset path, provider model, embedding model, and relevant `OPENMEM_*` variables with each result. Temporal queries accept an explicit reference date so results do not depend on the current machine clock.
