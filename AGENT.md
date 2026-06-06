# AGENT.md

## Project Summary

This repository benchmarks how well LLM systems convert messy Jupyter notebooks into clean, reusable Python code.

For each notebook under `data/nb1` through `data/nb10`, a runner generates:

- `train.py`
- `inference.py`
- `requirements.txt`

The benchmark then scores those outputs for syntax validity, execution success, artifact reproduction, inference correctness, and dependency overlap.

The repo is organized around three phases:

1. Notebook loading and prompt-driven code generation
2. Output scoring and report generation
3. Aggregation and visualization for thesis/reporting use

## Main Entry Points

### `benchmark.py`

Primary orchestration script.

- Loads model definitions from `configs/config.yml`
- Loads prompt text from `configs/prompt.yml`
- Parses notebooks from `data/`
- Selects a runner: `simple`, `cot`, or `agentic`
- Executes generation across notebooks and models using a `ProcessPoolExecutor`
- Optionally runs scoring through `src.scoring.pipeline.score_pipeline`

Important behavior:

- Output directories are skipped if they already exist
- Prompt complexity is stored in output folder names
- `--score-only` skips generation and only evaluates existing outputs
- Multiprocessing logging is configured through `src.core.logger`

### `visualize.py`

Builds publication-style figures from `output/scoring_report.csv` and saves them into `output/figures/`.

This is for post-processing benchmark results, not for generation or scoring.

### `aggregate_metrics.py`

Aggregates runtime and token usage from `output/*/*/*/metrics.json` into a single CSV report.

This is useful when comparing runner cost and latency, especially across `simple`, `cot`, and `agentic` runs.

## Core Architecture

### Notebook ingestion

`src/core/notebook_parser.py`

- Recursively finds `.ipynb` files under `data/`
- Converts notebook JSON into a flattened text representation
- Preserves code and markdown cell boundaries using markers like `--- CODE CELL ---`

This flattened notebook text is what the runners send to the model.

### Prompt and DSPy configuration

`src/core/dspy_config.py`

- Defines the default DSPy signature used for code generation
- Expects three output fields: `train`, `inference`, `requirements`
- Allows the active prompt text to be swapped dynamically with `set_prompt()`

Complexity-specific prompts come from `configs/prompt.yml` and are injected at runtime by `benchmark.py`.

### File generation and output persistence

`src/core/file_io.py`

- Cleans model output
- Extracts file contents from DSPy or fallback responses
- Writes generated files into the benchmark output tree
- Saves optional `history.json`
- Saves `metrics.json` with time and usage data when available
- Copies the notebook-specific `data/<nb>/input` directory into the generated run directory

The canonical output layout is:

```text
output/
  nbX/
    <runner>/
      <complexity>/
        <run>/
          <model>/
            train.py
            inference.py
            requirements.txt
            metrics.json
            history.json
            input/
```

## Runner System

The runner registry lives in `src/runners/__init__.py`.

All runners implement `BaseRunner.run(...)` from `src/runners/base.py`.

### `simple` runner

Implemented by `src/runners/llm_runner.py::LLMRunner`.

- Uses `dspy.Predict`
- Single-pass structured generation
- Writes outputs through `generate_files_from_answer()`

### `cot` runner

Implemented by `src/runners/llm_runner.py::CoTRunner`.

- Uses `dspy.ChainOfThought`
- Same persistence path as the simple runner

### `agentic` runner

Implemented by `src/runners/openhands_runner.py::OpenHandsRunner`.

- Uses the OpenHands SDK
- Creates a temporary local workspace
- Writes the notebook content into that workspace
- Instructs the agent to create `train.py`, `inference.py`, and `requirements.txt`
- Falls back to extracting code from conversation history if files are missing on disk

When changing or adding runners, keep the output folder convention stable because the scoring pipeline depends on it.

## Scoring System

### Main pipeline

`src/scoring/pipeline.py`

This is the central evaluator for generated runs.

For each generated run directory, it:

1. Parses notebook ID, runner, model, complexity, and run number from the output path
2. Compares generated requirements against the notebook's reference `requirements.txt`
3. Checks syntax and complexity for `train.py` and `inference.py`
4. Executes `train.py` in a subprocess from the run directory
5. Compares produced artifacts with the notebook's reference output artifacts
6. Runs two inference checks when training succeeded:
   - benchmark-owned inference over the generated artifacts
   - generated `inference.py` imported dynamically and called via `inference(prompt)`
7. Writes detailed and aggregated CSV reports

### Low-level scoring helpers

`src/scoring/utils.py`

- `check_syntax()` parses Python with `ast`
- `run_dynamic_analysis()` runs `train.py` via `subprocess.run()` with a 10-minute timeout
- `compare_requirements()` scores dependency overlap
- `get_inference_strategy()` resolves the per-notebook inference harness
- `test_generated_inference()` imports generated `inference.py` dynamically and runs it
- `test_own_inference()` validates generated artifacts with repo-owned inference logic

### Complexity scoring

`src/scoring/evaluator.py`

- Uses `radon` maintainability and cyclomatic complexity metrics
- Maps the result to an integer score from 1 to 5

## Inference Test Harness

The repo contains notebook-specific reference inference logic in `src/inference/`.

### Generic strategies

`src/inference/strategies.py`

Defines reusable base strategies for artifact-backed inference, including:

- transformer-based models
- CatBoost models
- sentence-transformer models
- custom helper classes used by some notebooks

### Notebook-specific strategies

`src/inference/notebooks.py`

Maps each benchmark notebook to a concrete strategy class:

- `InferenceNb1` through `InferenceNb10`

Some notebooks use the base strategy directly; others override prompt construction or model loading behavior.

If scoring for a notebook is failing, inspect its matching `InferenceNbX` implementation first.

## Config and Data

### `configs/config.yml`

Defines:

- model/provider pairs
- API keys via environment variables
- optional API base URLs
- global settings like temperature and save-history

### `configs/prompt.yml`

Maps complexity levels to prompt text. `benchmark.py` reads the selected complexity and injects that prompt into the runner input.

### `configs/inference.yml`

Present in the repo, but the current scoring path primarily resolves inference behavior from `src/inference/notebooks.py`.

### `data/`

Contains benchmark fixtures.

Each notebook directory generally contains:

- the source notebook
- a notebook-specific `requirements.txt`
- input data under `input/`
- sometimes reference output artifacts under `output/`

Do not change benchmark data casually. Scoring logic assumes these directories are the ground truth.

## Typical Commands

Run generation for all configured models and notebooks:

```bash
python benchmark.py
```

Run a single model and notebook:

```bash
python benchmark.py --model gemini-2.5-flash --notebook nb1
```

Score existing outputs only:

```bash
python benchmark.py --score-only
```

Generate then score:

```bash
python benchmark.py --score
```

Generate thesis figures:

```bash
python visualize.py --format png
```

Aggregate timing and token metrics:

```bash
python aggregate_metrics.py
```

## Development Notes For Agents

### When making changes

- Prefer small, local edits over broad refactors
- Preserve output folder naming: `<complexity>/<run>/<model>` under each `output/<nb>/<runner>/`
- Preserve the expected generated filenames: `train.py`, `inference.py`, `requirements.txt`
- Keep scoring behavior backward-compatible with existing `output/` contents when possible
- Treat `data/` as benchmark ground truth, not a casual workspace

### Safe extension points

- Add a new runner by implementing `BaseRunner` and registering it in `src/runners/__init__.py`
- Add a new benchmark notebook by placing it under `data/nbX` and creating a matching `InferenceNbX` strategy if inference scoring needs custom logic
- Adjust evaluation logic in `src/scoring/pipeline.py` and `src/scoring/utils.py`
- Adjust prompt behavior through `configs/prompt.yml` or `src/core/dspy_config.py`

### Places where bugs are likely to surface

- Parsing model output into valid `train.py` / `inference.py` / `requirements.txt`
- Endpoint incompatibilities around structured JSON responses in DSPy and LiteLLM
- Output directory naming mismatches that prevent the scoring pipeline from discovering runs
- Notebook-specific artifact loading in `src/inference/notebooks.py`
- Relative-path assumptions inside generated scripts executed from the run directory

## Runtime Environment

- Python version is pinned to `>=3.12,<3.13` in `pyproject.toml`
- Dependencies are managed in `pyproject.toml` via Poetry metadata
- The project uses DSPy, LiteLLM-compatible providers, OpenHands, pandas, seaborn, matplotlib, radon, transformers, and several ML libraries used by the benchmark notebooks

## Practical Orientation

If you need to understand a failure quickly, inspect in this order:

1. `benchmark.py` for orchestration and selected arguments
2. the chosen runner implementation in `src/runners/`
3. the generated run directory under `output/nbX/<runner>/...`
4. `src/scoring/pipeline.py` and `src/scoring/utils.py`
5. the matching `InferenceNbX` strategy in `src/inference/notebooks.py`

That path mirrors the actual control flow of the repository.