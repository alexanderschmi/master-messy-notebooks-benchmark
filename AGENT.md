# AGENT.md

## Project Summary

This repository benchmarks how well LLM systems convert messy Jupyter notebooks into reusable Python code.

For each notebook under `data/nb1` through `data/nb10`, a runner is expected to produce:

- `train.py`
- `inference.py`
- `requirements.txt`

The benchmark then scores those outputs for:

- dependency overlap with the notebook reference environment
- Python syntax validity
- training execution success
- artifact reproduction
- generated inference execution
- repo-owned inference execution over produced artifacts
- output agreement between generated and repo-owned inference

The repository is organized around four connected phases:

1. notebook parsing and prompt preparation
2. code generation through a selected runner
3. scoring of generated runs
4. aggregation and visualization for reporting

## Primary Entry Points

### `benchmark.py`

Primary orchestration script.

It:

- loads model definitions from `configs/config.yml`
- loads prompt text from `configs/prompt.yml`
- parses notebooks from `data/`
- injects the selected prompt into `src.core.dspy_config`
- selects a runner: `simple`, `cot`, or `agentic`
- executes generation across notebooks and models using a `ProcessPoolExecutor`
- optionally runs scoring through `src.scoring.pipeline.score_pipeline`

Important implemented behavior:

- generation is skipped for any run directory that already exists
- scoring can be run independently with `--score-only`
- scoring is filtered by notebook, model, runner, complexity, and notebook order when those CLI flags are set
- multiprocessing logging is configured through `src.core.logger`
- notebook order can be `original` or deterministic `adjacent-swap`

### `migrate_output_layout.py`

Migration utility for older output layouts.

Use this when older benchmark outputs need to be moved into the current directory layout expected by `src.core.output_layout` and the scoring pipeline.

### `aggregate_metrics.py`

Aggregates `metrics.json` files across benchmark runs.

Current grouping is by:

- model
- runner
- notebook order

It computes mean time and token usage and writes `output/metrics_report_aggregated.csv` unless overridden.

### `visualize.py`

Builds thesis-style figures from `output/scoring_report.csv` and saves them into `output/figures/`.

This is a reporting layer over the scoring outputs, not part of generation or scoring itself.

## Control Path To Check First

When debugging or extending the benchmark, inspect code in this order:

1. `benchmark.py`
2. `src/core/output_layout.py`
3. the selected runner in `src/runners/`
4. the generated run directory under `output/`
5. `src/scoring/pipeline.py`
6. `src/scoring/utils.py`
7. the matching notebook strategy in `src/inference/notebooks.py`

That path mirrors the actual control flow used by generation and scoring.

## Notebook Parsing

### `src/core/notebook_parser.py`

This module owns notebook loading and text flattening.

It:

- recursively finds `.ipynb` files under `data/`
- renders each notebook into a flat text stream
- preserves cell boundaries using `--- CODE CELL ---` and `--- MARKDOWN CELL ---`
- supports deterministic adjacent swaps via `order_mode="adjacent-swap"`

Important detail:

- the adjacent swap pattern is deterministic per notebook and cell-pair index
- `run_id` is currently accepted by the API but does not affect the swap pattern

If a generation issue depends on notebook content order, this file is the first place to inspect.

## Prompting And Model IO

### `src/core/dspy_config.py`

Defines the DSPy signature used for code generation.

The benchmark expects three output fields:

- `train`
- `inference`
- `requirements`

`benchmark.py` loads the selected prompt from `configs/prompt.yml` and injects it at runtime.

### `src/core/file_io.py`

Owns model-response cleanup and persistence.

It:

- sanitizes raw model output
- extracts file contents from DSPy fields or fallback text output
- writes `train.py`, `inference.py`, and `requirements.txt`
- writes `history.json` when history is available and saving is enabled
- writes `metrics.json` when runtime and usage data are available
- copies `data/<nb>/input` into the generated run directory

This module is the key place to inspect when models return malformed JSON, fenced code blocks, or incomplete field content.

## Output Layout

### `src/core/output_layout.py`

This file defines the canonical run directory layout.

Current canonical path:

```text
output/
  <nb>/
    <runner>/
      <complexity>/
        <notebook_order>/
          <run>/
            <model>/
              train.py
              inference.py
              requirements.txt
              metrics.json
              history.json
              input/
```

Important details:

- `build_run_dir()` always includes `notebook_order`
- `parse_run_dir()` still accepts legacy paths without `notebook_order` and treats them as `original`
- `iter_run_dirs()` supports both the current and legacy layouts so scoring and aggregation remain backward-compatible

Do not change this layout casually. The scoring and reporting pipeline keys on this metadata.

## Runner System

### `src/runners/__init__.py`

Runner registration currently includes:

- `simple`
- `cot`
- `agentic`

All runners implement `BaseRunner.run(...)` from `src/runners/base.py`.

### `simple` runner

Implemented by `src/runners/llm_runner.py::LLMRunner`.

- uses `dspy.Predict`
- performs a single structured generation pass
- writes outputs through `generate_files_from_answer()`

### `cot` runner

Implemented by `src/runners/llm_runner.py::CoTRunner`.

- uses `dspy.ChainOfThought`
- shares the same persistence path as `simple`

### `agentic` runner

Implemented by `src/runners/openhands_runner.py::OpenHandsRunner`.

- uses the OpenHands SDK with terminal, file editor, and task tracker tools
- creates a temporary workspace
- writes notebook content into that workspace as `notebook.ipynb` and `notebook.md`
- instructs the agent to create `train.py`, `inference.py`, and `requirements.txt`
- falls back to extracting code blocks from conversation history if files are missing on disk

### Structured-output fallback

`src/runners/llm_runner.py` also contains a text-response fallback path for OpenAI-compatible endpoints that reject the default DSPy structured response format.

If generation fails with response-format errors, inspect this fallback before changing prompts or scoring.

## Scoring System

### `src/scoring/pipeline.py`

This is the central scoring pipeline.

For each discovered model run directory, it:

1. parses notebook ID, runner, model, complexity, notebook order, and run number from the output path
2. compares generated requirements against `data/<nb>/requirements.txt`
3. checks syntax and complexity for `train.py` and `inference.py`
4. executes `train.py` in a subprocess from the generated run directory
5. compares produced artifacts against `data/<nb>/output`
6. runs inference checks only if training succeeded
7. appends new rows to `output/scoring_report.csv`
8. regenerates `output/scoring_report_aggregated.csv`

Important implemented behavior:

- exact runs are de-duplicated by the key `(notebook_id, runner, model, complexity, notebook_order, run)`
- inference scoring is skipped when training failed
- aggregated reporting groups by `(model, runner, complexity, notebook_order)`
- legacy rows without `notebook_order` are normalized to `original` when loaded

### `src/scoring/evaluator.py`

Measures maintainability and cyclomatic complexity using `radon` and maps that into the benchmark's numeric complexity score.

### `src/scoring/utils.py`

Contains the low-level helpers used by the pipeline, including:

- syntax parsing
- subprocess execution of `train.py`
- requirement comparison
- artifact comparison
- repo-owned inference evaluation
- dynamic import and execution of generated `inference.py`

If scoring fails for only one notebook or one artifact family, inspect this file together with the matching notebook strategy.

## Inference Harness

### `src/inference/notebooks.py`

Maps each benchmark notebook to its notebook-specific inference strategy:

- `InferenceNb1` through `InferenceNb10`

### `src/inference/strategies.py`

Provides reusable strategy implementations for different artifact families, including transformer-backed and classic-model patterns.

If scoring succeeds on training but fails on inference, inspect the matching `InferenceNbX` implementation first.

## Config And Data

### `configs/config.yml`

Defines:

- models and providers
- API keys via environment variables
- optional API base URLs
- global settings such as temperature and default history saving

### `configs/prompt.yml`

Maps prompt complexity levels to prompt text consumed by `benchmark.py`.

### `configs/inference.yml`

Present in the repository, but the current scoring path is driven primarily by `src/inference/notebooks.py`.

### `data/`

Contains benchmark fixtures and ground truth.

Each notebook directory generally contains:

- the source notebook
- a notebook-specific `requirements.txt`
- input data under `input/`
- reference output artifacts under `output/` when artifact comparison is expected

Do not change `data/` casually. Scoring assumes it is the benchmark ground truth.

## Typical Commands

Run generation with default settings:

```bash
python benchmark.py
```

Run a single model and notebook:

```bash
python benchmark.py --model gemini-2.5-flash --notebook nb1
```

Use a different runner and prompt complexity:

```bash
python benchmark.py --runner cot --complexity 5
```

Run multiple repetitions:

```bash
python benchmark.py --runs 3
```

Use deterministic adjacent cell swaps:

```bash
python benchmark.py --runs 3 --notebook-order adjacent-swap
```

Generate and then score:

```bash
python benchmark.py --score
```

Score existing outputs only:

```bash
python benchmark.py --score-only
```

Preview output-layout migration:

```bash
python migrate_output_layout.py --dry-run
```

Aggregate metrics:

```bash
python aggregate_metrics.py
```

Generate figures:

```bash
python visualize.py --format png
```

## Development Guidance For Agents

When making changes:

- prefer small, local edits over broad refactors
- preserve the generated filenames `train.py`, `inference.py`, and `requirements.txt`
- preserve the canonical output layout `output/<nb>/<runner>/<complexity>/<notebook_order>/<run>/<model>/`
- keep scoring behavior backward-compatible with legacy output layouts when possible
- treat `data/` as benchmark ground truth, not a scratch area
- fix bugs at the controlling layer first: parser, runner, file extraction, output layout, scoring, then reporting

Safe extension points:

- add a new runner by implementing `BaseRunner` and registering it in `src/runners/__init__.py`
- add new prompt variants through `configs/prompt.yml`
- adjust file extraction behavior in `src/core/file_io.py`
- extend scoring in `src/scoring/pipeline.py` and `src/scoring/utils.py`
- add or refine notebook-specific inference logic in `src/inference/notebooks.py`

Common failure surfaces:

- malformed or partially structured model responses
- endpoint incompatibilities around JSON response formatting
- output-path mismatches that prevent scoring discovery
- missing copied `input/` data in generated run directories
- notebook-specific model loading or prompt formatting in inference strategies
- generated scripts assuming the wrong working directory at runtime

## Runtime Environment

- Python is pinned to `>=3.12,<3.13` in `pyproject.toml`
- dependency management is defined through Poetry metadata in `pyproject.toml`
- the repository uses DSPy, LiteLLM-compatible providers, OpenHands, pandas, matplotlib, seaborn, radon, transformers, and notebook-specific ML dependencies

## Practical Rule Of Thumb

If a task is about generation quality, start with:

1. `benchmark.py`
2. the selected runner
3. `src/core/file_io.py`
4. the generated run directory

If a task is about scoring correctness, start with:

1. `src/scoring/pipeline.py`
2. `src/scoring/utils.py`
3. `src/inference/notebooks.py`
4. the matching `data/<nb>/output`

If a task is about reports or summaries, start with:

1. `output/scoring_report.csv`
2. `output/scoring_report_aggregated.csv`
3. `aggregate_metrics.py`
4. `visualize.py`