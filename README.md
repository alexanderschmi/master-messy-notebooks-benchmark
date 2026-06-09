# Messy Notebooks Benchmark

A benchmark that evaluates LLMs on their ability to convert messy Jupyter notebooks into clean, production-ready Python code. Given a notebook, models must generate a `train.py`, `inference.py`, and `requirements.txt`. The generated code is then scored on whether training executes successfully and whether inference runs without error.

---

## How It Works

1. **Load** — Jupyter notebooks are read from `data/nb1` … `data/nb10`
2. **Generate** — Each notebook is sent to one or more LLMs with a complexity-stratified prompt; the model returns `train.py`, `inference.py`, and `requirements.txt`
3. **Execute** — Up to 5 notebooks are processed in parallel via `ProcessPoolExecutor`
4. **Score** — Generated code is statically and dynamically evaluated; results are written to `output/scoring_report.csv`

---

## Installation

1. Install PyTorch for your OS and hardware from https://pytorch.org/get-started/locally/ — for example:

```bash
# CPU only
pip install torch==2.9.1

# CUDA 12.1
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu121
```

**Running the scoring pipeline and some of the notebooks needs CUDA enabled.**

2. Install remaining dependencies with Poetry:

```bash
poetry install
```

Or for the agentic runner (OpenHands):

```bash
poetry install --with agentic
```

3. Set environment variables for your API keys (referenced in `configs/config.yml`):

```bash
export GEMINI_API_KEY="your-key"
export CHATAI_API_KEY="your-key"   # For OpenAI-compatible endpoints (Academic Cloud, etc.)
```

The benchmark currently tests against models available through the Gemini API and the GWDG Academic Cloud OpenAI-compatible endpoint.

---

## Usage

```bash
# Run all models against all notebooks (complexity 4, simple runner)
python benchmark.py

# Specific model and notebook
python benchmark.py --model gemini-2.5-flash --notebook nb1

# Change prompt complexity and runner
python benchmark.py --complexity 5 --runner cot

# Multiple runs per model/notebook pair
python benchmark.py --runs 3

# Deterministically swap some adjacent notebook cells with the same pattern across runs
python benchmark.py --runs 3 --notebook-order adjacent-swap

# Generate and then score
python benchmark.py --score

# Score previously generated outputs only
python benchmark.py --score-only

# Save LLM prompt/response history to history.json
python benchmark.py --save-history
<<<<<<< Updated upstream
=======

# Preview migration from legacy output folders to the new layout
python migrate_output_layout.py --dry-run

# Apply the migration
python migrate_output_layout.py

# Aggregate timing and token usage metrics
python aggregate_metrics.py

# Evaluate real ML performance of generated model artifacts
python evaluate_model_performance.py

# Compare generated-artifact inference outputs against real notebook artifacts
python compare_own_inference_vs_real.py

# Retry failed inference-scoring rows
python rerun_failed_inference.py

# Generate thesis-quality figures from scoring data
python visualize.py --format png
>>>>>>> Stashed changes
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--runs` | `1` | Number of repetitions per model/notebook pair |
| `--complexity` | `4` | Prompt complexity level (1–5) |
| `--notebook-order` | `original` | Notebook load mode: `original` or deterministic `adjacent-swap` fixed across runs |
| `--config` | `configs/config.yml` | Path to models config file |
| `--model` | all models | Filter to a single model by name |
| `--notebook` | all notebooks | Filter to a single notebook (e.g. `nb1`) |
| `--runner` | `simple` | Runner strategy: `simple`, `cot`, or `agentic` |
| `--score` | off | Run scoring pipeline after generation |
| `--score-only` | off | Skip generation, run scoring only |
| `--save-history` | off | Save LLM prompt/response history as `history.json` |

<<<<<<< Updated upstream
=======
### Adjacent-Swap Notebook Order

The `--notebook-order adjacent-swap` flag deterministically reorders notebook cells before sending them to the LLM. This tests whether models can reconstruct correct code even when logically dependent cells appear out of order.

**How it works:**

- Cells are walked in pairs. For each pair at index `i`, the benchmark computes `SHA256("{nb_id}:{i}")` and swaps the pair if the first byte mod 3 equals 0 (~33% of pairs).
- Swapped pairs are skipped (index advances by 2), so a cell is involved in at most one swap.
- The swap pattern is deterministic per notebook — the same cells are swapped every run (the `run_id` is not part of the seed).

**Concrete swap counts per notebook:**

| Notebook | Total Cells | Swapped Pairs | Swapped Cell Indices |
|----------|-------------|---------------|----------------------|
| nb1 | 17 | 3 | (7,8), (10,11), (12,13) |
| nb2 | 32 | 9 | (1,2), (5,6), (7,8), (10,11), (12,13), (17,18), (19,20), (24,25), (29,30) |
| nb3 | 30 | 7 | (8,9), (10,11), (13,14), (15,16), (20,21), (26,27), (28,29) |
| nb4 | 9 | 3 | (0,1), (4,5), (6,7) |
| nb5 | 9 | 2 | (0,1), (4,5) |
| nb6 | 18 | 5 | (5,6), (9,10), (11,12), (13,14), (16,17) |
| nb7 | 23 | 7 | (1,2), (3,4), (5,6), (11,12), (14,15), (18,19), (20,21) |
| nb8 | 22 | 5 | (1,2), (4,5), (7,8), (11,12), (19,20) |
| nb9 | 25 | 4 | (1,2), (3,4), (5,6), (10,11) |
| nb10 | 43 | 7 | (1,2), (4,5), (12,13), (19,20), (27,28), (31,32), (34,35) |

The implementation lives in [`src/core/notebook_parser.py`](src/core/notebook_parser.py).

---

### Migrating Existing Outputs

If you already have benchmark runs stored in the legacy layout
`output/<nb>/<runner>/<model>_complexity_<N>_run_<R>`, migrate them to the
current layout `output/<nb>/<runner>/<complexity>/<notebook_order>/<run>/<model>` with:

```bash
# inspect planned moves
python migrate_output_layout.py --dry-run

# perform the move
python migrate_output_layout.py
```

By default, the migration stops on conflicts. You can also use
`--on-conflict skip` or `--on-conflict overwrite`.

>>>>>>> Stashed changes
---

## Runners

Runners control how the LLM generates code. All runners share a common `BaseRunner` interface and are registered in `src/runners/__init__.py`.

| Runner | Strategy | Description |
|---|---|---|
| `simple` | `dspy.Predict` | Single-pass LLM call, no reasoning chain |
| `cot` | `dspy.ChainOfThought` | Encourages step-by-step reasoning before generating code |
| `agentic` | OpenHands SDK | Autonomous agent with file editing and terminal access; writes files directly to a temporary workspace |

To add a custom runner, create a class extending `BaseRunner` and register it:

```python
from src.runners import register_runner
from src.runners.base import BaseRunner

class MyRunner(BaseRunner):
    RUNNER_NAME = "my-runner"

    def run(self, params, notebook, temperature, save_history, output_dir, complexity, run):
        ...

register_runner(MyRunner.RUNNER_NAME, MyRunner)
```

---

## Configuration

### `configs/config.yml`

Defines models to test and global settings:

```yaml
settings:
  temperature: 0.1
  save_history: false

models:
  - provider: gemini
    model: gemini-2.5-flash
    api_key: ${GEMINI_API_KEY}
    api_base: "https://generativelanguage.googleapis.com/v1beta"
  - provider: openai
    model: devstral-2-123b-instruct-2512
    api_key: ${CHATAI_API_KEY}
    api_base: "https://chat-ai.academiccloud.de/v1"
```

Models use the OpenAI-compatible provider format. Set `api_base` to point at any compatible endpoint (OpenRouter, Academic Cloud, local LM Studio, etc.).

### `configs/prompt.yml`

Maps complexity levels to prompt text. Level 1 is minimal; level 5 includes role context, documentation expectations, and explicit justification requirements.

---

## Scoring

Scoring is run with `--score` or `--score-only` and produces two CSV files in `output/`:

**`scoring_report.csv`** — one row per model / notebook / runner / complexity / run:

| Column | Description |
|---|---|
| `train_syntax_valid` | Whether `train.py` parses without AST errors |
| `train_complexity_score` | Radon maintainability index mapped to 1–5 |
| `train_execution_success` | Whether `train.py` runs successfully against the original data |
| `artifact_match_score` | How closely generated model artifacts match the originals |
| `inference_syntax_valid` | Whether `inference.py` parses without AST errors |
| `inference_complexity_score` | Radon maintainability index mapped to 1–5 |
| `own_inference_success` | Whether `inference.py` runs without error |
| `llm_inference_success` | Whether the LLM-generated inference function executes |
| `outputs_match` | Whether inference outputs match expected results |
| `requirements_match_score` | Overlap between generated and original requirements |

**`scoring_report_aggregated.csv`** — mean `train_execution_success` and `own_inference_success` per model / runner / complexity / notebook_order, sorted by those two columns.

### Additional Reports

| Script | Output File(s) | Purpose |
|--------|---------------|----------|
| `aggregate_metrics.py` | `metrics_report_aggregated.csv` | Mean time and token usage per model/runner/order |
| `evaluate_model_performance.py` | `model_performance.csv`, `model_performance_summary.csv` | Real ML metrics (accuracy, F1, RMSE) of generated artifacts |
| `compare_own_inference_vs_real.py` | `own_inference_vs_real_artifacts.csv`, `own_inference_vs_real_artifacts_summary.csv` | Continuous similarity (0–1) between generated and reference inference outputs |
| `visualize.py` | `output/figures/*.png` | Publication-quality charts for thesis figures |

---

## Output Structure

```
output/
├── nb1/
│   ├── simple/
│   │   └── 4/
│   │       ├── original/
│   │       │   └── 1/
│   │       │       └── gemini-2.5-flash/
│   │       │           ├── train.py
│   │       │           ├── inference.py
│   │       │           ├── requirements.txt
│   │       │           ├── metrics.json       # timing and token usage
│   │       │           ├── history.json       # only with --save-history
│   │       │           └── input/             # copy of the notebook's original dataset
│   │       └── adjacent-swap/
│   │           └── 1/
│   │               └── gemini-2.5-flash/
│   │                   └── …
│   ├── cot/
│   └── agentic/
├── nb2/ … nb10/
├── scoring_report.csv
├── scoring_report_aggregated.csv
├── metrics_report_aggregated.csv
├── model_performance.csv
├── model_performance_summary.csv
├── own_inference_vs_real_artifacts.csv
├── own_inference_vs_real_artifacts_summary.csv
└── figures/                                    # generated by visualize.py
```

Layout path: `output/<nb>/<runner>/<complexity>/<notebook_order>/<run>/<model>/`

---

## Analysis Scripts

Post-processing scripts for reporting and visualization live in `scripts/`:

| Script | Purpose |
|---|---|
| `scripts/visualize.py` | Builds publication-style figures from scoring results |
| `scripts/aggregate_metrics.py` | Aggregates runtime and token metrics into a single CSV |
| `scripts/evaluate_model_performance.py` | Evaluates real ML model performance of generated code |
| `scripts/compare_own_inference_vs_real.py` | Compares generated inference outputs against reference artifacts |

---

## Project Structure

```
<<<<<<< Updated upstream
benchmark.py              # Entry point and orchestration
setup.py                  # Notebook pre-execution (generates reference outputs)
configs/
  config.yml              # Models and settings
  prompt.yml              # Complexity-stratified prompts
  inference.yml           # Per-notebook inference hints
data/
  nb1/ … nb10/            # Notebooks and input datasets
src/
  core/                   # Notebook parsing, DSPy config, file I/O, logging
  runners/                # BaseRunner, LLMRunner, CoTRunner, OpenHandsRunner
  scoring/                # Evaluator, pipeline, utilities
  inference/              # Inference testing helpers
scripts/                  # Post-processing analysis and visualization
output/                   # Generated code and scoring results (git-ignored)
=======
benchmark.py                    # Entry point and orchestration
aggregate_metrics.py            # Aggregate time/token metrics from metrics.json files
evaluate_model_performance.py   # Run inference on generated artifacts and compute ML metrics
compare_own_inference_vs_real.py # Compare generated-artifact inference vs real notebook artifacts
rerun_failed_inference.py       # Retry failed inference-scoring rows
visualize.py                    # Generate publication-quality figures from scoring reports
migrate_output_layout.py        # Migrate legacy output folders to current layout
configs/
  config.yml                    # Models and settings
  prompt.yml                    # Complexity-stratified prompts
  inference.yml                 # Per-notebook inference hints
data/
  nb1/ … nb10/                  # Notebooks, input datasets, and reference artifacts
src/
  core/                         # Notebook parsing, DSPy config, file I/O, output layout, logging
  runners/                      # BaseRunner, LLMRunner, CoTRunner, OpenHandsRunner
  scoring/                      # Evaluator, pipeline, utilities
  inference/                    # Inference testing helpers and notebook-specific strategies
output/                         # Generated code, scoring results, and figures
>>>>>>> Stashed changes
```
