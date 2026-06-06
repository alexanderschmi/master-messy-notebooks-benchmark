# Master Messy Notebooks Benchmark

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
export CHATAI_API_KEY="your-key"   # For OpenAI-compatible endpoints
```

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

# Generate and then score
python benchmark.py --score

# Score previously generated outputs only
python benchmark.py --score-only

# Save LLM prompt/response history to history.json
python benchmark.py --save-history
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--runs` | `1` | Number of repetitions per model/notebook pair |
| `--complexity` | `4` | Prompt complexity level (1–5) |
| `--config` | `configs/config.yml` | Path to models config file |
| `--model` | all models | Filter to a single model by name |
| `--notebook` | all notebooks | Filter to a single notebook (e.g. `nb1`) |
| `--runner` | `simple` | Runner strategy: `simple`, `cot`, or `agentic` |
| `--score` | off | Run scoring pipeline after generation |
| `--score-only` | off | Skip generation, run scoring only |
| `--save-history` | off | Save LLM prompt/response history as `history.json` |

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
  - provider: openai
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}
    api_base: "https://api.openai.com/v1"  # optional
```

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

**`scoring_report_aggregated.csv`** — mean `train_execution_success` and `own_inference_success` per model / runner / complexity, sorted by those two columns.

---

## Output Structure

```
output/
├── nb1/
│   ├── simple/
│   │   └── 4/
│   │       └── 1/
│   │           └── gemini-2.5-flash/
│   │               ├── train.py
│   │               ├── inference.py
│   │               ├── requirements.txt
│   │               ├── history.json          # only with --save-history
│   │               └── input/                # copy of the notebook's original dataset
│   ├── cot/
│   └── agentic/
├── nb2/ … nb10/
├── scoring_report.csv
└── scoring_report_aggregated.csv
```

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
```
