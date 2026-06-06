"""
Visualization script for the master-messy-notebooks-benchmark.

Reads output/scoring_report.csv and generates publication-quality figures
suitable for a master thesis. All figures are saved to output/figures/.

Usage:
    python visualize.py [--output-dir output] [--show] [--format pdf|png|both]
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

THESIS_RC = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

# Colour palette — accessible and print-friendly
PALETTE = sns.color_palette("tab10")

PIPELINE_STAGES = [
    ("train_syntax_valid", "Syntax Valid"),
    ("train_execution_success", "Train Executes"),
    ("own_inference_success", "Own Inference OK"),
    ("llm_inference_success", "LLM Inference OK"),
    ("outputs_match", "Outputs Match"),
]
PIPELINE_STAGE_LABELS = dict(PIPELINE_STAGES)
TRAIN_PIPELINE_STEPS = {"train_syntax_valid", "train_execution_success"}
TRAIN_REQUIRED_STEPS = ["train_syntax_valid", "train_execution_success"]
INFERENCE_REQUIRED_STEPS = [
    "inference_syntax_valid",
    "own_inference_success",
    "llm_inference_success",
]
NOTEBOOK_ORDER_LABELS = {
    "original": "Original",
    "adjacent-swap": "Adjacent Swap",
}


def apply_style():
    matplotlib.rcParams.update(THESIS_RC)
    sns.set_theme(style="whitegrid", rc=THESIS_RC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_name(model: str) -> str:
    """Shorten long model names for axis labels."""
    replacements = {
        "meta-llama-3.1-8b-instruct": "LLaMA-3.1-8B",
        "llama-3.3-70b-instruct": "LLaMA-3.3-70B",
        "llama-3.1-sauerkrautlm-70b-instruct": "SauerkrautLM-70B",
        "mistral-large-3-675b-instruct-2512": "Mistral-Large-675B",
        "deepseek-r1-distill-llama-70b": "DeepSeek-R1-70B",
        "gemini-2.5-flash": "Gemini-2.5-Flash",
        "openai-gpt-oss-120b": "GPT-OSS-120B",
        "qwen3-235b-a22b": "Qwen3-235B",
        "qwen3-32b": "Qwen3-32B",
        "qwen3-30b-a3b-instruct-2507": "Qwen3-30B",
        "qwen3-30b-a3b-thinking-2507": "Qwen3-30B-Think",
        "qwen3-coder-30b-a3b-instruct": "Qwen3-Coder-30B",
        "qwen3-omni-30b-a3b-instruct": "Qwen3-Omni-30B",
        "qwen3-vl-30b-a3b-instruct": "Qwen3-VL-30B",
        "devstral-2-123b-instruct-2512": "Devstral-123B",
        "apertus-70b-instruct-2509": "Apertus-70B",
        "internvl3.5-30b-a3b": "InternVL3.5-30B",
        "medgemma-27b-it": "MedGemma-27B",
        "gemma-3-27b-it": "Gemma-3-27B",
        "gemma-4-26b-a4b-it-mlx": "Gemma-4-26B-A4B",
        "gemma-4-31b-it": "Gemma-4-31B",
        "gemma-4-e2b-it-mlx": "Gemma-4-E2B",
        "gemma-4-e4b-it-mlx": "Gemma-4-E4B",
        "glm-4.7": "GLM-4.7",
    }
    return replacements.get(model, model)


def save_fig(fig, figures_dir: Path, stem: str, fmt: str):
    figures_dir.mkdir(parents=True, exist_ok=True)
    if fmt in ("pdf", "both"):
        path = figures_dir / f"{stem}.pdf"
        fig.savefig(path)
        logger.info(f"Saved {path}")
    if fmt in ("png", "both"):
        path = figures_dir / f"{stem}.png"
        fig.savefig(path)
        logger.info(f"Saved {path}")


def pipeline_step_label(pipeline_step: str) -> str:
    return PIPELINE_STAGE_LABELS.get(pipeline_step, pipeline_step)


def pipeline_step_values(df: pd.DataFrame, pipeline_step: str) -> pd.Series:
    return df[pipeline_step].astype(float) * 100


def pipeline_step_complexity_column(pipeline_step: str) -> str:
    return "train_complexity_score" if pipeline_step in TRAIN_PIPELINE_STEPS else "inference_complexity_score"


def natural_notebook_order(values) -> list[str]:
    return sorted(values, key=lambda value: int("".join(filter(str.isdigit, str(value))) or 0))


def ordered_notebook_orders(values) -> list[str]:
    preferred = ["original", "adjacent-swap"]
    available = list(dict.fromkeys(values))
    ordered = [value for value in preferred if value in available]
    ordered.extend(sorted(value for value in available if value not in preferred))
    return ordered


def notebook_order_label(order: str) -> str:
    return NOTEBOOK_ORDER_LABELS.get(order, order)


def filter_simple_complexity(df: pd.DataFrame, complexity: int = 4) -> pd.DataFrame:
    return df[(df["runner"] == "simple") & (df["complexity"] == complexity) & (df["notebook_order"] == "original")].copy()


def models_with_all_complexities(df: pd.DataFrame) -> list[str]:
    if "complexity" not in df.columns or df["complexity"].nunique() < 2:
        return []

    all_complexities = set(df["complexity"].unique())
    complete_models = (
        df.groupby("model")["complexity"]
        .apply(lambda series: all_complexities.issubset(set(series)))
    )
    return sorted(complete_models[complete_models].index.tolist())


def filter_models_with_all_complexities(df: pd.DataFrame) -> pd.DataFrame:
    complete_models = models_with_all_complexities(df)
    if not complete_models:
        return df.iloc[0:0].copy()
    return df[df["model"].isin(complete_models)].copy()


def ordered_runners(df: pd.DataFrame) -> list[str]:
    return sorted(df["runner"].unique(), key=lambda runner: RUNNER_ORDER.index(runner) if runner in RUNNER_ORDER else 99)


def drop_models_with_all_nan_inference_scores(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns or "inference_score" not in df.columns:
        return df

    valid_models = (
        df.groupby("model")["inference_score"]
        .apply(lambda series: series.notna().any())
    )
    valid_models = valid_models[valid_models].index.tolist()
    if len(valid_models) == df["model"].nunique():
        return df

    dropped_models = sorted(set(df["model"].unique()) - set(valid_models))
    logger.info(f"Dropping {len(dropped_models)} models with all-NaN inference scores: {', '.join(dropped_models)}")
    return df[df["model"].isin(valid_models)].copy()


def prepare_success_rate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Replace train/inference score columns with strict all-steps success rates.

    train_score:      1 iff syntax + execution + artifact_match all pass (× 100)
    inference_score:  P(inference | train succeeded) — NaN when train failed so
                      grouped means give the conditional success rate directly.
    """
    df = df.copy()

    if all(col in df.columns for col in TRAIN_REQUIRED_STEPS) and "artifact_match_score" in df.columns:
        train_success = df[TRAIN_REQUIRED_STEPS].fillna(False).all(axis=1) & df["artifact_match_score"].fillna(0).ge(1.0)
        df["train_score"] = train_success.astype(float) * 100

    if all(col in df.columns for col in INFERENCE_REQUIRED_STEPS):
        inference_success = df[INFERENCE_REQUIRED_STEPS].fillna(False).all(axis=1)
        # Conditional: set to NaN on rows where training failed so that
        # groupby().mean() computes P(inference | train succeeded); 0 otherwise
        cond_mask = df["train_score"] > 0 if "train_score" in df.columns else pd.Series(True, index=df.index)
        df["inference_score"] = (inference_success.astype(float) * 100).where(cond_mask, other=np.nan)

    return df


# ---------------------------------------------------------------------------
# Figure 1 — Model Leaderboard (horizontal bar)
# ---------------------------------------------------------------------------

def plot_leaderboard(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Ranked horizontal bar chart of overall model performance (simple runner only)."""
    df_simple = filter_simple_complexity(df, complexity=4)
    if df_simple.empty:
        logger.warning("Skipping leaderboard — no rows for simple runner / complexity 4.")
        return

    if pipeline_step is not None:
        agg = df_simple.groupby("model")[[pipeline_step]].mean().mul(100).round(1).reset_index()
        agg["plot_value"] = agg[pipeline_step]
        agg = agg.sort_values("plot_value", ascending=True)
        agg["label"] = agg["model"].apply(short_name)

        fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(agg))))
        colors = [PALETTE[0] if v >= agg["plot_value"].median() else PALETTE[1] for v in agg["plot_value"]]
        bars = ax.barh(agg["label"], agg["plot_value"], color=colors, edgecolor="white", height=0.7)

        for bar, val in zip(bars, agg["plot_value"]):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", ha="left", fontsize=8)

        ax.set_xlabel("Success Rate (%)")
        ax.set_title(f"Model Leaderboard — {pipeline_step_label(pipeline_step)}")
        ax.set_xlim(0, 105)
        ax.axvline(agg["plot_value"].median(), color="grey", linestyle=":", linewidth=1, label="Median")
        ax.legend(loc="lower right")
        fig.tight_layout()
        save_fig(fig, figures_dir, f"1_leaderboard_{pipeline_step}", fmt)
        if show:
            plt.show()
        plt.close(fig)
        return

    agg = (
        df_simple.groupby("model")[["inference_score", "train_score", "requirements_match_score"]]
        .mean()
        .round(4)
        .reset_index()
    )
    agg["aggregated_score"] = (agg["train_score"] + agg["inference_score"] + agg["requirements_match_score"] * 100) / 3
    agg = agg.sort_values("aggregated_score", ascending=True)
    agg["label"] = agg["model"].apply(short_name)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(agg))))
    colors = [PALETTE[0] if v >= agg["aggregated_score"].median() else PALETTE[1] for v in agg["aggregated_score"]]
    bars = ax.barh(agg["label"], agg["aggregated_score"], color=colors, edgecolor="white", height=0.7)

    for bar, val in zip(bars, agg["aggregated_score"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="left", fontsize=8)

    ax.set_xlabel("Combined Success Rate (%)")
    ax.set_title("Model Leaderboard")
    ax.set_xlim(0, 105)
    ax.axvline(agg["aggregated_score"].median(), color="grey", linestyle=":", linewidth=1, label="Median")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, figures_dir, "1_leaderboard", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Score Breakdown per Model
# ---------------------------------------------------------------------------

def plot_score_breakdown(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Horizontal grouped bar chart: train/inference success rates and requirements per model."""
    if pipeline_step is not None:
        agg = df.groupby("model")[[pipeline_step]].mean().mul(100).round(1).reset_index()
        agg = agg.sort_values(pipeline_step, ascending=True)
        agg["label"] = agg["model"].apply(short_name)

        fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(agg))))
        ax.barh(agg["label"], agg[pipeline_step], height=0.6, color=PALETTE[0], edgecolor="white")
        ax.set_xlabel("Success Rate (%)")
        ax.set_title(f"{pipeline_step_label(pipeline_step)} Success Rate per Model")
        ax.set_xlim(0, 100)

        fig.tight_layout()
        save_fig(fig, figures_dir, f"2_score_breakdown_{pipeline_step}", fmt)
        if show:
            plt.show()
        plt.close(fig)
        return

    metrics = ["train_score", "inference_score"]
    labels = ["Train Success", "Inference Success"]

    agg = df.groupby("model")[metrics + ["requirements_match_score"]].mean().round(4).reset_index()
    agg["aggregated_score"] = (agg["train_score"] + agg["inference_score"] + agg["requirements_match_score"] * 100) / 3
    agg = agg.sort_values("aggregated_score", ascending=True)
    agg["label"] = agg["model"].apply(short_name)

    y = np.arange(len(agg))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(agg))))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, PALETTE)):
        offset = (i - 0.5) * height
        ax.barh(y + offset, agg[metric], height, label=label, color=color, edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(agg["label"])
    ax.set_xlabel("Rate (%)")
    ax.set_title("Success Breakdown per Model")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_fig(fig, figures_dir, "2_score_breakdown", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Effect of Complexity on Scores (line chart)
# ---------------------------------------------------------------------------

def plot_complexity_effect(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Grouped bar chart with variance: average train and inference success vs complexity level."""
    if "complexity" not in df.columns or df["complexity"].nunique() < 2:
        logger.warning("Skipping complexity plot — not enough complexity levels in data.")
        return

    all_complexities = sorted(df["complexity"].unique())
    complete_models = models_with_all_complexities(df)
    if not complete_models:
        logger.warning("Skipping complexity plot — no model was run on all complexity levels.")
        return
    df_filtered = filter_models_with_all_complexities(df)
    logger.info(f"Complexity effect: {len(complete_models)} models with all {sorted(all_complexities)} complexities")

    agg_mean = (
        df_filtered.groupby("complexity")[["train_score", "inference_score"]]
        .mean()
        .round(2)
    )
    agg_std = (
        df_filtered.groupby("complexity")[["train_score", "inference_score"]]
        .std()
        .fillna(0)
        .round(2)
    )

    complexities = sorted(agg_mean.index.unique())
    x = np.arange(len(complexities))
    width = 0.35

    if pipeline_step is not None:
        success_df = df_filtered.assign(_success=pipeline_step_values(df_filtered, pipeline_step))
        agg_mean = success_df.groupby("complexity")["_success"].mean().round(2)
        agg_std = success_df.groupby("complexity")["_success"].std().fillna(0).round(2)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            x,
            agg_mean.loc[complexities],
            width=0.55,
            label=pipeline_step_label(pipeline_step),
            color=PALETTE[0],
            edgecolor="white",
            yerr=agg_std.loc[complexities],
            capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "#444", "capthick": 1.2},
        )
        ax.set_xlabel("Complexity Level")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(f"Effect of Prompt Complexity on {pipeline_step_label(pipeline_step)}")
        ax.set_xticks(x)
        ax.set_xticklabels(complexities)
        ax.set_ylim(0, 100)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        fig.tight_layout(rect=[0, 0, 0.84, 1])
        save_fig(fig, figures_dir, f"3_complexity_effect_{pipeline_step}", fmt)
        if show:
            plt.show()
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        x - width / 2,
        agg_mean.loc[complexities, "train_score"],
        width,
        label="Train Success",
        color=PALETTE[0],
        edgecolor="white",
        yerr=agg_std.loc[complexities, "train_score"],
        capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#444", "capthick": 1.2},
    )
    ax.bar(
        x + width / 2,
        agg_mean.loc[complexities, "inference_score"],
        width,
        label="Inference Success",
        color=PALETTE[1],
        edgecolor="white",
        yerr=agg_std.loc[complexities, "inference_score"],
        capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#444", "capthick": 1.2},
    )

    ax.set_xlabel("Complexity Level")
    ax.set_ylabel("Average Success Rate (%)")
    ax.set_title("Effect of Prompt Complexity on Pipeline Success")
    ax.set_xticks(x)
    ax.set_xticklabels(complexities)
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    save_fig(fig, figures_dir, "3_complexity_effect", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Complexity Effect per Model (small multiples / facet)
# ---------------------------------------------------------------------------

def plot_complexity_per_model(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Faceted chart: train/inference success rate vs complexity, one panel per model."""
    if "complexity" not in df.columns or df["complexity"].nunique() < 2:
        logger.warning("Skipping per-model complexity plot — not enough complexity levels.")
        return

    complete_models = models_with_all_complexities(df)
    if not complete_models:
        logger.warning("Skipping per-model complexity plot — no model was run on all complexity levels.")
        return
    df = filter_models_with_all_complexities(df)

    models = sorted(df["model"].unique())
    n = len(models)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), sharey=True)
    axes_flat = np.array(axes).flatten()

    complexities = sorted(df["complexity"].unique())

    for ax, model in zip(axes_flat, models):
        sub_df = df[df["model"] == model]
        x = np.arange(len(complexities))
        width = 0.35
        if pipeline_step is not None:
            sub = sub_df.assign(_success=pipeline_step_values(sub_df, pipeline_step)).groupby("complexity")["_success"].mean().reindex(complexities)
            ax.bar(x, sub, width=0.55, label=pipeline_step_label(pipeline_step), color=PALETTE[0], edgecolor="white")
        else:
            sub = sub_df.groupby("complexity")[["train_score", "inference_score"]].mean().reindex(complexities)
            ax.bar(x - width / 2, sub["train_score"], width, label="Train Success", color=PALETTE[0], edgecolor="white")
            ax.bar(x + width / 2, sub["inference_score"], width, label="Inference Success", color=PALETTE[1], edgecolor="white")
        ax.set_title(short_name(model), fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(complexities, fontsize=7)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    if pipeline_step is not None:
        fig.suptitle(f"{pipeline_step_label(pipeline_step)} vs Complexity", fontsize=13)
        save_stem = f"4_complexity_per_model_{pipeline_step}"
    else:
        fig.suptitle("Train & Inference Success vs Complexity", fontsize=13)
        save_stem = "4_complexity_per_model"
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, figures_dir, save_stem, fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 15 — Complexity Score vs Performance
# ---------------------------------------------------------------------------

def plot_complexity_scores_first5(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Two scatter plots comparing train and inference complexity scores against
    train and inference success rates for all models that were evaluated on every
    prompt complexity."""
    if pipeline_step is not None:
        complexity_col = pipeline_step_complexity_column(pipeline_step)
        required_cols = {"complexity", complexity_col, pipeline_step}
        if not required_cols.issubset(df.columns):
            logger.warning("Skipping complexity-score plot — required columns are missing for the selected pipeline step.")
            return
        if df["complexity"].nunique() < 2:
            logger.warning("Skipping complexity-score plot — not enough complexity levels.")
            return

        complete_models = models_with_all_complexities(df)
        if not complete_models:
            logger.warning("Skipping complexity-score plot — no model was run on all complexity levels.")
            return

        complete_df = filter_models_with_all_complexities(df)
        plot_df = (
            complete_df
            .assign(_success=pipeline_step_values(complete_df, pipeline_step))
            .groupby(["model", "complexity"])[[complexity_col, "_success"]]
            .mean()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(6.5, 5))
        model_order = sorted(plot_df["model"].unique())
        colors = {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(model_order)}

        for model in model_order:
            sub_model = plot_df[plot_df["model"] == model].sort_values("complexity")
            ax.scatter(
                sub_model[complexity_col],
                sub_model["_success"],
                s=45,
                alpha=0.8,
                color=colors[model],
                label=short_name(model),
                edgecolors="white",
                linewidths=0.5,
            )

        ax.set_title(pipeline_step_label(pipeline_step))
        ax.set_xlabel("Train Complexity Score" if complexity_col == "train_complexity_score" else "Inference Complexity Score")
        ax.set_ylabel("Success Rate (%)")
        ax.set_xlim(0.8, 100.2)
        ax.set_ylim(0, 100)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.93, 0.5), borderaxespad=0.0)
        fig.suptitle("Complexity Score vs Pipeline-Step Success", fontsize=13)
        fig.tight_layout(rect=[0, 0, 0.90, 1])
        save_fig(fig, figures_dir, f"15_complexity_score_vs_{pipeline_step}", fmt)
        if show:
            plt.show()
        plt.close(fig)
        return

    required_cols = {
        "complexity",
        "train_complexity_score",
        "inference_complexity_score",
        "train_score",
        "inference_score",
    }
    if not required_cols.issubset(df.columns):
        logger.warning("Skipping complexity-score plot — required columns are missing.")
        return
    if df["complexity"].nunique() < 2:
        logger.warning("Skipping complexity-score plot — not enough complexity levels.")
        return

    complete_models = models_with_all_complexities(df)
    if not complete_models:
        logger.warning("Skipping complexity-score plot — no model was run on all complexity levels.")
        return

    sub_df = filter_models_with_all_complexities(df)
    plot_df = (
        sub_df.groupby(["model", "complexity"])[
            [
                "train_complexity_score",
                "inference_complexity_score",
                "train_score",
                "inference_score",
            ]
        ]
        .mean()
        .reset_index()
    )

    fig, (ax_train, ax_inf) = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    model_order = sorted(plot_df["model"].unique())
    colors = {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(model_order)}

    for model in model_order:
        sub_model = plot_df[plot_df["model"] == model].sort_values("complexity")
        ax_train.scatter(
            sub_model["train_complexity_score"],
            sub_model["train_score"],
            s=45,
            alpha=0.8,
            color=colors[model],
            label=short_name(model),
            edgecolors="white",
            linewidths=0.5,
        )
        ax_inf.scatter(
            sub_model["inference_complexity_score"],
            sub_model["inference_score"],
            s=45,
            alpha=0.8,
            color=colors[model],
            label=short_name(model),
            edgecolors="white",
            linewidths=0.5,
        )

    ax_train.set_title("Train File")
    ax_train.set_xlabel("Train Complexity Score")
    ax_train.set_ylabel("Train Success Rate (%)")
    ax_train.set_xlim(0.8, 5.2)
    ax_train.set_xticks([1, 2, 3, 4, 5])
    ax_train.set_ylim(0, 100)

    ax_inf.set_title("Inference File")
    ax_inf.set_xlabel("Inference Complexity Score")
    ax_inf.set_ylabel("Inference Success Rate (%)")
    ax_inf.set_xlim(0.8, 5.2)
    ax_inf.set_xticks([1, 2, 3, 4, 5])
    ax_inf.set_ylim(0, 100)

    handles, labels = ax_train.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.93, 0.5), borderaxespad=0.0)
    fig.suptitle("Complexity Score vs Success Rate (Models with All Prompt Complexities)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.90, 1])
    save_fig(fig, figures_dir, "15_complexity_score_vs_performance", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Pipeline Success Rates
# ---------------------------------------------------------------------------

def plot_success_funnel(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Visualize pipeline-step success per model.

    When ``pipeline_step`` is provided, plot only that step's success rate.
    Otherwise, show the grouped comparison across all available steps.
    """
    valid_stages = [(stage, label) for stage, label in PIPELINE_STAGES if stage in df.columns]
    if not valid_stages:
        logger.warning("Skipping pipeline success plot — no funnel stage columns found.")
        return

    if pipeline_step is not None:
        selected_stage = next((item for item in valid_stages if item[0] == pipeline_step), None)
        if selected_stage is None:
            available_steps = ", ".join(stage for stage, _ in valid_stages)
            logger.warning(
                f"Skipping pipeline success plot — pipeline step '{pipeline_step}' is unavailable. "
                f"Available steps: {available_steps}"
            )
            return

        stage, label = selected_stage
        agg = df.groupby("model")[[stage]].mean().mul(100).round(1).reset_index()
        agg["label"] = agg["model"].apply(short_name)
        agg = agg.sort_values(stage, ascending=True)

        fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(agg))))
        bars = ax.barh(agg["label"], agg[stage], color=PALETTE[0], edgecolor="white", height=0.7)

        for bar, val in zip(bars, agg[stage]):
            ax.text(
                val + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                ha="left",
                fontsize=8,
            )

        ax.set_xlabel("Success Rate (%)")
        ax.set_title(f"Pipeline Step Success: {label}")
        ax.set_xlim(0, 105)

        fig.tight_layout()
        save_fig(fig, figures_dir, f"5_pipeline_step_success_{stage}", fmt)
        if show:
            plt.show()
        plt.close(fig)
        return

    agg = (
        df.groupby("model")[[stage for stage, _ in valid_stages]]
        .mean().mul(100).round(1).reset_index()
    )
    agg["label"] = agg["model"].apply(short_name)
    agg["mean_success"] = agg[[stage for stage, _ in valid_stages]].mean(axis=1)
    agg = agg.sort_values("mean_success", ascending=True)

    y = np.arange(len(agg))
    height = 0.8 / len(valid_stages)

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(agg))))
    for i, (stage, label) in enumerate(valid_stages):
        offset = (i - len(valid_stages) / 2 + 0.5) * height
        ax.barh(y + offset, agg[stage], height, label=label, color=PALETTE[i], edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(agg["label"])
    ax.set_xlabel("Success Rate (%)")
    ax.set_title("Pipeline Stage Success Rates")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_fig(fig, figures_dir, "5_success_funnel", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Per-Notebook Heatmap (train success)
# ---------------------------------------------------------------------------

def plot_notebook_heatmap(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Heatmap: rows = models, columns = notebook IDs, value = mean train success rate (simple runner, complexity 4)."""
    df_filtered = filter_simple_complexity(df, complexity=4)
    if df_filtered.empty:
        logger.warning("Skipping notebook heatmap — no rows for simple runner / complexity 4.")
        return
    value_col = pipeline_step if pipeline_step is not None else "train_score"
    pivot = df_filtered.pivot_table(index="model", columns="notebook_id", values=value_col, aggfunc="mean")
    if pipeline_step is not None:
        pivot = pivot.mul(100)
    pivot.index = [short_name(m) for m in pivot.index]

    # Sort rows by mean score
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    # Sort columns naturally (nb1, nb2, …)
    cols = natural_notebook_order(pivot.columns)
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(cols)), max(4, 0.45 * len(pivot))))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"{pipeline_step_label(pipeline_step)} (%)" if pipeline_step else "Train Success Rate (%)", "shrink": 0.8},
    )
    ax.set_title(f"{pipeline_step_label(pipeline_step)} per Model × Notebook" if pipeline_step else "Train Success Rate per Model × Notebook")
    ax.set_xlabel("Notebook")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_fig(fig, figures_dir, f"6_notebook_heatmap_{pipeline_step}" if pipeline_step else "6_notebook_heatmap", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — Score Distribution Boxplots
# ---------------------------------------------------------------------------

def plot_score_distributions(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Box plots of train and inference success-rate distributions per model."""
    df_plot = df.copy()
    df_plot["label"] = df_plot["model"].apply(short_name)

    if pipeline_step is not None:
        df_plot["success_rate"] = pipeline_step_values(df_plot, pipeline_step)
        order = (
            df_plot.groupby("label")["success_rate"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

        fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(order)), 5))
        sns.boxplot(
            data=df_plot,
            x="label",
            y="success_rate",
            order=order,
            palette="tab10",
            flierprops={"marker": ".", "markersize": 4},
            ax=ax,
        )
        ax.set_title(pipeline_step_label(pipeline_step))
        ax.set_xlabel("")
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(-5, 105)
        ax.tick_params(axis="x", rotation=40)

        fig.suptitle("Pipeline-Step Success Distribution per Model", fontsize=13)
        fig.tight_layout()
        save_fig(fig, figures_dir, f"7_score_distributions_{pipeline_step}", fmt)
        if show:
            plt.show()
        plt.close(fig)
        return

    order = (
        df_plot.groupby("label")["train_score"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.8 * len(order)), 5), sharey=True)

    for ax, (score_col, title) in zip(axes, [("train_score", "Train Success"), ("inference_score", "Inference Success")]):
        sns.boxplot(
            data=df_plot,
            x="label",
            y=score_col,
            order=order,
            palette="tab10",
            flierprops={"marker": ".", "markersize": 4},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Success Rate (%)" if ax == axes[0] else "")
        ax.set_ylim(-5, 100)
        ax.tick_params(axis="x", rotation=40)

    fig.suptitle("Success-Rate Distributions per Model", fontsize=13)
    fig.tight_layout()
    save_fig(fig, figures_dir, "7_score_distributions", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — Runner Comparison (only when multiple runners present)
# ---------------------------------------------------------------------------

def plot_runner_comparison(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Grouped bar chart comparing runners across models (only if >1 runner)."""
    if df["runner"].nunique() < 2:
        logger.info("Skipping runner comparison — only one runner in data.")
        return

    if pipeline_step is not None:
        agg = df.groupby(["runner", "model"])[[pipeline_step]].mean().mul(100).round(2).reset_index()
        agg["aggregated"] = agg[pipeline_step]
        agg["label"] = agg["model"].apply(short_name)

        fig, ax = plt.subplots(figsize=(max(10, 0.8 * agg["label"].nunique()), 5))
        runners = agg["runner"].unique()
        x = np.arange(agg["label"].nunique())
        width = 0.8 / len(runners)
        labels_sorted = agg.groupby("label")["aggregated"].mean().sort_values(ascending=False).index.tolist()

        for i, runner in enumerate(runners):
            sub = agg[agg["runner"] == runner].set_index("label").reindex(labels_sorted)
            offset = (i - len(runners) / 2 + 0.5) * width
            ax.bar(x + offset, sub["aggregated"].fillna(0), width, label=runner,
                   color=PALETTE[i], edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_sorted, rotation=35, ha="right")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(f"Runner Comparison — {pipeline_step_label(pipeline_step)}")
        ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        save_fig(fig, figures_dir, f"8_runner_comparison_{pipeline_step}", fmt)
        if show:
            plt.show()
        plt.close(fig)
        return

    agg = df.groupby(["runner", "model"])[["inference_score", "requirements_match_score"]].mean().round(4).reset_index()
    agg["aggregated"] = agg["inference_score"] * agg["requirements_match_score"]
    agg["label"] = agg["model"].apply(short_name)

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * agg["label"].nunique()), 5))
    runners = agg["runner"].unique()
    x = np.arange(agg["label"].nunique())
    width = 0.8 / len(runners)
    labels_sorted = agg.groupby("label")["aggregated"].mean().sort_values(ascending=False).index.tolist()

    for i, runner in enumerate(runners):
        sub = agg[agg["runner"] == runner].set_index("label").reindex(labels_sorted)
        offset = (i - len(runners) / 2 + 0.5) * width
        ax.bar(x + offset, sub["aggregated"].fillna(0), width, label=runner,
               color=PALETTE[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted, rotation=35, ha="right")
    ax.set_ylabel("Combined Success Rate (%)")
    ax.set_title("Runner Comparison")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, figures_dir, "8_runner_comparison", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9 — Correlation matrix of all score metrics
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, figures_dir: Path, fmt: str, show: bool):
    """Correlation heatmap of the numeric scoring and success-rate columns."""
    numeric_cols = [
        "train_complexity_score",
        "inference_complexity_score",
        "artifact_match_score",
        "requirements_match_score",
        "train_score",
        "inference_score",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    if len(available) < 3:
        logger.warning("Skipping correlation matrix — not enough numeric columns.")
        return

    corr = df[available].corr()
    nice_labels = {
        "train_complexity_score": "Train Complexity",
        "inference_complexity_score": "Inf. Complexity",
        "artifact_match_score": "Artifact Match",
        "requirements_match_score": "Req. Match",
        "train_score": "Train Success",
        "inference_score": "Inference Success",
    }
    corr.index = [nice_labels.get(c, c) for c in corr.index]
    corr.columns = [nice_labels.get(c, c) for c in corr.columns]

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "ticks": [-1, -0.5, 0, 0.5, 1]},
    )
    ax.set_title("Correlation Matrix of Scoring Metrics")
    fig.tight_layout()
    save_fig(fig, figures_dir, "9_correlation_matrix", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10 — Radar / Spider chart: top-N models across all dimensions
# ---------------------------------------------------------------------------

def plot_radar(df: pd.DataFrame, figures_dir: Path, fmt: str, show: bool, top_n: int = 8):
    """Spider/radar chart comparing the top-N models across key metrics."""
    metrics = ["train_score", "inference_score", "requirements_match_score",
               "artifact_match_score", "train_syntax_valid", "train_execution_success"]
    labels_map = {
        "train_score": "Train\nSuccess",
        "inference_score": "Inference\nSuccess",
        "requirements_match_score": "Req.\nMatch",
        "artifact_match_score": "Artifact\nMatch",
        "train_syntax_valid": "Syntax\nValid",
        "train_execution_success": "Train\nExecution",
    }

    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        logger.warning("Skipping radar chart — not enough metric columns.")
        return

    agg = df.groupby("model")[available].mean().reset_index()

    # Normalise to 0–1
    norm = agg.copy()
    for col in available:
        col_max = norm[col].max()
        if col_max > 0:
            norm[col] = norm[col] / col_max

    # Select top-N by mean normalised score
    norm["_mean"] = norm[available].mean(axis=1)
    norm = norm.nlargest(top_n, "_mean")
    agg = agg[agg["model"].isin(norm["model"])]

    # Re-normalise per metric for the selected subset
    plot_data = agg.copy()
    for col in available:
        col_max = agg[col].max()
        if col_max > 0:
            plot_data[col] = agg[col] / col_max

    N = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([labels_map.get(m, m) for m in available], size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7)

    colors = sns.color_palette("tab10", n_colors=len(plot_data))
    for (_, row), color in zip(plot_data.iterrows(), colors):
        values = [row[m] for m in available]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, color=color, label=short_name(row["model"]))
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    ax.set_title(f"Top-{top_n} Models — Normalised Metric Radar", pad=20)
    fig.tight_layout()
    save_fig(fig, figures_dir, "10_radar_chart", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 11 — Aggregated Breakdown per Notebook with Variance
# ---------------------------------------------------------------------------

def plot_notebook_score_breakdown(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Grouped bar chart with error bars: mean train/inference success and requirements score
    per notebook, aggregated over all models and runs. Error bars show ±1 std dev."""
    if pipeline_step is not None:
        success_df = df.assign(_success=pipeline_step_values(df, pipeline_step))
        agg_mean = success_df.groupby("notebook_id")["_success"].mean()
        agg_std = success_df.groupby("notebook_id")["_success"].std().fillna(0)

        nb_order = natural_notebook_order(agg_mean.index)
        means = agg_mean.loc[nb_order].values
        stds = agg_std.loc[nb_order].values
        x = np.arange(len(nb_order))

        fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(nb_order)), 5))
        ax.bar(
            x, means, 0.55,
            label=pipeline_step_label(pipeline_step),
            color=PALETTE[0],
            edgecolor="white",
            yerr=stds,
            capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "#444", "capthick": 1.2},
        )

        tick_labels = [f"{nb}\n(σ={agg_std.loc[nb]:.1f})" for nb in nb_order]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_xlabel("Notebook  (σ = std dev)")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(f"Per-Notebook {pipeline_step_label(pipeline_step)}")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right")

        fig.tight_layout()
        save_fig(fig, figures_dir, f"11_notebook_score_breakdown_{pipeline_step}", fmt)
        if show:
            plt.show()
        plt.close(fig)
        return

    df = filter_simple_complexity(df, complexity=4)

    metrics = ["train_score", "inference_score", "requirements_match_score"]
    labels  = ["Train Success", "Inference Success", "Requirements Match"]

    agg_mean = df.groupby("notebook_id")[metrics].mean()
    agg_std  = df.groupby("notebook_id")[metrics].std().fillna(0)

    agg_mean["requirements_match_score"] = agg_mean["requirements_match_score"].mul(100)
    agg_std["requirements_match_score"] = agg_std["requirements_match_score"].mul(100)

    # Sort notebooks naturally (nb1, nb2, …)
    nb_order = natural_notebook_order(agg_mean.index)
    agg_mean = agg_mean.loc[nb_order]
    agg_std  = agg_std.loc[nb_order]

    x = np.arange(len(nb_order))
    width = 0.25
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(nb_order)), 5))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, PALETTE)):
        offset = (i - n_metrics / 2 + 0.5) * width
        means = agg_mean[metric].values
        stds  = agg_std[metric].values
        bars = ax.bar(
            x + offset, means, width,
            label=label,
            color=color,
            edgecolor="white",
            yerr=stds,
            capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "#444", "capthick": 1.2},
        )

    tick_labels = [
        f"{nb}"
        for nb in nb_order
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel("Notebook")
    ax.set_ylabel("Mean Success Rate (%)")
    ax.set_title("Success Breakdown per Notebook")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_fig(fig, figures_dir, "11_notebook_score_breakdown", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12 — Cross-runner model comparison (models run on all runners)
# ---------------------------------------------------------------------------

RUNNER_ORDER = ["simple", "cot", "agentic"]
RUNNER_LABELS = {"simple": "Simple", "cot": "Chain-of-Thought", "agentic": "Agentic"}


def _plot_cross_runner_single(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    score_col: str,
    score_label: str,
    fig_stem: str,
):
    """Helper: cross-runner comparison figure for a single score metric."""
    runners_present = ordered_runners(df)

    if len(runners_present) < 2:
        logger.info(f"Skipping cross-runner comparison ({score_label}) — fewer than 2 runners in data.")
        return

    model_runner_counts = df.groupby("model")["runner"].nunique()
    complete_models = model_runner_counts[model_runner_counts == len(runners_present)].index.tolist()

    if not complete_models:
        logger.warning(f"Skipping cross-runner comparison ({score_label}) — no model was run on all runners.")
        return

    sub = df[df["model"].isin(complete_models)].copy()
    if "notebook_order" in sub.columns:
        sub = sub[sub["notebook_order"] == "original"].copy()
    sub["label"] = sub["model"].apply(short_name)

    # ── Panel 1: grouped bar — mean score per model × runner ──
    agg = (
        sub.groupby(["model", "runner"])[score_col]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)
    agg["label"] = agg["model"].apply(short_name)

    model_order = (
        agg.groupby("label")["mean"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    x = np.arange(len(model_order))
    width = 0.8 / len(runners_present)
    runner_colors = {r: PALETTE[i] for i, r in enumerate(runners_present)}

    fig, (ax_bar, ax_nb) = plt.subplots(
        2, 1,
        figsize=(max(10, 0.9 * len(model_order)), 10),
        gridspec_kw={"height_ratios": [2, 1.4]},
    )

    for i, runner in enumerate(runners_present):
        sub_r = agg[agg["runner"] == runner].set_index("label").reindex(model_order)
        offset = (i - len(runners_present) / 2 + 0.5) * width
        ax_bar.bar(
            x + offset,
            sub_r["mean"].fillna(0),
            width,
            yerr=sub_r["std"].fillna(0),
            label=RUNNER_LABELS.get(runner, runner),
            color=runner_colors[runner],
            edgecolor="white",
            capsize=3,
            error_kw={"elinewidth": 1.2, "ecolor": "#444", "capthick": 1.2},
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(model_order, rotation=30, ha="right")
    ax_bar.set_ylabel(f"{score_label} (%)")
    ax_bar.set_title(f"Cross-Runner Model Comparison — {score_label}")
    ax_bar.set_ylim(0, 100)
    ax_bar.legend(title="Runner", loc="upper right")

    # ── Panel 2: per-notebook score — grouped bars per runner ─
    nb_agg = (
        sub.groupby(["runner", "notebook_id"])[score_col]
        .mean()
        .reset_index()
    )
    nb_order = natural_notebook_order(nb_agg["notebook_id"].unique())

    x_nb = np.arange(len(nb_order))
    nb_width = 0.8 / len(runners_present)

    for i, runner in enumerate(runners_present):
        sub_r = nb_agg[nb_agg["runner"] == runner].set_index("notebook_id").reindex(nb_order)
        offset = (i - len(runners_present) / 2 + 0.5) * nb_width
        ax_nb.bar(
            x_nb + offset,
            sub_r[score_col].fillna(0),
            nb_width,
            label=RUNNER_LABELS.get(runner, runner),
            color=runner_colors[runner],
            edgecolor="white",
        )

    ax_nb.set_xticks(x_nb)
    ax_nb.set_xticklabels(nb_order, rotation=0)
    ax_nb.set_xlabel("Notebook")
    ax_nb.set_ylabel(f"{score_label} (%)")
    ax_nb.set_title(f"Per-Notebook {score_label} by Runner")
    ax_nb.set_ylim(0, 100)
    ax_nb.legend(title="Runner", loc="upper right")
    fig.tight_layout(h_pad=6.0)
    save_fig(fig, figures_dir, fig_stem, fmt)
    if show:
        plt.show()
    plt.close(fig)


def plot_cross_runner_comparison(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """For each model that was evaluated on all available runners, compare their
    train and inference success rates side-by-side as two separate figures."""
    if pipeline_step is not None:
        _plot_cross_runner_single(
            df, figures_dir, fmt, show,
            score_col=pipeline_step,
            score_label=pipeline_step_label(pipeline_step),
            fig_stem=f"12_cross_runner_comparison_{pipeline_step}",
        )
        return

    _plot_cross_runner_single(
        df, figures_dir, fmt, show,
        score_col="train_score",
        score_label="Train Success Rate",
        fig_stem="12a_cross_runner_train_success",
    )
    _plot_cross_runner_single(
        df, figures_dir, fmt, show,
        score_col="inference_score",
        score_label="Inference Success Rate",
        fig_stem="12b_cross_runner_inference_success",
    )


# ---------------------------------------------------------------------------
# Figure 13 — Per-notebook combined success rate (simple runner, complexity 4)
# ---------------------------------------------------------------------------

def plot_notebook_scores_simple_c4(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Bar chart: combined success rate per notebook,
    filtered to simple runner at complexity 4, aggregated over all models & runs."""
    sub = filter_simple_complexity(df, complexity=4)
    if sub.empty:
        logger.warning("Skipping per-notebook simple/c4 plot — no matching rows.")
        return

    if pipeline_step is not None:
        sub["aggregated_score"] = pipeline_step_values(sub, pipeline_step)
    else:
        sub["aggregated_score"] = sub["inference_score"] * sub["requirements_match_score"]

    agg_mean = sub.groupby("notebook_id")["aggregated_score"].mean()
    agg_std  = sub.groupby("notebook_id")["aggregated_score"].std().fillna(0)

    nb_order = natural_notebook_order(agg_mean.index)
    means = agg_mean.loc[nb_order].values
    stds  = agg_std.loc[nb_order].values

    x = np.arange(len(nb_order))

    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(nb_order)), 5))

    bars = ax.bar(
        x, means,
        color=PALETTE[0],
        edgecolor="white",
        yerr=stds,
        capsize=4,
        error_kw={"elinewidth": 1.3, "ecolor": "#444", "capthick": 1.3},
    )

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(nb_order, rotation=0)
    ax.set_xlabel("Notebook")
    ax.set_ylabel("Success Rate (%)" if pipeline_step else "Combined Success Rate (%)")
    ax.set_title(f"{pipeline_step_label(pipeline_step)} per Notebook" if pipeline_step else "Combined Success Rate per Notebook")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    save_fig(fig, figures_dir, f"13_notebook_scores_simple_c4_{pipeline_step}" if pipeline_step else "13_notebook_scores_simple_c4", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 19 — Swapped notebook cells comparison (simple runner, complexity 4)
# ---------------------------------------------------------------------------

def _plot_notebook_order_single(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    score_col: str,
    score_label: str,
    fig_stem: str,
):
    """Helper: notebook-order comparison figure for a single score metric."""
    if "notebook_order" not in df.columns:
        logger.info(f"Skipping notebook-order comparison ({score_label}) — notebook_order column is missing.")
        return

    # Filter to simple runner, complexity 4, but keep ALL notebook orders
    sub = df[(df["runner"] == "simple") & (df["complexity"] == 4)].copy()
    if sub.empty:
        logger.warning(f"Skipping notebook-order comparison ({score_label}) — no rows for simple runner / complexity 4.")
        return

    order_counts = sub["notebook_order"].value_counts()
    notebook_orders = ordered_notebook_orders(order_counts.index.tolist())
    if len(notebook_orders) < 2:
        logger.info(f"Skipping notebook-order comparison ({score_label}) — fewer than 2 notebook orders in the filtered data.")
        return

    # Keep only models that appear in both notebook orders
    model_order_counts = sub.groupby("model")["notebook_order"].nunique()
    complete_models = model_order_counts[model_order_counts == len(notebook_orders)].index.tolist()
    if not complete_models:
        logger.warning(f"Skipping notebook-order comparison ({score_label}) — no model was run on all notebook orders.")
        return

    plot_df = sub[sub["model"].isin(complete_models)].copy()

    model_agg = (
        plot_df.groupby(["model", "notebook_order"])[score_col]
        .mean()
        .reset_index()
    )
    notebook_agg = (
        plot_df.groupby(["notebook_id", "notebook_order"])[score_col]
        .mean()
        .reset_index()
    )

    model_order = (
        model_agg.groupby("model")[score_col]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    notebook_ordered = natural_notebook_order(notebook_agg["notebook_id"].unique())
    x_models = np.arange(len(model_order))
    x_notebooks = np.arange(len(notebook_ordered))
    width = 0.8 / len(notebook_orders)
    order_colors = {order: PALETTE[i % len(PALETTE)] for i, order in enumerate(notebook_orders)}

    fig, (ax_models, ax_notebooks) = plt.subplots(
        2,
        1,
        figsize=(max(11, 0.55 * len(model_order)), 10),
        gridspec_kw={"height_ratios": [1.5, 1]},
    )

    for i, notebook_order_value in enumerate(notebook_orders):
        model_slice = (
            model_agg[model_agg["notebook_order"] == notebook_order_value]
            .set_index("model")
            .reindex(model_order)
        )
        notebook_slice = (
            notebook_agg[notebook_agg["notebook_order"] == notebook_order_value]
            .set_index("notebook_id")
            .reindex(notebook_ordered)
        )
        offset = (i - len(notebook_orders) / 2 + 0.5) * width

        ax_models.bar(
            x_models + offset,
            model_slice[score_col].fillna(0),
            width,
            label=notebook_order_label(notebook_order_value),
            color=order_colors[notebook_order_value],
            edgecolor="white",
        )
        ax_notebooks.bar(
            x_notebooks + offset,
            notebook_slice[score_col].fillna(0),
            width,
            label=notebook_order_label(notebook_order_value),
            color=order_colors[notebook_order_value],
            edgecolor="white",
        )

    ax_models.set_xticks(x_models)
    ax_models.set_xticklabels([short_name(model) for model in model_order], rotation=35, ha="right")
    ax_models.set_ylabel(f"{score_label} (%)")
    ax_models.set_title(f"Original vs Adjacent Swap — Model Comparison ({score_label})")
    ax_models.set_ylim(0, 100)
    ax_models.legend(title="Notebook Order", loc="upper right")

    ax_notebooks.set_xticks(x_notebooks)
    ax_notebooks.set_xticklabels(notebook_ordered, rotation=0)
    ax_notebooks.set_xlabel("Notebook")
    ax_notebooks.set_ylabel(f"{score_label} (%)")
    ax_notebooks.set_title(f"Original vs Adjacent Swap — Notebook Comparison ({score_label})")
    ax_notebooks.set_ylim(0, 100)
    ax_notebooks.legend(title="Notebook Order", loc="upper right")

    fig.tight_layout(h_pad=3.0)
    save_fig(fig, figures_dir, fig_stem, fmt)
    if show:
        plt.show()
    plt.close(fig)


def plot_notebook_order_comparison(
    df: pd.DataFrame,
    figures_dir: Path,
    fmt: str,
    show: bool,
    pipeline_step: str | None = None,
):
    """Compare original vs adjacent-swap notebook order with separate figures
    for train and inference success rates.

    Uses the swapped-cells benchmark slice: simple runner at complexity 4.
    """
    if pipeline_step is not None:
        _plot_notebook_order_single(
            df, figures_dir, fmt, show,
            score_col=pipeline_step,
            score_label=pipeline_step_label(pipeline_step),
            fig_stem=f"19_notebook_order_comparison_{pipeline_step}",
        )
        return

    _plot_notebook_order_single(
        df, figures_dir, fmt, show,
        score_col="train_score",
        score_label="Train Success Rate",
        fig_stem="19a_notebook_order_train_success",
    )
    _plot_notebook_order_single(
        df, figures_dir, fmt, show,
        score_col="inference_score",
        score_label="Inference Success Rate",
        fig_stem="19b_notebook_order_inference_success",
    )



# ---------------------------------------------------------------------------
# Figure 14 — Token & time usage for models run on all 3 runners
# ---------------------------------------------------------------------------

def plot_token_time_usage(metrics_path: Path, figures_dir: Path, fmt: str, show: bool):
    """Two-panel grouped bar chart: mean inference time and mean total tokens
    for every model that was evaluated on all three runners (simple, cot, agentic)."""
    if not metrics_path.exists():
        logger.warning(f"Skipping token/time plot — metrics file not found: {metrics_path}")
        return

    mdf = pd.read_csv(metrics_path)

    all_runners = set(mdf["runner"].unique())
    if len(all_runners) < 2:
        logger.info("Skipping token/time plot — fewer than 2 runners in metrics data.")
        return

    # Models present on every runner
    model_runner_counts = mdf.groupby("model")["runner"].nunique()
    complete_models = model_runner_counts[model_runner_counts == len(all_runners)].index.tolist()

    if not complete_models:
        logger.warning("Skipping token/time plot — no model was run on all runners.")
        return

    logger.info(f"Token/time plot: {len(complete_models)} models on runners {sorted(all_runners)}")

    sub = mdf[mdf["model"].isin(complete_models)].copy()
    sub["label"] = sub["model"].apply(short_name)

    runners_present = ordered_runners(sub)
    runner_colors = {r: PALETTE[i] for i, r in enumerate(runners_present)}

    model_order = (
        sub[sub["runner"] == runners_present[0]]
        .set_index("label")["mean_time_taken"]
        .sort_values(ascending=False)
        .index.tolist()
    )
    # Fall back to alphabetical if simple not present
    if not model_order:
        model_order = sorted(sub["label"].unique())

    x = np.arange(len(model_order))
    width = 0.8 / len(runners_present)

    fig, (ax_time, ax_tok) = plt.subplots(
        2, 1, figsize=(max(8, 0.9 * len(model_order)), 9),
        gridspec_kw={"hspace": 0.45},
    )

    for i, runner in enumerate(runners_present):
        sub_r = sub[sub["runner"] == runner].set_index("label").reindex(model_order)
        offset = (i - len(runners_present) / 2 + 0.5) * width
        label = RUNNER_LABELS.get(runner, runner)

        ax_time.bar(
            x + offset, sub_r["mean_time_taken"].fillna(0), width,
            label=label, color=runner_colors[runner], edgecolor="white",
        )
        ax_tok.bar(
            x + offset, sub_r["mean_total_tokens"].fillna(0), width,
            label=label, color=runner_colors[runner], edgecolor="white",
        )

    for ax, ylabel, title in [
        (ax_time, "Mean Inference Time (s)", "Mean Inference Time per Model & Runner"),
        (ax_tok,  "Mean Total Tokens",        "Mean Total Tokens per Model & Runner"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="Runner", loc="upper right")
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_fig(fig, figures_dir, "14_token_time_usage", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 16 — Own inference vs real artifacts summary
# ---------------------------------------------------------------------------

def plot_own_inference_artifact_summary(details_path: Path, figures_dir: Path, fmt: str, show: bool):
    """Compare exact and continuous artifact-match rates per model for simple runs at complexity 4."""
    if not details_path.exists():
        logger.info(f"Skipping own-inference artifact summary plot — file not found: {details_path}")
        return

    df = pd.read_csv(details_path)
    required_cols = {
        "model",
        "runner",
        "complexity",
        "generated_inference_success",
        "real_artifact_inference_success",
        "outputs_match",
        "continuous_similarity",
        "continuous_match",
    }
    if not required_cols.issubset(df.columns):
        logger.warning("Skipping own-inference artifact summary plot — required columns are missing.")
        return

    filtered_df = filter_simple_complexity(df, complexity=4)
    if filtered_df.empty:
        logger.warning("Skipping own-inference artifact summary plot — no rows for simple runner / complexity 4.")
        return

    successful_df = filtered_df[
        filtered_df["generated_inference_success"].fillna(False)
        & filtered_df["real_artifact_inference_success"].fillna(False)
    ].copy()
    if successful_df.empty:
        logger.warning(
            "Skipping own-inference artifact summary plot — no successful paired inference rows for simple runner / complexity 4."
        )
        return

    plot_df = (
        successful_df.groupby("model")
        .agg(
            outputs_match_rate=("outputs_match", lambda s: s.fillna(False).mean()),
            continuous_match_rate=("continuous_match", lambda s: s.fillna(False).mean()),
            avg_continuous_similarity=(
                "continuous_similarity",
                lambda s: s.fillna(0).clip(lower=0, upper=1).mean(),
            ),
        )
        .reset_index()
    )
    rate_cols = ["outputs_match_rate", "continuous_match_rate", "avg_continuous_similarity"]
    plot_df[rate_cols] = plot_df[rate_cols].mul(100)
    plot_df["label"] = plot_df["model"].apply(short_name)
    plot_df = plot_df.sort_values("continuous_match_rate", ascending=True)

    y = np.arange(len(plot_df))
    height = 0.24

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(plot_df))))
    for i, (metric, label, color) in enumerate([
        ("outputs_match_rate", "Exact Match Rate", PALETTE[0]),
        ("continuous_match_rate", "Continuous Match Rate", PALETTE[1]),
        ("avg_continuous_similarity", "Avg. Similarity", PALETTE[2]),
    ]):
        offset = (i - 1) * height
        ax.barh(y + offset, plot_df[metric], height, label=label, color=color, edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Rate (%)")
    ax.set_title("Own Inference vs Real Artifacts — Model Summary (Simple, Complexity 4)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_fig(fig, figures_dir, "16_own_inference_artifact_summary", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 17 — Own inference vs real artifacts heatmap
# ---------------------------------------------------------------------------

def plot_own_inference_artifact_heatmap(details_path: Path, figures_dir: Path, fmt: str, show: bool):
    """Heatmap of average continuous similarity per notebook and model for simple runs at complexity 4."""
    if not details_path.exists():
        logger.info(f"Skipping own-inference artifact heatmap — file not found: {details_path}")
        return

    df = pd.read_csv(details_path)
    required_cols = {
        "model",
        "notebook_id",
        "runner",
        "complexity",
        "generated_inference_success",
        "real_artifact_inference_success",
        "continuous_similarity",
    }
    if not required_cols.issubset(df.columns):
        logger.warning("Skipping own-inference artifact heatmap — required columns are missing.")
        return

    filtered_df = filter_simple_complexity(df, complexity=4)
    if filtered_df.empty:
        logger.warning("Skipping own-inference artifact heatmap — no rows for simple runner / complexity 4.")
        return

    valid_df = filtered_df[
        filtered_df["generated_inference_success"].fillna(False)
        & filtered_df["real_artifact_inference_success"].fillna(False)
    ].copy()
    if valid_df.empty:
        logger.warning(
            "Skipping own-inference artifact heatmap — no successful paired inference rows for simple runner / complexity 4."
        )
        return

    valid_df["continuous_similarity_pct"] = valid_df["continuous_similarity"].fillna(0).clip(lower=0, upper=1).mul(100)
    pivot = valid_df.pivot_table(
        index="model",
        columns="notebook_id",
        values="continuous_similarity_pct",
        aggfunc="mean",
    )
    if pivot.empty:
        logger.warning("Skipping own-inference artifact heatmap — no plottable similarity values.")
        return

    pivot.index = [short_name(model) for model in pivot.index]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    cols = natural_notebook_order(pivot.columns)
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(cols)), max(4, 0.45 * len(pivot))))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Average Continuous Similarity (%)", "shrink": 0.8},
    )
    ax.set_title("Own Inference vs Real Artifacts — Notebook Heatmap (Simple, Complexity 4)")
    ax.set_xlabel("Notebook")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    save_fig(fig, figures_dir, "17_own_inference_artifact_heatmap", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 18 — Train success vs artifact similarity
# ---------------------------------------------------------------------------

def plot_train_success_vs_artifact_similarity(
    scoring_df: pd.DataFrame,
    details_path: Path,
    figures_dir: Path,
    fmt: str,
    show: bool,
):
    """Compare strict and similarity-adjusted train success per model for simple runs at complexity 4."""
    if not details_path.exists():
        logger.info(f"Skipping train-success vs artifact-similarity plot — file not found: {details_path}")
        return

    required_scoring_cols = {
        "notebook_id",
        "model",
        "runner",
        "complexity",
        "run",
        "train_score",
        "train_syntax_valid",
        "train_execution_success",
    }
    if not required_scoring_cols.issubset(scoring_df.columns):
        logger.warning("Skipping train-success vs artifact-similarity plot — scoring columns are missing.")
        return

    details_df = pd.read_csv(details_path)
    required_detail_cols = {
        "notebook_id",
        "model",
        "runner",
        "complexity",
        "run",
        "generated_inference_success",
        "real_artifact_inference_success",
        "continuous_similarity",
    }
    if not required_detail_cols.issubset(details_df.columns):
        logger.warning("Skipping train-success vs artifact-similarity plot — artifact detail columns are missing.")
        return

    scoring_filtered = filter_simple_complexity(scoring_df, complexity=4)
    if scoring_filtered.empty:
        logger.warning("Skipping train-success vs artifact-similarity plot — no scoring rows for simple runner / complexity 4.")
        return

    details_filtered = filter_simple_complexity(details_df, complexity=4)
    if details_filtered.empty:
        logger.warning("Skipping train-success vs artifact-similarity plot — no artifact rows for simple runner / complexity 4.")
        return

    merge_keys = ["notebook_id", "model", "runner", "complexity", "run"]
    merged_df = scoring_filtered[
        merge_keys + ["train_score", "train_syntax_valid", "train_execution_success"]
    ].merge(
        details_filtered[
            merge_keys + ["generated_inference_success", "real_artifact_inference_success", "continuous_similarity"]
        ],
        on=merge_keys,
        how="left",
    )
    if merged_df.empty:
        logger.warning("Skipping train-success vs artifact-similarity plot — no overlapping rows after merge.")
        return

    train_prereq_success = (
        merged_df["train_syntax_valid"].fillna(False)
        & merged_df["train_execution_success"].fillna(False)
    )
    valid_similarity = (
        merged_df["generated_inference_success"].fillna(False)
        & merged_df["real_artifact_inference_success"].fillna(False)
    )
    merged_df["artifact_similarity_pct"] = (
        merged_df["continuous_similarity"].where(valid_similarity, 0).fillna(0).clip(lower=0, upper=1).mul(100)
    )
    merged_df["train_score_with_similarity"] = np.where(
        train_prereq_success,
        merged_df["artifact_similarity_pct"],
        0,
    )

    plot_df = (
        merged_df.groupby("model")[["train_score", "train_score_with_similarity"]]
        .mean()
        .reset_index()
    )
    plot_df["label"] = plot_df["model"].apply(short_name)
    plot_df = plot_df.sort_values("train_score_with_similarity", ascending=True)

    y = np.arange(len(plot_df))
    height = 0.34

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(plot_df))))
    ax.barh(
        y - height / 2,
        plot_df["train_score"],
        height,
        label="Strict Train Success",
        color=PALETTE[1],
        edgecolor="white",
    )
    ax.barh(
        y + height / 2,
        plot_df["train_score_with_similarity"],
        height,
        label="Train Success with Artifact Similarity",
        color=PALETTE[0],
        edgecolor="white",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Rate (%)")
    ax.set_title("Train Success with Artifact Similarity (Simple, Complexity 4)")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_fig(fig, figures_dir, "18_train_success_vs_artifact_similarity", fmt)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Synthetic demo data (for testing before real runs complete)
# ---------------------------------------------------------------------------

def generate_demo_data(output_dir: Path):
    """Create a demo scoring_report.csv so the script can be validated early."""
    rng = np.random.default_rng(42)
    models = [
        "gemini-2.5-flash", "llama-3.3-70b-instruct", "qwen3-235b-a22b",
        "deepseek-r1-distill-llama-70b", "mistral-large-3-675b-instruct-2512",
        "meta-llama-3.1-8b-instruct", "qwen3-32b", "devstral-2-123b-instruct-2512",
    ]
    notebooks = [f"nb{i}" for i in range(1, 11)]
    complexities = [1, 2, 3, 4, 5]
    runners = ["simple", "cot", "agentic"]
    # Simulate that only a subset of models ran on all 3 runners
    models_all_runners = models[:5]
    runner_bias = {"simple": 1.0, "cot": 1.1, "agentic": 0.85}
    rows = []

    model_bias = {m: rng.uniform(0.4, 1.0) for m in models}

    for model in models:
        bias = model_bias[model]
        for nb in notebooks:
            for complexity in complexities:
                for runner in (runners if model in models_all_runners else ["simple"]):
                    rb = runner_bias.get(runner, 1.0)
                    syntax_ok = rng.random() < min(1.0, (0.7 + bias * 0.3) * rb)
                    exec_ok = syntax_ok and rng.random() < min(1.0, (0.5 + bias * 0.3) * rb)
                    art_score = rng.uniform(0, 1) * bias if exec_ok else 0.0
                    inf_syntax = rng.random() < (0.7 + bias * 0.3)
                    own_ok = inf_syntax and rng.random() < (0.4 + bias * 0.4)
                    llm_ok = inf_syntax and rng.random() < (0.3 + bias * 0.4)
                    match_ok = own_ok and llm_ok and rng.random() < 0.5

                    train_score = (
                        (20 if syntax_ok else 0)
                        + (60 if exec_ok else 0)
                        + 20 * art_score
                    )
                    inference_score = (
                        (20 if inf_syntax else 0)
                        + (40 if own_ok else 0)
                        + (30 if llm_ok else 0)
                        + (10 if match_ok else 0)
                    )
                    req_score = rng.uniform(40, 100) * bias

                    rows.append({
                        "notebook_id": nb,
                        "model": model,
                        "runner": runner,
                        "complexity": complexity,
                        "run": 1,
                        "train_syntax_valid": syntax_ok,
                        "inference_syntax_valid": inf_syntax,
                        "train_complexity_score": rng.uniform(0, 100),
                        "inference_complexity_score": rng.uniform(0, 100),
                        "train_execution_success": exec_ok,
                        "artifact_match_score": art_score,
                        "requirements_match_score": req_score / 100,
                        "requirements_score": round(req_score, 2),
                        "own_inference_success": own_ok,
                        "llm_inference_success": llm_ok,
                        "outputs_match": match_ok,
                        "train_score": round(train_score, 2),
                        "inference_score": round(inference_score, 2),
                    })

    demo_path = output_dir / "scoring_report_demo.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(demo_path, index=False)
    logger.info(f"Demo data written to {demo_path} (real scoring_report.csv left untouched)")
    return demo_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate thesis visualizations from benchmark results.")
    parser.add_argument("--output-dir", default="output", help="Path to the benchmark output directory.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    parser.add_argument("--format", choices=["pdf", "png", "both"], default="png",
                        help="Output format for figures (default: png).")
    parser.add_argument(
        "--pipeline-step",
        choices=[
            "train_syntax_valid",
            "train_execution_success",
            "own_inference_success",
            "llm_inference_success",
            "outputs_match",
        ],
        help="If set, plots use this pipeline step's success rate where that metric is applicable.",
    )
    parser.add_argument("--demo", action="store_true",
                        help="Generate and use synthetic demo data (useful before real runs finish).")
    parser.add_argument("--top-n-radar", type=int, default=8,
                        help="Number of top models to include in the radar chart.")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    figures_dir = output_path / "figures"
    report_path = output_path / "scoring_report.csv"

    if args.demo:
        report_path = generate_demo_data(output_path)

    if not report_path.exists():
        logger.error(
            f"No scoring report found at {report_path}.\n"
            "Run the scoring pipeline first (`python benchmark.py --score-only`), "
            "or use --demo to generate synthetic data."
        )
        return

    df = pd.read_csv(report_path, dtype={"complexity": int})
    df = prepare_success_rate_metrics(df)
    df = drop_models_with_all_nan_inference_scores(df)
    logger.info(f"Loaded {len(df)} rows from {report_path}")
    logger.info(f"Models: {df['model'].nunique()} | Notebooks: {df['notebook_id'].nunique()} | "
                f"Complexities: {sorted(df['complexity'].unique())}")

    apply_style()

    plot_leaderboard(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_score_breakdown(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_complexity_effect(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_complexity_per_model(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_complexity_scores_first5(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_success_funnel(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_notebook_heatmap(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_score_distributions(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_runner_comparison(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_correlation_matrix(df, figures_dir, args.format, args.show)
    plot_radar(df, figures_dir, args.format, args.show, top_n=args.top_n_radar)
    plot_notebook_score_breakdown(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_cross_runner_comparison(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_notebook_scores_simple_c4(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)
    plot_notebook_order_comparison(df, figures_dir, args.format, args.show, pipeline_step=args.pipeline_step)

    metrics_path = output_path / "my_metrics_summary.csv"
    plot_token_time_usage(metrics_path, figures_dir, args.format, args.show)

    own_inference_artifact_details_path = output_path / "own_inference_vs_real_artifacts.csv"
    plot_own_inference_artifact_summary(own_inference_artifact_details_path, figures_dir, args.format, args.show)
    plot_own_inference_artifact_heatmap(own_inference_artifact_details_path, figures_dir, args.format, args.show)
    plot_train_success_vs_artifact_similarity(df, own_inference_artifact_details_path, figures_dir, args.format, args.show)

    logger.info(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
