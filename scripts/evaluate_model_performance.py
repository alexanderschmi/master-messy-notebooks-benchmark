#!/usr/bin/env python3
"""
Evaluate real ML model performance of LLM-generated train.py artifacts.

For each notebook, loads a test subset from the training data, runs inference
on each LLM-generated model, and computes task-appropriate metrics (accuracy,
F1, RMSE) against ground-truth labels.

Output:
  - output/model_performance.csv          (per-run detail)
  - output/model_performance_summary.csv  (aggregated per model)
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

from src.core.output_layout import iter_run_dirs
from src.inference.strategies import (
    load_nb10_inference_bundle,
    load_nb5_inference_bundle,
    load_nb6_inference_bundle,
    load_nb8_inference_bundle,
    load_nb9_inference_bundle,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

RUN_KEY_COLUMNS = ["notebook_id", "runner", "model", "complexity", "notebook_order", "run"]


def write_progress_log(level: str, message: str) -> None:
    """Write a progress-safe log line without corrupting the tqdm bar."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    tqdm.write(f"{timestamp} {level}: {message}")


def load_successful_train_runs(
    output_path: Path,
    target_nb: str | None = None,
    target_runner: str | None = None,
    target_model: str | None = None,
    target_complexity: int | None = None,
) -> set[tuple[str, str, str, int, str, int]]:
    """Return run keys whose train.py completed with strict scoring success."""

    report_path = output_path / "scoring_report.csv"
    if not report_path.exists():
        logger.warning("Scoring report not found at %s; no model runs will be evaluated.", report_path)
        return set()

    df = pd.read_csv(report_path)
    required_cols = {
        "notebook_id",
        "runner",
        "model",
        "complexity",
        "notebook_order",
        "run",
        "train_syntax_valid",
        "train_execution_success",
        "artifact_match_score",
    }
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        if missing_cols == {"notebook_order"}:
            df["notebook_order"] = "original"
            missing_cols = set()
        else:
            logger.warning(
                "Scoring report is missing required train success columns %s; no model runs will be evaluated.",
                sorted(missing_cols),
            )
            return set()

    success_mask = (
        df["train_syntax_valid"].fillna(False).astype(bool)
        & df["train_execution_success"].fillna(False).astype(bool)
        & df["artifact_match_score"].fillna(0).eq(1.0)
    )
    successful_df = df.loc[success_mask].copy()

    if target_nb:
        successful_df = successful_df[successful_df["notebook_id"] == target_nb]
    if target_runner:
        successful_df = successful_df[successful_df["runner"] == target_runner]
    if target_model:
        successful_df = successful_df[successful_df["model"] == target_model]
    if target_complexity is not None:
        successful_df = successful_df[successful_df["complexity"] == target_complexity]

    return {
        (
            row.notebook_id,
            row.runner,
            row.model,
            int(row.complexity),
            row.notebook_order,
            int(row.run),
        )
        for row in successful_df.itertuples(index=False)
    }


def iter_model_dirs(output_path: Path, target_nb=None, target_runner=None, target_model=None, target_complexity=None):
    successful_runs = load_successful_train_runs(
        output_path,
        target_nb=target_nb,
        target_runner=target_runner,
        target_model=target_model,
        target_complexity=target_complexity,
    )

    if not successful_runs:
        logger.info("No model runs met the strict train.py success criteria.")
        return

    for nb_id, runner_name, model_dir, meta in iter_run_dirs(output_path):
        if target_nb and nb_id != target_nb:
            continue
        if target_runner and runner_name != target_runner:
            continue
        if target_model and meta["model"] != target_model:
            continue
        if target_complexity is not None and meta["complexity"] != target_complexity:
            continue

        run_key = (nb_id, runner_name, meta["model"], meta["complexity"], meta["notebook_order"], meta["run"])
        if run_key not in successful_runs:
            continue

        yield nb_id, runner_name, model_dir, meta


def load_existing_results(detailed_path: Path) -> pd.DataFrame:
    """Load prior detailed evaluation results when present."""

    if not detailed_path.exists():
        return pd.DataFrame()

    try:
        existing_df = pd.read_csv(detailed_path)
    except pd.errors.EmptyDataError:
        logger.warning("Existing detailed results at %s are empty; starting fresh.", detailed_path)
        return pd.DataFrame()
    if "notebook_order" not in existing_df.columns:
        existing_df["notebook_order"] = "original"
    missing_cols = [column for column in RUN_KEY_COLUMNS if column not in existing_df.columns]
    if missing_cols:
        logger.warning(
            "Existing detailed results at %s are missing key columns %s; ignoring cached evaluation results.",
            detailed_path,
            missing_cols,
        )
        return pd.DataFrame()

    return existing_df


def successful_run_keys(existing_df: pd.DataFrame) -> set[tuple[str, str, str, int, str, int]]:
    """Return run keys that already evaluated successfully."""

    if existing_df.empty or "evaluation_success" not in existing_df.columns:
        return set()

    success_df = existing_df[existing_df["evaluation_success"].fillna(False).astype(bool)].copy()
    if success_df.empty:
        return set()
    if "notebook_order" not in success_df.columns:
        success_df["notebook_order"] = "original"

    return {
        (
            row.notebook_id,
            row.runner,
            row.model,
            int(row.complexity),
            row.notebook_order,
            int(row.run),
        )
        for row in success_df.itertuples(index=False)
    }


def filter_results_to_allowed_runs(
    existing_df: pd.DataFrame,
    allowed_run_keys: set[tuple[str, str, str, int, str, int]],
) -> pd.DataFrame:
    """Keep only detailed result rows whose run keys still qualify for evaluation."""

    if existing_df.empty:
        return existing_df
    if not allowed_run_keys:
        return existing_df.iloc[0:0].copy()

    keyed_df = existing_df.copy()
    if "notebook_order" not in keyed_df.columns:
        keyed_df["notebook_order"] = "original"
    keyed_df["_run_key"] = list(
        zip(
            keyed_df["notebook_id"],
            keyed_df["runner"],
            keyed_df["model"],
            keyed_df["complexity"].astype(int),
            keyed_df["notebook_order"],
            keyed_df["run"].astype(int),
        )
    )
    filtered_df = keyed_df[keyed_df["_run_key"].isin(allowed_run_keys)].drop(columns=["_run_key"])
    return filtered_df.reset_index(drop=True)


def build_summary(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successful evaluations per model."""

    if detailed_df.empty or not detailed_df["evaluation_success"].fillna(False).astype(bool).any():
        return pd.DataFrame()

    success_df = detailed_df[detailed_df["evaluation_success"].fillna(False).astype(bool)].copy()
    all_metric_cols = set()
    for ev in EVALUATORS.values():
        all_metric_cols.update(ev.metric_names)

    summary_df = (
        success_df.groupby(["model", "notebook_order"])
        .agg(
            total_evaluations=("notebook_id", "count"),
            successful_evaluations=("evaluation_success", "sum"),
            **{f"avg_{col}": (col, "mean") for col in all_metric_cols if col in success_df.columns},
        )
        .reset_index()
    )

    sort_col = "avg_accuracy" if "avg_accuracy" in summary_df.columns else summary_df.columns[-1]
    return summary_df.sort_values(sort_col, ascending=False)


def merge_results(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new evaluation rows into the detailed results, replacing matching run keys."""

    if existing_df.empty:
        return new_df.reset_index(drop=True)
    if new_df.empty:
        return existing_df.reset_index(drop=True)

    combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    combined_df = combined_df.drop_duplicates(subset=RUN_KEY_COLUMNS, keep="last")
    return combined_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Notebook-specific evaluators
# ---------------------------------------------------------------------------

class NotebookEvaluator:
    """Base class. Subclasses define how to load test data, run inference, and score."""

    notebook_id: str = ""
    metric_names: list[str] = ["accuracy"]

    def load_test_data(self, data_dir: Path) -> tuple[list[Any], list[Any]]:
        """Return (inputs, labels) for evaluation."""
        raise NotImplementedError

    def load_model(self, model_dir: Path) -> Any:
        """Load model artifacts once. Return model handle."""
        raise NotImplementedError

    def predict(self, model: Any, inputs: list[Any]) -> list[Any]:
        """Run inference on all inputs. Return list of predictions."""
        raise NotImplementedError

    def compute_metrics(self, predictions: list, labels: list) -> dict[str, float]:
        """Compute evaluation metrics."""
        raise NotImplementedError


class SimpleTextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 66, num_classes: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(128, num_classes, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.drop(x)
        return self.linear(x)


class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        import torch

        e = self.emb(x)
        _, h = self.rnn(e)
        return torch.sigmoid(self.fc(h[-1]))


class EmotionCNNCompat(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 66, num_classes: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear(x)


class Permute(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


def extract_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def resolve_state_dict(obj: Any) -> OrderedDict[str, Any] | None:
    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], (dict, OrderedDict)):
        return OrderedDict(obj["model_state_dict"])
    if isinstance(obj, (dict, OrderedDict)) and obj:
        first_value = next(iter(obj.values()))
        if hasattr(first_value, "shape"):
            return OrderedDict(obj)
    if hasattr(obj, "state_dict"):
        return OrderedDict(obj.state_dict())
    return None


def build_nb8_cnn_from_state_dict(state_dict: OrderedDict[str, Any]) -> EmotionCNNCompat:
    embedding_key = next(
        (key for key in state_dict if key.endswith("embedding.weight") or key.endswith("emb.weight")),
        None,
    )
    conv_key = next((key for key in state_dict if key.endswith("conv.weight")), None)
    linear_key = next(
        (key for key in state_dict if key.endswith("linear.weight") or key.endswith("fc.weight")),
        None,
    )

    if embedding_key is None:
        embedding_key = next(
            (
                key
                for key, value in state_dict.items()
                if getattr(value, "ndim", 0) == 2 and value.shape[0] > value.shape[1]
            ),
            None,
        )
    if conv_key is None:
        conv_key = next((key for key, value in state_dict.items() if getattr(value, "ndim", 0) == 3), None)
    if linear_key is None and conv_key is not None:
        conv_out_channels = state_dict[conv_key].shape[0]
        linear_key = next(
            (
                key
                for key, value in state_dict.items()
                if getattr(value, "ndim", 0) == 2 and value.shape[1] == conv_out_channels and key != embedding_key
            ),
            None,
        )

    if embedding_key is None or conv_key is None or linear_key is None:
        raise ValueError(f"Unsupported nb8 state_dict keys: {list(state_dict.keys())[:5]}")

    vocab_size, embedding_dim = state_dict[embedding_key].shape
    hidden_dim = state_dict[conv_key].shape[0]
    num_classes = state_dict[linear_key].shape[0]
    model = EmotionCNNCompat(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
    )

    remapped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if new_key.endswith("emb.weight"):
            new_key = new_key[: -len("emb.weight")] + "embedding.weight"
        elif key == embedding_key:
            new_key = "embedding.weight"
        elif key == conv_key:
            new_key = "conv.weight"
        elif key == conv_key.replace("weight", "bias"):
            new_key = "conv.bias"
        elif key == linear_key:
            new_key = "linear.weight"
        elif key == linear_key.replace("weight", "bias"):
            new_key = "linear.bias"
        elif new_key.endswith("fc.weight"):
            new_key = new_key[: -len("fc.weight")] + "linear.weight"
        elif new_key.endswith("fc.bias"):
            new_key = new_key[: -len("fc.bias")] + "linear.bias"
        elif new_key.endswith("drop.p"):
            continue
        remapped_state_dict[new_key] = value

    model.load_state_dict(remapped_state_dict, strict=False)
    model.eval()
    return model


def remap_nb6_state_dict(state_dict: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
    """Normalize alternate nb6 classifier wrapper prefixes to the evaluator's Model class."""

    remapped = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("bert_lora."):
            new_key = "bert_lora_model." + new_key[len("bert_lora."):]
        remapped[new_key] = value
    return remapped


class EvaluatorNb1(NotebookEvaluator):
    """nb1: Binary classification — True_Correct (1) vs True_Neither (0)."""

    notebook_id = "nb1"
    metric_names = ["accuracy", "f1"]

    def load_test_data(self, data_dir: Path):
        from src.inference.strategies import format_input

        df = pd.read_csv(data_dir / "input" / "map-charting-student-math-misunderstandings" / "train.csv")
        df["Misconception"] = df["Misconception"].fillna("NA")

        # Determine correct answers
        idx = df.apply(lambda row: row.Category.split("_")[0], axis=1) == "True"
        correct = df.loc[idx].copy()
        correct["c"] = correct.groupby(["QuestionId", "MC_Answer"]).MC_Answer.transform("count")
        correct = correct.sort_values("c", ascending=False).drop_duplicates(["QuestionId"])
        correct = correct[["QuestionId", "MC_Answer"]]
        correct["is_correct"] = 1

        df = df.merge(correct, on=["QuestionId", "MC_Answer"], how="left")
        df["is_correct"] = df["is_correct"].fillna(0)

        # Labels: 1 = True_Correct, 0 = True_Neither
        labels = (df["Category"] == "True_Correct").astype(int).tolist()
        inputs = df.apply(format_input, axis=1).tolist()
        return inputs, labels

    def load_model(self, model_dir: Path):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

        path_obj = Path(model_dir)
        tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
        model_paths = list(path_obj.rglob("config.json"))

        if not model_paths:
            raise ValueError(f"No config.json found in {model_dir}")

        model_path = model_paths[0].parent
        tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else model_path

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception:
            model = AutoModel.from_pretrained(model_path)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}

    def predict(self, model_handle: Any, inputs: list[Any]) -> list[Any]:
        import torch

        tokenizer = model_handle["tokenizer"]
        model = model_handle["model"]
        predictions = []
        for text in inputs:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(**enc)
            if hasattr(out, "logits"):
                pred = out.logits.argmax(dim=-1).item()
            else:
                pred = 0
            predictions.append(pred)
        return predictions

    def compute_metrics(self, predictions, labels):
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="binary", zero_division=0)
        return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


class EvaluatorNb4(NotebookEvaluator):
    """nb4: Binary toxic comment classification."""

    notebook_id = "nb4"
    metric_names = ["accuracy", "f1"]

    def load_test_data(self, data_dir: Path):
        df = pd.read_csv(data_dir / "input" / "jigsaw-toxic-comment-classification-challenge" / "train.csv")
        # Binary: toxic if any label is 1
        toxic_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        labels = (df[toxic_cols].sum(axis=1) > 0).astype(int).tolist()
        inputs = df["comment_text"].tolist()
        return inputs, labels

    def load_model(self, model_dir: Path):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

        path_obj = Path(model_dir)
        tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
        model_paths = list(path_obj.rglob("config.json"))

        if not model_paths:
            raise ValueError(f"No config.json found in {model_dir}")

        model_path = model_paths[0].parent
        tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else model_path

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception:
            model = AutoModel.from_pretrained(model_path)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}

    def predict(self, model_handle: Any, inputs: list[Any]) -> list[Any]:
        import torch

        tokenizer = model_handle["tokenizer"]
        model = model_handle["model"]
        predictions = []
        for text in inputs:
            enc = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = extract_logits(model(**enc))
            if hasattr(logits, "ndim") and logits.ndim == 2 and logits.shape[-1] > 2:
                pred = int(torch.sigmoid(logits).gt(0.5).any(dim=-1).item())
            elif hasattr(logits, "ndim") and logits.ndim == 2 and logits.shape[-1] == 1:
                pred = int(torch.sigmoid(logits).gt(0.5).item())
            else:
                pred = logits.argmax(dim=-1).item() if hasattr(logits, "argmax") else 0
            predictions.append(pred)
        return predictions

    def compute_metrics(self, predictions, labels):
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="binary", zero_division=0)
        return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


class EvaluatorNb5(NotebookEvaluator):
    """nb5: IMDB binary sentiment classification (LSTM)."""

    notebook_id = "nb5"
    metric_names = ["accuracy", "f1"]

    def load_test_data(self, data_dir: Path):
        df = pd.read_csv(data_dir / "input" / "imdb-dataset-of-50k-movie-reviews" / "IMDB Dataset.csv")
        labels = (df["sentiment"] == "positive").astype(int).tolist()
        inputs = df["review"].tolist()
        return inputs, labels

    def load_model(self, model_dir: Path):
        return load_nb5_inference_bundle(model_dir)

    def _encode(self, text: str, vocab: dict) -> list[int]:
        tokens = text.split()
        ids = [2]  # <START>
        for t in tokens:
            ids.append(vocab.get(t, 1))  # <UNK>
        ids.append(3)  # <END>
        return ids

    def predict(self, model_handle: Any, inputs: list[Any]) -> list[Any]:
        import torch

        vocab = model_handle["vocab"]
        model = model_handle["model"]
        predictions = []
        for text in inputs:
            encoded = self._encode(str(text), vocab)
            tensor = torch.tensor([encoded], dtype=torch.long)
            with torch.no_grad():
                out = model(tensor)
            # Sigmoid output: >0.5 = positive (1)
            pred = 1 if out.item() > 0.5 else 0
            predictions.append(pred)
        return predictions

    def compute_metrics(self, predictions, labels):
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="binary", zero_division=0)
        return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


class EvaluatorNb2(NotebookEvaluator):
    """nb2: Multi-output regression (6 essay scoring dimensions)."""

    notebook_id = "nb2"
    metric_names = ["rmse", "mae"]

    def load_test_data(self, data_dir: Path):
        df = pd.read_csv(data_dir / "input" / "feedback-prize-english-language-learning" / "train.csv")
        target_cols = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
        labels = df[target_cols].values.tolist()  # list of 6-element lists
        inputs = df["full_text"].tolist()
        return inputs, labels

    def load_model(self, model_dir: Path):
        from catboost import CatBoostRegressor
        import joblib

        path_obj = Path(model_dir)

        # Load meta model and sub-models
        meta_models = list(path_obj.rglob("meta_model*.cbm"))
        model2_files = list(path_obj.rglob("model2*.cbm"))
        model1_files = list(path_obj.rglob("model1*.pkl"))
        model3_files = list(path_obj.rglob("model3*.pkl"))
        model4_files = list(path_obj.rglob("model4*.pkl"))
        vectorizer_files = list(path_obj.rglob("vectorizer*.pkl"))
        normalizer_files = list(path_obj.rglob("normalizer*.pkl"))

        if not meta_models:
            raise ValueError(f"No meta_model*.cbm found in {model_dir}")

        handle = {}
        handle["meta_model"] = CatBoostRegressor()
        handle["meta_model"].load_model(str(meta_models[0]))

        if model2_files:
            handle["model2"] = CatBoostRegressor()
            handle["model2"].load_model(str(model2_files[0]))

        if model1_files:
            handle["model1"] = joblib.load(model1_files[0])
        if model3_files:
            handle["model3"] = joblib.load(model3_files[0])
        if model4_files:
            handle["model4"] = joblib.load(model4_files[0])
        if vectorizer_files:
            handle["vectorizer"] = joblib.load(vectorizer_files[0])
        if normalizer_files:
            handle["normalizer"] = joblib.load(normalizer_files[0])

        return handle

    def predict(self, model_handle: Any, inputs: list[Any]) -> list[Any]:
        from src.inference.strategies import generate_sentiment_score
        from scipy.sparse import hstack
        import numpy as np

        vectorizer = model_handle.get("vectorizer")
        normalizer = model_handle.get("normalizer")
        meta_model = model_handle["meta_model"]
        model1 = model_handle.get("model1")
        model2 = model_handle.get("model2")
        model3 = model_handle.get("model3")
        model4 = model_handle.get("model4")

        if not all([vectorizer, normalizer, model1, model2, model3, model4]):
            raise ValueError("Missing required sub-models for nb2 evaluation")

        predictions = []
        for text in inputs:
            comp, neg, pos, neu = generate_sentiment_score(str(text))
            text_len = len(str(text).split())

            X_tfidf = vectorizer.transform([str(text)])
            numeric_row = np.array([[comp, neg, pos, neu, text_len]], dtype=float)

            normalizer_feature_count = getattr(normalizer, "n_features_in_", None)
            if normalizer_feature_count == numeric_row.shape[1]:
                X_num = normalizer.transform(numeric_row)
                X_final = hstack((X_tfidf, X_num))
            else:
                X_com = normalizer.transform([[comp]])
                X_neg = normalizer.transform([[neg]])
                X_pos = normalizer.transform([[pos]])
                X_neu = normalizer.transform([[neu]])
                X_len = normalizer.transform([[text_len]])
                X_final = hstack((X_tfidf, X_com, X_neg, X_pos, X_neu, X_len))

            pred1 = model1.predict(X_final)
            pred2 = model2.predict(X_final)
            pred3 = model3.predict(X_final)
            pred4 = model4.predict(X_final)
            pred_stack = np.column_stack([pred1, pred2, pred3, pred4])
            result = meta_model.predict(pred_stack)
            if hasattr(result, "flatten"):
                predictions.append(float(result.flatten()[0]))
            else:
                predictions.append(float(result[0]))
        return predictions

    def compute_metrics(self, predictions, labels):
        # For regression, compare predicted scalar (avg) vs mean of 6 targets
        label_means = [np.mean(l) for l in labels]
        pred_arr = np.array(predictions)
        label_arr = np.array(label_means)
        rmse = float(np.sqrt(np.mean((pred_arr - label_arr) ** 2)))
        mae = float(np.mean(np.abs(pred_arr - label_arr)))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4)}


class EvaluatorNb6(NotebookEvaluator):
    """nb6: Binary fake-vs-real article classification from paired documents."""

    notebook_id = "nb6"
    metric_names = ["accuracy", "f1"]

    def load_test_data(self, data_dir: Path):
        base_dir = data_dir / "input" / "fake-or-real-the-impostor-hunt" / "data"
        labels_df = pd.read_csv(base_dir / "train.csv")

        article_dirs = sorted((base_dir / "train").glob("article_*"))
        texts = []
        labels = []
        for article_dir, real_text_id in zip(article_dirs, labels_df["real_text_id"].tolist()):
            file_1 = (article_dir / "file_1.txt").read_text(encoding="utf-8")
            file_2 = (article_dir / "file_2.txt").read_text(encoding="utf-8")
            texts.extend([file_1, file_2])
            labels.extend([1 if real_text_id == 1 else 0, 1 if real_text_id == 2 else 0])
        return texts, labels

    def load_model(self, model_dir: Path):
        return load_nb6_inference_bundle(model_dir)

    def predict(self, model_handle: Any, inputs: list[Any]) -> list[Any]:
        import torch

        tokenizer = model_handle["tokenizer"]
        model = model_handle["model"]
        predictions = []
        for text in inputs:
            enc = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            predictions.append(out.argmax(dim=-1).item())
        return predictions

    def compute_metrics(self, predictions, labels):
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="binary", zero_division=0)
        return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


class EvaluatorNb7(NotebookEvaluator):
    """nb7: Product retrieval via sentence embeddings, scored by nearest-neighbor category match."""

    notebook_id = "nb7"
    metric_names = ["accuracy", "f1"]

    def load_test_data(self, data_dir: Path):
        df = pd.read_csv(
            data_dir / "input" / "amazon-eco-friendly-products-dataset" / "amazon_eco-friendly_products.csv"
        )
        text_cols = ["title", "name", "category", "material", "brand", "description"]
        combined = df[text_cols].fillna("").agg(" ".join, axis=1).str.strip().tolist()
        labels = df["category"].fillna("unknown").tolist()
        return combined, labels

    def load_model(self, model_dir: Path):
        import pickle

        model_paths = list(Path(model_dir).rglob("sentence_transformer_model.pkl"))
        if not model_paths:
            raise ValueError(f"No sentence_transformer_model.pkl found in {model_dir}")
        with open(model_paths[0], "rb") as handle:
            model = pickle.load(handle)
        return model

    def predict(self, model: Any, inputs: list[Any]) -> list[Any]:
        from sklearn.metrics.pairwise import cosine_similarity

        embeddings = model.encode(inputs, convert_to_numpy=True)
        sims = cosine_similarity(embeddings)
        np.fill_diagonal(sims, -np.inf)
        nearest_indices = sims.argmax(axis=1)
        return nearest_indices.tolist()

    def compute_metrics(self, predictions, labels):
        from sklearn.metrics import accuracy_score, f1_score

        predicted_labels = [labels[index] for index in predictions]
        acc = accuracy_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels, average="weighted", zero_division=0)
        return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


class EvaluatorNb8(NotebookEvaluator):
    """nb8: Multi-class emotion classification."""

    notebook_id = "nb8"
    metric_names = ["accuracy", "f1"]

    def __init__(self):
        self.label_map: dict[str, int] | None = None

    def load_test_data(self, data_dir: Path):
        train_df = pd.read_csv(
            data_dir / "input" / "emotions-dataset-for-nlp" / "train.txt",
            sep=";",
            header=None,
            names=["text", "emotion"],
        )
        val_df = pd.read_csv(
            data_dir / "input" / "emotions-dataset-for-nlp" / "val.txt",
            sep=";",
            header=None,
            names=["text", "emotion"],
        )
        label_names = sorted(pd.concat([train_df["emotion"], val_df["emotion"]]).unique().tolist())
        self.label_map = {label: index for index, label in enumerate(label_names)}
        inputs = val_df["text"].astype(str).tolist()
        labels = val_df["emotion"].map(self.label_map).tolist()
        return inputs, labels

    def load_model(self, model_dir: Path):
        return load_nb8_inference_bundle(model_dir)

    def predict(self, model_handle: Any, inputs: list[Any]) -> list[Any]:
        import torch

        tokenizer = model_handle["tokenizer"]
        model = model_handle["model"]
        predictions = []
        for text in inputs:
            enc = tokenizer(
                str(text),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=90,
            )
            with torch.no_grad():
                if model_handle["type"] == "hf":
                    logits = extract_logits(model(**enc))
                else:
                    logits = model(enc["input_ids"])
            predictions.append(logits.argmax(dim=-1).item())
        return predictions

    def compute_metrics(self, predictions, labels):
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
        return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


class EvaluatorNb9(NotebookEvaluator):
    """nb9: Binary recommendation classification from clothing reviews."""

    notebook_id = "nb9"
    metric_names = ["accuracy", "f1"]

    def load_test_data(self, data_dir: Path):
        df = pd.read_csv(data_dir / "input" / "Womens Clothing E-Commerce Reviews.csv")
        inputs = df["Review Text"].fillna("").astype(str).tolist()
        labels = df["Recommended IND"].astype(int).tolist()
        return inputs, labels

    def load_model(self, model_dir: Path):
        return load_nb9_inference_bundle(model_dir)

    def predict(self, model_handle: Any, inputs: list[Any]) -> list[Any]:
        import torch

        tokenizer = model_handle["tokenizer"]
        model = model_handle["model"]
        predictions = []
        for text in inputs:
            enc = tokenizer(str(text), return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            with torch.no_grad():
                if model_handle["type"] == "hf":
                    out = model(**enc)
                else:
                    out = model(enc["input_ids"], attention_mask=enc["attention_mask"])
                logits = out.logits if hasattr(out, "logits") else out
            predictions.append(logits.argmax(dim=-1).item())
        return predictions

    def compute_metrics(self, predictions, labels):
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="binary", zero_division=0)
        return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


class EvaluatorNb3(NotebookEvaluator):
    """nb3: Text classification (feedback-prize-2021 discourse type)."""

    notebook_id = "nb3"
    metric_names = ["accuracy", "f1"]

    LABEL_MAP = {
        "Lead": 0, "Position": 1, "Claim": 2, "Evidence": 3,
        "Counterclaim": 4, "Rebuttal": 5, "Concluding Statement": 6,
    }

    def load_test_data(self, data_dir: Path):
        df = pd.read_csv(data_dir / "input" / "feedback-prize-2021" / "train.csv")
        labels = df["discourse_type"].map(self.LABEL_MAP).tolist()
        inputs = df["discourse_text"].tolist()
        return inputs, labels

    def load_model(self, model_dir: Path):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

        path_obj = Path(model_dir)
        tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
        model_paths = list(path_obj.rglob("config.json"))

        if not model_paths:
            raise ValueError(f"No config.json found in {model_dir}")

        model_path = model_paths[0].parent
        tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else model_path

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception:
            model = AutoModel.from_pretrained(model_path)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}

    def predict(self, model_handle: Any, inputs: list[Any]) -> list[Any]:
        import torch

        tokenizer = model_handle["tokenizer"]
        model = model_handle["model"]
        predictions = []
        for text in inputs:
            enc = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(**enc)
            if hasattr(out, "logits"):
                pred = out.logits.argmax(dim=-1).item()
            else:
                pred = 0
            predictions.append(pred)
        return predictions

    def compute_metrics(self, predictions, labels):
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
        return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


class EvaluatorNb10(NotebookEvaluator):
    """nb10: Medical NER / sequence classification."""

    notebook_id = "nb10"
    metric_names = ["accuracy", "f1"]

    def load_test_data(self, data_dir: Path):
        df = pd.read_csv(data_dir / "input" / "med-train" / "train.csv")
        # Group by sentence and use first sentence as test
        sentences = df.groupby(["Doc_ID", "Sent_ID"])["Word"].apply(" ".join).tolist()
        # For this small dataset, use dummy binary labels (O vs non-O)
        tag_groups = df.groupby(["Doc_ID", "Sent_ID"])["tag"].apply(
            lambda x: 1 if any(t != "O" for t in x) else 0
        ).tolist()
        return sentences, tag_groups

    def load_model(self, model_dir: Path):
        return load_nb10_inference_bundle(model_dir)

    def predict(self, model_handle: Any, inputs: list[Any]) -> list[Any]:
        import torch

        tokenizer = model_handle["tokenizer"]
        model = model_handle["model"]
        tag2idx = model_handle.get("tag2idx") or {}
        o_label_idx = tag2idx.get("O", 0)
        pad_label_idx = tag2idx.get("PAD")
        predictions = []
        for text in inputs:
            enc = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=128, padding="max_length")
            with torch.no_grad():
                out = model(**enc) if model_handle["type"] == "hf" else model(enc["input_ids"], attention_mask=enc["attention_mask"])
                logits = extract_logits(out)

            if hasattr(logits, "ndim") and logits.ndim == 3:
                token_preds = logits.argmax(dim=-1)
                valid_mask = enc["attention_mask"].bool()
                valid_token_preds = token_preds[valid_mask]
                pred = int(any(
                    token_id != o_label_idx and (pad_label_idx is None or token_id != pad_label_idx)
                    for token_id in valid_token_preds.tolist()
                ))
            elif hasattr(logits, "ndim") and logits.ndim == 2:
                pred = logits.argmax(dim=-1).item()
            else:
                pred = logits.argmax(dim=-1).item() if hasattr(logits, "argmax") else 0
            predictions.append(pred)
        return predictions

    def compute_metrics(self, predictions, labels):
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="binary", zero_division=0)
        return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EVALUATORS: dict[str, NotebookEvaluator] = {
    "nb1": EvaluatorNb1(),
    "nb2": EvaluatorNb2(),
    "nb3": EvaluatorNb3(),
    "nb4": EvaluatorNb4(),
    "nb5": EvaluatorNb5(),
    "nb6": EvaluatorNb6(),
    "nb7": EvaluatorNb7(),
    "nb8": EvaluatorNb8(),
    "nb9": EvaluatorNb9(),
    "nb10": EvaluatorNb10(),
}


# ---------------------------------------------------------------------------
# Reference model evaluation (baseline)
# ---------------------------------------------------------------------------

def evaluate_reference(evaluator: NotebookEvaluator, data_dir: Path) -> dict[str, float] | None:
    """Evaluate the reference (ground-truth) model as a baseline."""
    ref_dir = data_dir / "output"
    if not ref_dir.exists():
        return None
    try:
        inputs, labels = evaluator.load_test_data(data_dir)
        model = evaluator.load_model(ref_dir)
        predictions = evaluator.predict(model, inputs)
        return evaluator.compute_metrics(predictions, labels)
    except Exception as e:
        write_progress_log("WARNING", f"Reference evaluation failed for {evaluator.notebook_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    project_root: Path,
    output_path: Path,
    existing_results: pd.DataFrame | None = None,
    target_nb: str | None = None,
    target_runner: str | None = None,
    target_model: str | None = None,
    target_complexity: int | None = None,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing_results = existing_results.copy() if existing_results is not None else pd.DataFrame()
    records = []
    reference_cache: dict[str, dict | None] = {}
    prior_successes = successful_run_keys(existing_results)

    all_candidate_runs = list(
        iter_model_dirs(
            output_path,
            target_nb=target_nb,
            target_runner=target_runner,
            target_model=target_model,
            target_complexity=target_complexity,
        )
    )
    model_runs = []
    skipped_runs = 0
    for nb_id, runner_name, model_dir, meta in all_candidate_runs:
        run_key = (nb_id, runner_name, meta["model"], meta["complexity"], meta["notebook_order"], meta["run"])
        if run_key in prior_successes:
            skipped_runs += 1
            continue
        model_runs.append((nb_id, runner_name, model_dir, meta))

    if skipped_runs:
        logger.info("Skipping %s runs already marked evaluation_success=True in existing detailed results.", skipped_runs)

    for nb_id, runner_name, model_dir, meta in tqdm(
        model_runs,
        desc="Evaluating models",
        unit="run",
    ):
        if nb_id not in EVALUATORS:
            continue

        evaluator = EVALUATORS[nb_id]
        data_dir = project_root / "data" / nb_id

        record = {
            "notebook_id": nb_id,
            "runner": runner_name,
            "model": meta["model"],
            "complexity": meta["complexity"],
            "notebook_order": meta["notebook_order"],
            "run": meta["run"],
            "model_dir": str(model_dir),
            "evaluation_success": False,
            "error": "",
        }
        # Initialize metric columns
        for m in evaluator.metric_names:
            record[m] = None

        # Compute reference baseline (cached)
        if nb_id not in reference_cache:
            reference_cache[nb_id] = evaluate_reference(evaluator, data_dir)
        ref_metrics = reference_cache[nb_id]
        for m in evaluator.metric_names:
            record[f"reference_{m}"] = ref_metrics.get(m) if ref_metrics else None

        try:
            inputs, labels = evaluator.load_test_data(data_dir)
            if max_samples and len(inputs) > max_samples:
                inputs = inputs[:max_samples]
                labels = labels[:max_samples]

            model_handle = evaluator.load_model(model_dir)
            predictions = evaluator.predict(model_handle, inputs)
            metrics = evaluator.compute_metrics(predictions, labels)

            record["evaluation_success"] = True
            for m in evaluator.metric_names:
                record[m] = metrics.get(m)

            write_progress_log("INFO", f"  {nb_id}/{runner_name}/{model_dir.name}: {metrics}")
        except Exception as e:
            record["error"] = str(e)[:200]
            write_progress_log("WARNING", f"  {nb_id}/{runner_name}/{model_dir.name}: FAILED - {e}")

        records.append(record)

    new_results_df = pd.DataFrame(records)
    detailed_df = merge_results(existing_results, new_results_df)
    summary_df = build_summary(detailed_df)

    return detailed_df, summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate real ML model performance of LLM-generated artifacts."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--notebook", type=str, default=None, help="Evaluate only this notebook (e.g. nb1)")
    parser.add_argument("--runner", type=str, default=None, help="Filter by runner (e.g. simple)")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name")
    parser.add_argument("--complexity", type=int, default=None, help="Filter by complexity level")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples per notebook")
    parser.add_argument("--detailed-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    project_root = args.project_root.resolve()
    output_path = (project_root / args.output_dir).resolve()
    detailed_path = args.detailed_output or output_path / "model_performance.csv"
    summary_path = args.summary_output or output_path / "model_performance_summary.csv"

    logger.info(f"Project root: {project_root}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Evaluating notebooks: {list(EVALUATORS.keys())}")

    existing_results = load_existing_results(detailed_path)
    if not existing_results.empty:
        logger.info("Loaded %s existing detailed evaluation rows from %s", len(existing_results), detailed_path)
        globally_allowed_runs = load_successful_train_runs(output_path)
        filtered_results = filter_results_to_allowed_runs(existing_results, globally_allowed_runs)
        removed_rows = len(existing_results) - len(filtered_results)
        if removed_rows:
            logger.info(
                "Dropping %s cached detailed rows whose runs no longer satisfy train/artifact match criteria.",
                removed_rows,
            )
        existing_results = filtered_results

    detailed_df, summary_df = run_evaluation(
        project_root=project_root,
        output_path=output_path,
        existing_results=existing_results,
        target_nb=args.notebook,
        target_runner=args.runner,
        target_model=args.model,
        target_complexity=args.complexity,
        max_samples=args.max_samples,
    )

    detailed_df.to_csv(detailed_path, index=False)
    logger.info(f"Detailed results: {detailed_path}")

    if not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary results: {summary_path}")
        print("\n=== Model Performance Summary (sorted by avg accuracy) ===")
        print(summary_df.to_string(index=False))
    else:
        print("No successful evaluations.")

    # Print per-notebook success rates
    if not detailed_df.empty:
        print("\n=== Per-Notebook Evaluation Success ===")
        nb_stats = detailed_df.groupby("notebook_id").agg(
            total=("evaluation_success", "count"),
            success=("evaluation_success", "sum"),
        ).reset_index()
        nb_stats["success_rate"] = (nb_stats["success"] / nb_stats["total"]).round(3)
        print(nb_stats.to_string(index=False))


if __name__ == "__main__":
    main()
