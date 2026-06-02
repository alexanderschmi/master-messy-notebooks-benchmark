import warnings
warnings.filterwarnings("ignore")
from abc import ABC, abstractmethod
from collections import OrderedDict
from transformers import AutoTokenizer
from pathlib import Path
import logging
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from typing import Any

logger = logging.getLogger(__name__)

# 1. Define the Interface (The Contract)
class InferenceStrategy(ABC):
    @abstractmethod
    def get_prompt(self):
        """Return a dummy prompt/feature for inference."""
        pass

    @abstractmethod
    def testInference(self, path: str | Path, prompt):
        """Run the actual inference using the model artifacts."""
        pass


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        import torch

        e = self.emb(x)
        out, (h, c) = self.lstm(e)
        return torch.sigmoid(self.fc(h[-1]))


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
    remapped = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("bert_lora."):
            new_key = "bert_lora_model." + new_key[len("bert_lora."):]
        remapped[new_key] = value
    return remapped


def load_nb5_inference_bundle(path: str | Path) -> dict[str, Any]:
    import __main__
    import json
    import torch

    path_obj = Path(path)
    vocab_files = [f for f in path_obj.rglob("*.json") if "vocab" in f.name]
    model_files = list(path_obj.rglob("*.pt"))

    if not model_files:
        raise ValueError(f"Missing model.pt in {path}")

    __main__.LSTMClassifier = LSTMClassifier
    __main__.RNNClassifier = SimpleRNNClassifier
    __main__.SentimentClassifier = LSTMClassifier
    __main__.SentimentLSTM = LSTMClassifier

    obj = torch.load(model_files[0], map_location="cpu", weights_only=False)

    if hasattr(obj, "eval"):
        model = obj
    elif isinstance(obj, dict):
        state_dict = obj.get("model_state_dict", obj)
        vocab_size = obj.get("vocab_size") if isinstance(obj.get("vocab_size"), int) else None
        model_type = str(obj.get("model_type", "lstm")).lower()

        if vocab_size is None:
            if "emb.weight" in state_dict:
                vocab_size = state_dict["emb.weight"].shape[0]
            elif "embedding.weight" in state_dict:
                vocab_size = state_dict["embedding.weight"].shape[0]
        if vocab_size is None:
            emb_keys = [k for k in state_dict.keys() if "emb" in k.lower() and "weight" in k.lower()]
            if emb_keys:
                vocab_size = state_dict[emb_keys[0]].shape[0]
        if vocab_size is None:
            raise ValueError(f"Cannot determine vocab_size from state_dict keys: {list(obj.keys())[:5]}")

        if "rnn" in model_type and "lstm" not in model_type:
            model = SimpleRNNClassifier(vocab_size)
        elif any("rnn." in key.lower() for key in state_dict.keys()) and not any(
            "lstm." in key.lower() for key in state_dict.keys()
        ):
            model = SimpleRNNClassifier(vocab_size)
        else:
            model = LSTMClassifier(vocab_size)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unexpected model format: {type(obj)}")

    model.eval()

    vocab = {}
    if vocab_files:
        with open(vocab_files[0], "r") as f:
            vocab = json.load(f)
    elif "emb.weight" in (obj if isinstance(obj, dict) else {}):
        raise ValueError("No vocab.json found for encoding")

    return {"vocab": vocab, "model": model}


def load_nb6_inference_bundle(path: str | Path) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    path_obj = Path(path)
    tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
    model_paths = list(path_obj.rglob("*.pth"))
    if not model_paths:
        raise ValueError(f"No .pth model found in {path}")

    tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else path_obj
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    base_model = AutoModel.from_pretrained("gaunernst/bert-mini-uncased")
    lora_model = get_peft_model(base_model, lora_config)
    model = Model(lora_model, 256, 256, 2)
    state_dict = torch.load(model_paths[0], map_location="cpu")
    if isinstance(state_dict, dict):
        state_dict = remap_nb6_state_dict(OrderedDict(state_dict))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


def load_nb8_inference_bundle(path: str | Path) -> dict[str, Any]:
    import __main__
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    __main__.EmotionCNN = EmotionCNNCompat
    __main__.TextCNN = EmotionCNNCompat
    __main__.CNNClassifier = EmotionCNNCompat
    __main__.CNNTextClassifier = EmotionCNNCompat
    __main__.Permute = Permute
    __main__.Transpose = Transpose

    path_obj = Path(path)
    tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
    pt_paths = list(path_obj.rglob("model.pt"))
    config_paths = list(path_obj.rglob("config.json"))

    tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else path_obj
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if config_paths:
        model = AutoModelForSequenceClassification.from_pretrained(config_paths[0].parent)
        model.eval()
        return {"tokenizer": tokenizer, "model": model, "type": "hf"}

    if not pt_paths:
        raise ValueError(f"No model.pt or config.json found in {path}")

    obj = torch.load(pt_paths[0], map_location="cpu", weights_only=False)
    state_dict = resolve_state_dict(obj)
    if state_dict is not None:
        model = build_nb8_cnn_from_state_dict(state_dict)
    elif hasattr(obj, "eval"):
        model = obj
    else:
        raise ValueError(f"Unexpected model format: {type(obj)}")

    model.eval()
    return {"tokenizer": tokenizer, "model": model, "type": "pt"}


def load_nb9_inference_bundle(path: str | Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path_obj = Path(path)
    tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
    config_paths = list(path_obj.rglob("config.json"))
    pt_paths = list(path_obj.rglob("model.pt"))

    if not tokenizer_paths:
        raise ValueError(f"No tokenizer_config.json found in {path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_paths[0].parent)

    if config_paths:
        model = AutoModelForSequenceClassification.from_pretrained(config_paths[0].parent)
        model.eval()
        return {"tokenizer": tokenizer, "model": model, "type": "hf"}

    if not pt_paths:
        raise ValueError(f"No model.pt or config.json found in {path}")

    obj = torch.load(pt_paths[0], map_location="cpu", weights_only=False)
    if hasattr(obj, "eval"):
        model = obj
    else:
        raise ValueError(f"Unexpected nb9 model format: {type(obj)}")
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "type": "pt"}


def load_nb10_inference_bundle(path: str | Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    path_obj = Path(path)
    tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
    model_paths = list(path_obj.rglob("*.pt"))

    if not model_paths:
        config_paths = list(path_obj.rglob("config.json"))
        if not config_paths:
            raise ValueError(f"No model found in {path}")
        model_path = config_paths[0].parent
        tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.eval()
        return {"tokenizer": tokenizer, "model": model, "type": "hf", "tag2idx": None}

    tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else path_obj
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    obj = torch.load(model_paths[0], map_location="cpu", weights_only=False)

    tag2idx = obj.get("tag2idx") if isinstance(obj, dict) else None
    state_dict = resolve_state_dict(obj)
    if state_dict is not None:
        num_labels = None
        classifier_key = next((key for key in state_dict if key.endswith("classifier.weight")), None)
        if classifier_key is not None:
            num_labels = state_dict[classifier_key].shape[0]
        elif tag2idx:
            num_labels = len(tag2idx)
        if num_labels is None:
            raise ValueError(f"Cannot infer num_labels for nb10 model from {path}")

        model = AutoModelForTokenClassification.from_pretrained(
            "google/bert_uncased_L-2_H-128_A-2",
            num_labels=num_labels,
        )
        model.load_state_dict(state_dict, strict=False)
    elif hasattr(obj, "eval"):
        model = obj
    else:
        raise ValueError(f"Unexpected model format: {type(obj)}")

    model.eval()
    return {"tokenizer": tokenizer, "model": model, "type": "pt", "tag2idx": tag2idx}

def generate_sentiment_score(sentence):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    sentence_sentiment_score = sid.polarity_scores(sentence)
    neg=sentence_sentiment_score['neg']
    pos=sentence_sentiment_score['pos']
    neu=sentence_sentiment_score['neu']
    comp=sentence_sentiment_score['compound']
    return comp,neg,pos,neu

def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

def clean_text(text):
    """
    Clean the input text by removing HTML tags, special characters, and normalizing whitespace.
    """
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s&\'%-]', '', text)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text


class Model(nn.Module):
    def __init__(self, lora_model, hidden_dim, new_hidden_dim, num_class):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, new_hidden_dim)
        self.classifier = nn.Linear(new_hidden_dim, num_class)
        self.bert_lora_model = lora_model

    def forward(self, input_ids, attention_mask):
        output_states = self.bert_lora_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        cls_embedding = output_states.last_hidden_state[:, 0, :]
        pooled_output = self.dense(cls_embedding.float())
        logits = self.classifier(pooled_output)
        return logits


# 2. Implementations
class InferenceTransformers(InferenceStrategy):
    def get_prompt(self):
        return "This is a dummy test string for NLP models."

    def testInference(self, path: str | Path, prompt):
        import torch
        from pathlib import Path

        # Find tokenizer and model paths
        path_obj = Path(path)
        tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
        model_paths = list(path_obj.rglob("config.json"))

        if not model_paths:
            raise ValueError(f"No config.json found in {path}")

        model_path = model_paths[0].parent
        tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else model_path

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Try finding a pt model or safe tensors
        from transformers import AutoModelForSequenceClassification, AutoModel

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception:
            model = AutoModel.from_pretrained(model_path)

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        if hasattr(outputs, "logits"):
            return outputs.logits.sum().item()
        return outputs.last_hidden_state.sum().item()


class InferenceCatboost(InferenceStrategy):
    def get_prompt(self):
        return [0.0] * 10

    def testInference(self, path: str | Path, prompt):
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model_path = list(Path(path).glob("*.cbm"))[0]
        model.load_model(model_path)
        # Adapt prompt to model's expected feature count
        try:
            feature_count = len(model.feature_names_)
        except Exception:
            feature_count = 8
        dummy = [0.0] * feature_count
        pred = model.predict([dummy])
        if hasattr(pred, "flatten"):
            return float(pred.flatten()[0])
        else:
            return float(pred[0])

class InferenceSentenceTransformer(InferenceStrategy):
    def get_prompt(self):
        return "Dummy sentence for embeddings"

    def testInference(self, path: str | Path, prompt):
        import pickle

        model_path = list(Path(path).rglob("*.pkl"))[0]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        try:
            result = model.encode([clean_text(prompt)], convert_to_tensor=True)[0]
            result = result.cpu().numpy()
            return float(result.sum())
        except Exception:
            return None
