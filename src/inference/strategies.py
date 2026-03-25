import warnings
warnings.filterwarnings("ignore")
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from pathlib import Path
import logging
import torch.nn as nn
import pandas as pd

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

        model_path = list(Path(path).glob("*.pkl"))[0]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        try:
            return float(model.encode([prompt]).sum())
        except Exception:
            return None
