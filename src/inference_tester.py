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


# Map nb1-nb10 to the right loader based on their expected outputs
class InferenceNb1(InferenceTransformers):
    def get_prompt(self):
        return pd.read_csv("data/nb1/input/map-charting-student-math-misunderstandings/train.csv").head(1)

    def testInference(self, path: str | Path, prompt):
        import torch
        from pathlib import Path

        prompt.Misconception = prompt.Misconception.fillna('NA')
        idx = prompt.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
        correct = prompt.loc[idx].copy()
        correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
        correct = correct.sort_values('c',ascending=False)
        correct = correct.drop_duplicates(['QuestionId'])
        correct = correct[['QuestionId','MC_Answer']]
        correct['is_correct'] = 1

        prompt = prompt.merge(correct, on=['QuestionId','MC_Answer'], how='left')
        prompt.is_correct = prompt.is_correct.fillna(0)

        prompt = prompt.apply(format_input, axis=1).values[0]

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
        # Return something comparable, like argmax or sum
        if hasattr(outputs, "logits"):
            return outputs.logits.argmax(dim=-1).item()
        return outputs.last_hidden_state.item()
        


class InferenceNb2(InferenceCatboost):
    def get_prompt(self):
        return "Hello World"

    def testInference(self, path: str | Path, prompt):
        from catboost import CatBoostRegressor
        import joblib
        import numpy as np
        from scipy.sparse import hstack

        df = pd.DataFrame()
        df["full_text"] = [prompt]

        model1 = joblib.load(list(Path(path).rglob("model1*.pkl"))[0])
        model2 = CatBoostRegressor()
        model2.load_model(list(Path(path).rglob("model2*.cbm"))[0])
        model3 = joblib.load(list(Path(path).rglob("model3*.pkl"))[0])
        model4 = joblib.load(list(Path(path).rglob("model4*.pkl"))[0])
        meta_model = CatBoostRegressor()
        meta_model.load_model(list(Path(path).rglob("meta_model*.cbm"))[0])
        vectorizer = joblib.load(list(Path(path).rglob("vectorizer*.pkl"))[0])
        normalizer = joblib.load(list(Path(path).rglob("normalizer*.pkl"))[0])

        df["compound"],df["negative"],df["positive"],df["neutral"] = generate_sentiment_score(prompt)

        df["text_len"] = df['full_text'].apply(lambda x:len(x.split()))
        
        X_tfidf = vectorizer.transform([prompt])
        X_com = normalizer.transform(df["compound"].values.reshape(-1, 1))
        X_neg = normalizer.transform(df["negative"].values.reshape(-1, 1))
        X_pos = normalizer.transform(df["positive"].values.reshape(-1, 1))
        X_neu = normalizer.transform(df["neutral"].values.reshape(-1, 1))
        X_len = normalizer.transform(df["text_len"].values.reshape(-1, 1))

        X_final = hstack((X_tfidf,X_com,X_neg,X_pos,X_neu,X_len))

        pred1 = model1.predict(X_final)
        pred2 = model2.predict(X_final)
        pred3 = model3.predict(X_final)
        pred4 = model4.predict(X_final)
        pred = np.column_stack([pred1, pred2, pred3, pred4])
        result = meta_model.predict(pred)
        if hasattr(result, "flatten"):
            return float(result.flatten()[0])
        else:
            return float(result[0])

class InferenceNb3(InferenceTransformers):
    pass


class InferenceNb4(InferenceTransformers):
    pass


class InferenceNb5(InferenceTransformers):
    def encode(self, text, vocab):
        tokens = text.split()
        ids = [2]  # <START>
        for t in tokens:
            ids.append(vocab.get(t, 1))   # <UNK>
        ids.append(3)  # <END>
        return ids

    def testInference(self, path: str | Path, prompt):
        import torch
        from pathlib import Path
        import json

        with open(Path(path) / "vocab.json", "r") as f:
            vocab = json.load(f)

        prompt = self.encode(prompt, vocab)

        model_path = list(Path(path).glob("*.pt"))[0]
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = torch.load(model_path, map_location="cpu", weights_only=False)
            model.eval()
            inputs = torch.tensor([prompt], dtype=torch.long)
            with torch.no_grad():
                out = model(inputs)
            return out.item()
        except Exception as e:
            logger.warning(f"Failed Nb5 PyTorch inference: {e}")
            return None


class InferenceNb6(InferenceTransformers):
    def testInference(self, path: str | Path, prompt):
        import torch
        from pathlib import Path
        from transformers import AutoTokenizer, AutoModel
        from peft import get_peft_model, LoraConfig, TaskType

        # Create a LoRA config
        lora_config = LoraConfig(
            r=8,                     # rank
            lora_alpha=32,           # scaling
            target_modules=["query","value"],  # which layers to inject LoRA into
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,     # since we want embeddings, not classification
        )

        base_model = AutoModel.from_pretrained("gaunernst/bert-mini-uncased")
        hidden_dim, new_hidden_dim = 256, 256
        # Wrap BERT with LoRA
        lora_model = get_peft_model(base_model, lora_config)
        lora_model.print_trainable_parameters()

        # Find tokenizer and model paths
        path_obj = Path(path)
        tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
        model_paths = list(path_obj.rglob("*.pth"))

        if not model_paths:
            raise ValueError(f"No config.json found in {path}")

        model_path = model_paths[0]
        tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else model_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Model(lora_model, hidden_dim, new_hidden_dim, 2)
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            return out.argmax(dim=-1).item()
        except Exception as e:
            logger.warning(f"Failed Nb6 PyTorch inference: {e}")
            return None


class InferenceNb7(InferenceSentenceTransformer):
    pass


class InferenceNb8(InferenceTransformers):
    def testInference(self, path: str | Path, prompt):
        import torch
        from transformers import AutoTokenizer

        path_obj = Path(path)
        tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
        model_paths = list(path_obj.rglob("*.pt"))

        if not model_paths:
            return super().testInference(path, prompt)

        tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else path_obj
        model_path = model_paths[0]

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        try:
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            model.eval()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=66,
                truncation=True,
            )
            with torch.no_grad():
                out = model(inputs["input_ids"])
            return out.argmax(dim=-1).item()
        except Exception as e:
            logger.warning(f"Failed Nb8 PyTorch inference: {e}")
            return None


class InferenceNb9(InferenceTransformers):
    def testInference(self, path: str | Path, prompt):
        import torch
        from transformers import AutoTokenizer

        path_obj = Path(path)
        tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
        model_paths = list(path_obj.rglob("*.pt"))

        if not model_paths:
            return super().testInference(path, prompt)

        tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else path_obj
        model_path = model_paths[0]

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        try:
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            model.eval()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=66,
                truncation=True,
            )
            with torch.no_grad():
                out = model(inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],)
            return out.logits.argmax(dim=-1).item()
        except Exception as e:
            logger.warning(f"Failed Nb9 PyTorch inference: {e}")
            return None


class InferenceNb10(InferenceTransformers):
    def testInference(self, path: str | Path, prompt):
        import torch
        from transformers import AutoTokenizer

        path_obj = Path(path)
        tokenizer_paths = list(path_obj.rglob("tokenizer_config.json"))
        model_paths = list(path_obj.rglob("*.pt"))

        if not model_paths:
            return super().testInference(path, prompt)

        tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else path_obj
        model_path = model_paths[0]

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        try:
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            model.eval()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=66,
                truncation=True,
            )
            with torch.no_grad():
                out = model(inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],)
            return out.logits.sum().item()
        except Exception as e:
            logger.warning(f"Failed Nb10 PyTorch inference: {e}")
            return None


# 3. Write your unified testing function
def test_artifact(script_instance: InferenceStrategy, path: str | Path = None):
    prompt = script_instance.get_prompt()
    if path:
        result = script_instance.testInference(path, prompt)
    else:
        result = script_instance.testInference("dummy_path", prompt)
    logger.info(f"Generated: {result}")
    return result


# Execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    for i in range(1, 11):
        nb_name = f"nb{i}"
        nb_dir = data_dir / nb_name / "output"

        inference_class_name = f"InferenceNb{i}"

        if inference_class_name in globals():
            inference_class = globals()[inference_class_name]

            if nb_dir.exists():
                logger.info(f"Testing {nb_name} from {nb_dir}")
                try:
                    test_artifact(inference_class(), nb_dir)
                except Exception as e:
                    logger.error(f"Error testing {nb_name}: {e}")
            else:
                logger.warning(f"Output directory not found for {nb_name}: {nb_dir}")
        else:
            logger.warning(f"Inference class {inference_class_name} not found.")
