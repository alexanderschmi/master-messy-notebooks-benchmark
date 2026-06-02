import logging
import pandas as pd
from pathlib import Path

from src.inference.strategies import (
    InferenceStrategy,
    InferenceTransformers,
    InferenceCatboost,
    InferenceSentenceTransformer,
    LSTMClassifier,
    extract_logits,
    generate_sentiment_score,
    format_input,
    load_nb10_inference_bundle,
    load_nb5_inference_bundle,
    load_nb6_inference_bundle,
    load_nb8_inference_bundle,
    load_nb9_inference_bundle,
    Model,
)

logger = logging.getLogger(__name__)

# Map nb1-nb10 to the right loader based on their expected outputs
class InferenceNb1(InferenceTransformers):
    def get_prompt(self):
        return pd.read_csv("data/nb1/input/map-charting-student-math-misunderstandings/train.csv").head(1)

    def testInference(self, path: str | Path, prompt):
        import torch
        from pathlib import Path
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

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

        try:
            bundle = load_nb5_inference_bundle(path)
            prompt = self.encode(prompt, bundle["vocab"])
            model = bundle["model"]
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

        try:
            bundle = load_nb6_inference_bundle(path)
            tokenizer = bundle["tokenizer"]
            model = bundle["model"]
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

        try:
            bundle = load_nb8_inference_bundle(path)
            tokenizer = bundle["tokenizer"]
            model = bundle["model"]
            model.eval()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=90,
                truncation=True,
            )
            with torch.no_grad():
                if bundle["type"] == "hf":
                    logits = extract_logits(model(**inputs))
                else:
                    logits = model(inputs["input_ids"])
            return logits.argmax(dim=-1).item()
        except Exception as e:
            logger.warning(f"Failed Nb8 PyTorch inference: {e}")
            return None


class InferenceNb9(InferenceTransformers):
    def testInference(self, path: str | Path, prompt):
        import torch

        try:
            bundle = load_nb9_inference_bundle(path)
            tokenizer = bundle["tokenizer"]
            model = bundle["model"]
            model.eval()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True,
            )
            with torch.no_grad():
                if bundle["type"] == "hf":
                    out = model(**inputs)
                else:
                    out = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                logits = extract_logits(out)
            return logits.argmax(dim=-1).item()
        except Exception as e:
            logger.warning(f"Failed Nb9 PyTorch inference: {e}")
            return None


class InferenceNb10(InferenceTransformers):
    def testInference(self, path: str | Path, prompt):
        import torch

        try:
            bundle = load_nb10_inference_bundle(path)
            tokenizer = bundle["tokenizer"]
            model = bundle["model"]
            model.eval()
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True,
            )
            with torch.no_grad():
                if bundle["type"] == "hf":
                    out = model(**inputs)
                else:
                    out = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                logits = extract_logits(out)
            return float(logits.sum().item())
        except Exception as e:
            logger.warning(f"Failed Nb10 PyTorch inference: {e}")
            return None


