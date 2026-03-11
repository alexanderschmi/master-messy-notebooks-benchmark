from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from pathlib import Path
import logging
import torch.nn as nn

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
        # Return something comparable, like argmax or sum
        if hasattr(outputs, "logits"):
            return outputs.logits.argmax(dim=-1).item()
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


class InferencePyTorch(InferenceStrategy):
    def get_prompt(self):
        return [0.5, 0.2, 0.1, 0.9]  # Dummy float features

    def testInference(self, path: str | Path, prompt):
        import torch

        model_path = list(Path(path).glob("*.pt"))[0]
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = torch.load(model_path, map_location="cpu", weights_only=False)
            model.eval()
            inputs = torch.tensor([prompt], dtype=torch.float32)
            with torch.no_grad():
                out = model(inputs)
            return out.sum().item()
        except Exception as e:
            logger.warning(f"Failed PyTorch inference: {e}")
            return None


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
    pass


class InferenceNb2(InferenceCatboost):
    pass


class InferenceNb3(InferenceTransformers):
    pass


class InferenceNb4(InferenceTransformers):
    pass


class InferenceNb5(InferencePyTorch):
    def get_prompt(self):
        return [2, 4, 5, 3]  # dummy int features

    def testInference(self, path: str | Path, prompt):
        import torch
        from pathlib import Path

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
            return out.sum().item()
        except Exception as e:
            logger.warning(f"Failed Nb5 PyTorch inference: {e}")
            return None


class InferenceNb6(InferencePyTorch):
    def get_prompt(self):
        return "Dummy prompt for BERT Model"

    def testInference(self, path: str | Path, prompt):
        import torch
        from transformers import AutoTokenizer
        from pathlib import Path

        model_path = list(Path(path).glob("*.pt"))[0]
        try:
            tokenizer = AutoTokenizer.from_pretrained("gaunernst/bert-mini-uncased")
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = torch.load(model_path, map_location="cpu", weights_only=False)
            model.eval()
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            return out.sum().item()
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
            return out.sum().item()
        except Exception as e:
            logger.warning(f"Failed Nb8 PyTorch inference: {e}")
            return None


class InferenceNb9(InferenceTransformers):
    pass


class InferenceNb10(InferencePyTorch):
    def get_prompt(self):
        return [1, 2, 3, 4]

    def testInference(self, path: str | Path, prompt):
        import torch
        from pathlib import Path

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

            if hasattr(out, "logits"):
                return out.logits.sum().item()
            elif hasattr(out, "last_hidden_state"):
                return out.last_hidden_state.sum().item()
            elif isinstance(out, tuple):
                return out[0].sum().item()
            else:
                return out.sum().item()

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
