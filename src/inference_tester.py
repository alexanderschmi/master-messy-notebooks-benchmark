from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from pathlib import Path
import yaml
import logging

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

# 2. Implementations
class InferenceTransformers(InferenceStrategy):
    def get_prompt(self):
        return "This is a dummy test string for NLP models."
        
    def testInference(self, path: str | Path, prompt):
        import torch
        from pathlib import Path
        
        # Find tokenizer and model paths
        path_obj = Path(path)
        tokenizer_paths = list(path_obj.rglob('tokenizer_config.json'))
        model_paths = list(path_obj.rglob('config.json'))
        
        if not model_paths:
            raise ValueError(f"No config.json found in {path}")
            
        model_path = model_paths[0].parent
        tokenizer_path = tokenizer_paths[0].parent if tokenizer_paths else model_path
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # Try finding a pt model or safe tensors
        from transformers import AutoModelForSequenceClassification, AutoModel
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except:
            model = AutoModel.from_pretrained(model_path)
            
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Return something comparable, like argmax or sum
        if hasattr(outputs, 'logits'):
            return outputs.logits.argmax(dim=-1).item()
        return outputs.last_hidden_state.sum().item()

class InferenceCatboost(InferenceStrategy):
    def get_prompt(self):
        return "This is a dummy text for catboost"
        
    def testInference(self, path: str | Path, prompt):
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model_path = list(Path(path).glob('*.cbm'))[0]
        model.load_model(model_path)
        return float(model.predict([prompt])[0])

class InferencePyTorch(InferenceStrategy):
    def get_prompt(self):
        return [0.5, 0.2, 0.1, 0.9] # Dummy float features
        
    def testInference(self, path: str | Path, prompt):
        import torch
        model_path = list(Path(path).glob('*.pt'))[0]
        try:
            model = torch.load(model_path, map_location='cpu')
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
        model_path = list(Path(path).glob('*.pkl'))[0]
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        try:
            return float(model.encode([prompt]).sum())
        except:
            return None

# Map nb1-nb10 to the right loader based on their expected outputs
class InferenceNb1(InferenceTransformers): pass
class InferenceNb2(InferenceCatboost): pass
class InferenceNb3(InferenceTransformers): pass
class InferenceNb4(InferenceTransformers): pass
class InferenceNb5(InferencePyTorch): pass
class InferenceNb6(InferencePyTorch): pass
class InferenceNb7(InferenceSentenceTransformer): pass
class InferenceNb8(InferenceTransformers): pass 
class InferenceNb9(InferenceTransformers): pass
class InferenceNb10(InferencePyTorch): pass

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
    test_artifact(InferenceNb1())
