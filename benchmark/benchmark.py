import pandas as pd
from utils import get_model_response, load_notebooks
import yaml

# Load the configuration
with open("benchmark/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# --- CONFIGURATION (Loaded from YAML) ---
MODELS_TO_TEST = config["models"]
TEMPERATURE = config["settings"]["temperature"]
DATA = config["data"]

def run_benchmark(notebooks):
    results = []
    print(f"Starting Benchmark on {len(MODELS_TO_TEST)} models with {2} test cases...\n")

    # Create tasks for all combinations of Models + Test Cases
    tasks = []
    
    for model in MODELS_TO_TEST:
        # We wrap the logic in an async function to capture variables correctly
        def process_single_run(m=model):
            # 1. Get the model's answer
            answer = get_model_response(model=model["name"], api_key=model["key"], notebooks=notebooks, temperature=TEMPERATURE)
        
            
            return {
                "Model": m,
                "Response": answer,
            }
        
        tasks.append(process_single_run())

    # Run all tasks concurrently
    results.extend(tasks)

    # --- REPORTING ---
    df = pd.DataFrame(results)
    
    # 1. Show Detailed Table
    print("\n--- Detailed Results ---")
    print(df.to_markdown(index=False))

    # 3. Save to CSV
    df.to_csv("llm_benchmark_results.csv", index=False)
    print("\n✅ Results saved to 'llm_benchmark_results.csv'")

if __name__ == "__main__":
    notebooks = load_notebooks(DATA)
    print(notebooks)
    run_benchmark(notebooks=notebooks)