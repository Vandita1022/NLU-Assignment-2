import os

def evaluate_model(gen_file, training_set):
    """
    Reads a generated file and calculates Novelty and Diversity metrics.
    """
    if not os.path.exists(gen_file):
        return None, None
    
    with open(gen_file, "r", encoding="utf-8") as f:
        # Load names and ensure they are cleaned (lowercase, stripped)
        gen_names = [line.strip().lower() for line in f if line.strip()]
    
    total_count = len(gen_names)
    if total_count == 0:
        return 0.0, 0.0

    # 1. Novelty Rate: (Generated names not in training set) / Total
    novel_count = sum(1 for name in gen_names if name not in training_set)
    novelty_rate = (novel_count / total_count) * 100

    # 2. Diversity Score: (Unique generated names) / Total
    unique_count = len(set(gen_names))
    diversity_score = unique_count / total_count

    return novelty_rate, diversity_score

# --- FILE PATHS ---
TRAINING_FILE = "TrainingNames.txt"
MODEL_FILES = {
    "Vanilla RNN": "vanilla_generated.txt",
    "Bidirectional LSTM": "blstm_generated.txt",
    "Attention GRU": "attention_generated.txt"
}

# --- LOAD TRAINING DATA ---
try:
    with open(TRAINING_FILE, "r", encoding="utf-8") as f:
        training_names = set(line.strip().lower() for line in f if line.strip())
    print(f"✅ Loaded {len(training_names)} training names.\n")
except FileNotFoundError:
    print(f"❌ Error: {TRAINING_FILE} not found. Please ensure it is in the same directory.")
    exit()

# --- EXECUTE EVALUATION ---
header = f"{'Model Architecture':<25} | {'Novelty (%)':<15} | {'Diversity':<10}"
print(header)
print("-" * len(header))

for model_name, file_path in MODEL_FILES.items():
    nov, div = evaluate_model(file_path, training_names)
    
    if nov is not None:
        print(f"{model_name:<25} | {nov:<15.2f} | {div:<10.2f}")
    else:
        print(f"{model_name:<25} | ⚠️ File not found: {file_path}")

print("\nEvaluation Complete.")