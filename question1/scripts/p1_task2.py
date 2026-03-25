import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec

# ---------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------
print("📂 Loading cleaned corpus...")
if not os.path.exists("cleaned_corpus.txt"):
    print("❌ Error: 'cleaned_corpus.txt' not found! Please run Task 1 first.")
    exit()

with open("cleaned_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Word2Vec expects a list of tokenized sentences
sentences = [text.split()]

# ---------------------------------------------------------
# 2. EXPERIMENT GRID SETUP
# ---------------------------------------------------------
# We vary 3 features as per requirements
dimensions = [50, 100, 200]    # Feature (i): Embedding dimension
windows = [2, 5, 8]            # Feature (ii): Context window size
neg_samples = [5, 10]          # Feature (iii): Negative samples

results_log = []

# Directory to store the 36 models
model_dir = "trained_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ---------------------------------------------------------
# 3. THE TRAINING ENGINE
# ---------------------------------------------------------
print(f"🚀 Training 36 model variations. This may take a moment...")

for dim in dimensions:
    for win in windows:
        for neg in neg_samples:
            
            # --- Architecture A: CBOW (sg=0) ---
            start = time.time()
            cbow_model = Word2Vec(
                sentences, 
                vector_size=dim, 
                window=win, 
                negative=neg, 
                sg=0, 
                min_count=1, 
                workers=4
            )
            cbow_time = time.time() - start
            cbow_fname = f"cbow_d{dim}_w{win}_n{neg}.model"
            cbow_model.save(os.path.join(model_dir, cbow_fname))
            
            results_log.append({
                "Architecture": "CBOW",
                "Dimension": dim,
                "Window": win,
                "Neg_Samples": neg,
                "Training_Time": cbow_time
            })

            # --- Architecture B: Skip-gram (sg=1) ---
            start = time.time()
            sg_model = Word2Vec(
                sentences, 
                vector_size=dim, 
                window=win, 
                negative=neg, 
                sg=1, 
                min_count=1, 
                workers=4
            )
            sg_time = time.time() - start
            sg_fname = f"sg_d{dim}_w{win}_n{neg}.model"
            sg_model.save(os.path.join(model_dir, sg_fname))
            
            results_log.append({
                "Architecture": "Skip-gram",
                "Dimension": dim,
                "Window": win,
                "Neg_Samples": neg,
                "Training_Time": sg_time
            })
            
            print(f"✅ Finished: {dim}d, {win}w, {neg}n")

# ---------------------------------------------------------
# 4. DATA CONSOLIDATION & VISUALIZATION
# ---------------------------------------------------------
df = pd.DataFrame(results_log)
df.to_csv("task2_experiment_results.csv", index=False)

# Visualization 1: Training Time vs Dimension (Grouped by Architecture)
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")

# Create the plot
plot = sns.barplot(data=df, x="Dimension", y="Training_Time", hue="Architecture", palette="mako")

# Aesthetics
plt.title("Task 2: Training Efficiency Analysis\n(CBOW vs Skip-gram)", fontsize=16, fontweight='bold')
plt.ylabel("Time (seconds)", fontsize=12)
plt.xlabel("Embedding Dimension", fontsize=12)
plt.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')

# Add labels on top of bars
for p in plot.patches:
    plot.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("task2_performance_visualization.png", dpi=300)
print("\n📊 Visualization saved as 'task2_performance_visualization.png'!")
print("📄 Results table saved as 'task2_experiment_results.csv'!")
plt.show()