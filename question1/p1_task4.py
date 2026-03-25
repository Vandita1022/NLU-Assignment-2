import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import numpy as np
import os

# ---------------------------------------------------------
# 1. LOAD REPRESENTATIVE MODELS
# ---------------------------------------------------------
model_files = {
    "CBOW": "trained_models/cbow_d200_w8_n10.model",
    "Skip-gram": "trained_models/sg_d200_w8_n10.model"
}

# ---------------------------------------------------------
# 2. SELECT WORDS TO VISUALIZE (Clustered by Theme)
# ---------------------------------------------------------
word_clusters = {
    "Departments": ["physics", "chemistry", "mechanical", "electrical", "civil", "mathematics"],
    "Academic Levels": ["student", "scholar", "candidate", "applicant"],
    "Degrees": ["bachelor", "master", "phd", "mtech", "degree"],
    "Regulatory": ["exam", "grade", "credit", "semester", "registration"],
    "Admin": ["dean", "office", "director", "senate", "committee"]
}

# Flatten the dictionary to get a unique list of words
all_words = [word for cluster in word_clusters.values() for word in cluster]

def plot_embeddings(models_dict, words_to_plot):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.3)

    for i, (name, path) in enumerate(models_dict.items()):
        if not os.path.exists(path):
            print(f"❌ Skipping {name}: file not found.")
            continue
        
        model = Word2Vec.load(path)
        vocab = model.wv.key_to_index
        
        # Filter words that actually exist in the vocab
        valid_words = [w for w in words_to_plot if w in vocab]
        vectors = np.array([model.wv[w] for w in valid_words])
        
        # Dimensionality Reduction (200D -> 2D)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vectors)
        
        # Plotting
        ax = axes[i]
        ax.set_title(f"2D Projection: {name} Model", fontsize=18, fontweight='bold')
        
        # Use different colors for different themes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(word_clusters)))
        for cluster_name, color in zip(word_clusters.keys(), colors):
            indices = [valid_words.index(w) for w in word_clusters[cluster_name] if w in valid_words]
            if indices:
                ax.scatter(coords[indices, 0], coords[indices, 1], c=[color], label=cluster_name, s=150, alpha=0.7)
        
        # Add labels to points
        for j, txt in enumerate(valid_words):
            ax.annotate(txt, (coords[j, 0], coords[j, 1]), xytext=(5, 5), textcoords='offset points', fontsize=12)
            
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Word Groups", loc='best')

    plt.suptitle("Task 4: Cluster Visualization (CBOW vs Skip-gram)", fontsize=24, y=1.02)
    plt.savefig("task4_clusters_comparison.png", dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'task4_clusters_comparison.png'!")
    plt.show()

# ---------------------------------------------------------
# 3. EXECUTE
# ---------------------------------------------------------
plot_embeddings(model_files, all_words)