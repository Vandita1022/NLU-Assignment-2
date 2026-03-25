# IIT Jodhpur NLP Assignment: Word Embeddings & Semantic Analysis

This repository contains the implementation for a two-part Natural Language Processing assignment.  
**Problem 1** focuses on building a custom "Institutional" Word2Vec model by scraping data from IIT Jodhpur's digital infrastructure.

---

## 📂 Project Structure

    .
    ├── Problem_1/
    │   ├── scripts/
    │   │   ├── task1_scraper.py      # Web scraping, PDF extraction, & preprocessing
    │   │   ├── task2_trainer.py      # Model training (36 variations) & performance viz
    │   │   ├── task3_analysis.py     # Semantic analysis (Neighbors & Analogies)
    │   │   └── task4_visualizer.py   # PCA dimensionality reduction & plotting
    │   ├── data/
    │   │   ├── raw_pdfs/             # Source Academic Regulation PDFs
    │   │   └── cleaned_corpus.txt    # Final preprocessed text for training
    │   ├── trained_models/           # Storage for .model files (36 variations)
    │   └── outputs/
    │       ├── word_cloud.png        # Task 1: Dataset visualization
    │       ├── task2_results.csv     # Task 2: Training time logs
    │       ├── performance_viz.png   # Task 2: Efficiency comparison plot
    │       └── cluster_viz.png       # Task 4: 2D PCA cluster comparison
    └── Problem_2/
        └── [Files for Problem 2 implementation]

---

## 🚀 Installation & Prerequisites

To run these scripts, you will need **Python 3.8+** and the following libraries:

    pip install requests beautifulsoup4 pypdf2 nltk gensim pandas matplotlib seaborn scikit-learn wordcloud

---

## 🛠️ Execution Flow (Problem 1)

### 1. Dataset Preparation (`task1_scraper.py`)

**Function:**  
Crawls 20+ URLs from IIT Jodhpur and extracts text from localized Academic Regulation PDFs.

**Cleaning Pipeline:**
- Lowercase normalization  
- Punctuation removal via regex (`[^a-z\s]`)  
- Filters out words shorter than 3 characters  

**Stopwords:**
- NLTK stopwords  
- Custom noise list (e.g., *iit*, *jodhpur*)

**Outcome:**
- Corpus size: **31,460 tokens**  
- Vocabulary size: **2,864 unique words**

---

### 2. Model Training (`task2_trainer.py`)

**Function:**  
Performs a grid search over **36 parameter combinations**

**Variations:**
- Architectures: **CBOW vs Skip-gram**  
- Embedding Dimensions: **{50, 100, 200}**  
- Context Windows: **{2, 5, 8}**  
- Negative Samples: **{5, 10}**  

**Outcome:**
- Logs training time for efficiency analysis  
- Saves all trained model variants  

---

### 3. Semantic Analysis (`task3_analysis.py`)

**Function:**  
Evaluates the best Skip-gram model using **cosine similarity**

**Tasks:**
- **Nearest Neighbors:** Top-5 similar words (e.g., *research*, *student*, *phd*)  
- **Analogy Testing:** Vector arithmetic (**A:B :: C:?**)  

**Result Example:**

    science : physics :: engineering : mechanical  (score: 0.9909)

---

### 4. Cluster Visualization (`task4_visualizer.py`)

**Function:**  
Reduces 200D embeddings to 2D using PCA

**Visualization:**
- 25+ labeled words  
- Categories: Departments, Admin, Degrees, etc.

**Observation:**
- Skip-gram → tight, meaningful clusters  
- CBOW → dispersed and less structured  

---

## 📊 Summary of Problem 1 Results

### 1. Training Efficiency
- Skip-gram is computationally heavier than CBOW  
- Example (200D):
  - Skip-gram: **1.28s**
  - CBOW: **0.27s**

---

### 2. Semantic Integrity
The model successfully captures institutional language:

- Structural:
  - *phd ↔ mtech*
  - *research ↔ development*

- Regulatory:
  - *faculty : research :: student : semesters*

---

### 3. Clustering Behavior
- Skip-gram performs better on small, domain-specific datasets  
- Clear separation between:
  - Academic concepts  
  - Administrative roles  

- CBOW suffers from **vector saturation**

---

## 📝 Problem 2

*(To be added — implementation details pending)*

---

## 👤 Author

**Vandita Gupta**  
Pre-final year Undergraduate  
B.Tech in Artificial Intelligence & Data Science  
IIT Jodhpur
