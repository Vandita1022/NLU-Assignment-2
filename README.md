# IIT Jodhpur NLP Assignment: Word Embeddings & Semantic Analysis

This repository contains the implementation for a two-part Natural Language Processing assignment.

- **Problem 1:** Institutional Word2Vec Model (IIT Jodhpur corpus)
- **Problem 2:** Character-Level Indian Name Generation using RNN architectures

---

# 📂 Project Structure

    .
    ├── Problem_1/
    │   ├── scripts/
    │   │   ├── task1_scraper.py
    │   │   ├── task2_trainer.py
    │   │   ├── task3_analysis.py
    │   │   └── task4_visualizer.py
    │   ├── data/
    │   │   ├── raw_pdfs/
    │   │   └── cleaned_corpus.txt
    │   ├── trained_models/
    │   └── outputs/
    │       ├── word_cloud.png
    │       ├── task2_results.csv
    │       ├── performance_viz.png
    │       └── cluster_viz.png
    │
    ├── Problem_2/
    │   ├── TrainingNames.txt
    │   ├── vanilla_rnn.py
    │   ├── blstm_model.py
    │   ├── attention_gru.py
    │   ├── p1_task2.py
    │   └── results/
    │
    └── README.md

---

# 🚀 Installation & Prerequisites

Install required libraries:

    pip install requests beautifulsoup4 pypdf2 nltk gensim pandas matplotlib seaborn scikit-learn wordcloud torch numpy

---

# 🛠️ Problem 1: Institutional Word2Vec Model

## 1. Dataset Preparation (`task1_scraper.py`)

**Function:**  
Scrapes 20+ IIT Jodhpur webpages and extracts text from academic PDFs.

**Processing Pipeline:**
- Lowercasing  
- Regex cleaning (`[^a-z\s]`)  
- Removing short tokens (<3 chars)  
- Stopword removal (NLTK + custom list)

**Output:**
- Tokens: **31,460**
- Vocabulary: **2,864 words**

---

## 2. Model Training (`task2_trainer.py`)

**Grid Search (36 configs):**
- Architectures: CBOW, Skip-gram  
- Dimensions: 50, 100, 200  
- Window sizes: 2, 5, 8  
- Negative samples: 5, 10  

**Output:**
- Saved models  
- Training time logs  
- Performance visualization  

---

## 3. Semantic Analysis (`task3_analysis.py`)

**Techniques:**
- Cosine similarity  
- Nearest neighbors  
- Word analogies  

**Example Result:**

    science : physics :: engineering : mechanical  (0.9909)

---

## 4. PCA Visualization (`task4_visualizer.py`)

- Reduced 200D → 2D  
- Plotted 25+ labeled words  
- Category clusters (Departments, Admin, Degrees)

**Observation:**
- Skip-gram → clear clusters  
- CBOW → scattered embeddings  

---

## 📊 Problem 1 Summary

### Training Efficiency
- Skip-gram slower than CBOW  
- Example (200D):
  - Skip-gram: 1.28s  
  - CBOW: 0.27s  

### Semantic Quality
- Captured institutional language:
  - phd ↔ mtech  
  - research ↔ development  

- Analogy:
  - faculty : research :: student : semesters  

### Clustering
- Skip-gram better for small datasets  
- CBOW suffers from vector saturation  

---

# 🧠 Problem 2: Character-Level Indian Name Generation

## 📌 Overview

This task implements three recurrent neural architectures from scratch to generate Indian names at the character level using a dataset of 1,000 names.

---

## 1. Dataset & Preprocessing

**Steps:**
- Character-level tokenization  
- Special tokens:
  - `<PAD>` = 0  
  - `<SOS>`  
  - `<EOS>`  

- Lowercasing → vocabulary size **V = 29**  
- Created:
  - `stoi` (string → index)  
  - `itos` (index → string)  

- Padding:
  - Used `pad_sequence` for batching  

---

## 2. Model Architectures

### A. Vanilla RNN

**Logic:**

    h_t = tanh(x_t W_ih + b_ih + h_{t-1} W_hh + b_hh)

- Parameters: ~40,477  
- Role: Baseline memorization model  

---

### B. Bidirectional LSTM (BLSTM)

- Custom LSTM cell (Input, Forget, Cell, Output gates)  
- Forward + backward pass  

**Representation:**
- 128D forward + 128D backward → 256D  

- Parameters: ~275,357  
- Role: Captures global structure  

---

### C. RNN with Causal Attention (GRU + Attention)

- Bahdanau-style additive attention  
- Computes context vector over past hidden states  

**Sampling Techniques:**
- Temperature scaling: T = 1.1  
- Top-K sampling: K = 5  

- Parameters: ~27,518  
- Role: Phonetic generalization  

---

## 3. Quantitative Results

| Model            | Novelty (%) | Diversity |
|------------------|------------|----------|
| Vanilla RNN      | 11.00%     | 0.98     |
| BLSTM            | 100.00%    | 1.00     |
| Attention GRU    | 17.00%     | 0.86     |

---

## 4. Qualitative Observations

### Vanilla RNN
- High realism  
- Heavy memorization  
- Behaves like lookup table  

### BLSTM
- Overfits  
- "Stuttering" issue  
  - Example: `aazosnhhaal`  

### Attention GRU
- Best balance  
- Generates realistic new names:
  - `Xavikrat`  
  - `Bhumakin`  

---

## 5. How to Run

### Requirements

- Python 3.10+  
- PyTorch  
- NumPy  

---

### Run Models

    # Vanilla RNN
    python vanilla_rnn.py

    # BLSTM
    python blstm_model.py

    # Attention GRU
    python attention_gru.py

---

### Evaluation

    python p1_task2.py

---

## 📂 Problem 2 Structure

    ├── TrainingNames.txt
    ├── vanilla_rnn.py
    ├── blstm_model.py
    ├── attention_gru.py
    ├── p1_task2.py
    └── results/

---

# 👤 Author

**Vandita Gupta**  
Pre-final year Undergraduate  
B.Tech in Artificial Intelligence & Data Science  
IIT Jodhpur
