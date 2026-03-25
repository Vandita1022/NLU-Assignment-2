import os
from gensim.models import Word2Vec

# ---------------------------------------------------------
# 1. LOAD THE BEST PERFORMING MODEL
# ---------------------------------------------------------
# We use Skip-gram 200d, w8 as it showed the best semantic separation
model_path = "trained_models/sg_d200_w8_n10.model"

if not os.path.exists(model_path):
    print(f"❌ Error: {model_path} not found. Check your 'trained_models' folder.")
    exit()

model = Word2Vec.load(model_path)
vocab = model.wv.key_to_index

# ---------------------------------------------------------
# 2. NEAREST NEIGHBORS (Top 5)
# ---------------------------------------------------------
target_words = ["research", "student", "phd", "exam"]

print("="*50)
print("🎯 PART 1: NEAREST NEIGHBORS")
print("="*50)

for word in target_words:
    print(f"\nNeighbors for '{word}':")
    if word in vocab:
        neighbors = model.wv.most_similar(word, topn=5)
        for i, (n, score) in enumerate(neighbors, 1):
            print(f"  {i}. {n:<15} (Similarity: {score:.4f})")
    else:
        print(f"  ⚠️ '{word}' not found in vocabulary.")

# ---------------------------------------------------------
# 3. ANALOGY EXPERIMENTS (Testing 10 Candidates)
# ---------------------------------------------------------
# Format: A is to B as C is to ??? (Target = B + C - A)
analogy_list = [
    ("faculty", "research", "student"),      # Faculty:Research :: Student:?
    ("bachelor", "degree", "master"),        # Bachelor:Degree :: Master:?
    ("mtech", "research", "btech"),          # MTech:Research :: BTech:?
    ("course", "credit", "program"),         # Course:Credit :: Program:?
    ("admission", "candidate", "exam"),      # Admission:Candidate :: Exam:?
    ("first", "semester", "final"),          # First:Semester :: Final:?
    ("dean", "office", "student"),           # Dean:Office :: Student:?
    ("science", "physics", "engineering"),   # Science:Physics :: Engineering:?
    ("tech", "program", "mtech"),            # Tech:Program :: MTech:?
    ("early", "registration", "late")        # Early:Registration :: Late:?
]

print("\n" + "="*50)
print("🧪 PART 2: ANALOGY EXPERIMENTS (10 TESTS)")
print("="*50)

success_count = 0
for a, b, c in analogy_list:
    # Check if all three words exist in our cleaned vocab
    if all(w in vocab for w in [a, b, c]):
        try:
            res = model.wv.most_similar(positive=[b, c], negative=[a], topn=1)
            print(f"✅ SUCCESS | {a}:{b} :: {c}:?  =>  RESULT: {res[0][0]} ({res[0][1]:.4f})")
            success_count += 1
        except Exception as e:
            print(f"❌ FAILED  | {a}:{b} :: {c}:?  =>  Error: {e}")
    else:
        # Identify which word is missing for the user
        missing = [w for w in [a, b, c] if w not in vocab]
        print(f"⚠️ SKIPPED | {a}:{b} :: {c}:?  =>  Missing tokens: {missing}")

print(f"\nTotal Successful Analogies: {success_count}/10")