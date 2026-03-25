import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. LOAD THE DATASET
file_path = "TrainingNames.txt"
with open(file_path, "r", encoding="utf-8") as f:
    names = [line.strip().lower() for line in f.readlines() if line.strip()]

print(f"✅ Loaded {len(names)} names.")

# 2. BASIC STATISTICS
lengths = [len(name) for name in names]
avg_len = sum(lengths) / len(lengths)
all_chars = "".join(names)
vocab = sorted(list(set(all_chars)))

print(f"📊 Average Name Length: {avg_len:.2f}")
print(f"📏 Min Length: {min(lengths)} | Max Length: {max(lengths)}")
print(f"𝔄 Vocabulary Size: {len(vocab)} characters")
print(f"🔡 Vocabulary: {vocab}")

# 3. CHARACTER FREQUENCY ANALYSIS
char_counts = Counter(all_chars)
start_chars = Counter([name[0] for name in names])
end_chars = Counter([name[-1] for name in names])

# 4. VISUALIZATION
plt.figure(figsize=(15, 5))

# Plot 1: Length Distribution
plt.subplot(1, 3, 1)
sns.histplot(lengths, bins=range(min(lengths), max(lengths)+1), color='skyblue')
plt.title("Distribution of Name Lengths")
plt.xlabel("Length")

# Plot 2: Top Starting Characters
plt.subplot(1, 3, 2)
starts_df = pd.DataFrame(start_chars.most_common(10), columns=['Char', 'Count'])
sns.barplot(data=starts_df, x='Char', y='Count', palette='viridis')
plt.title("Top 10 Starting Characters")

# Plot 3: Top Ending Characters
plt.subplot(1, 3, 3)
ends_df = pd.DataFrame(end_chars.most_common(10), columns=['Char', 'Count'])
sns.barplot(data=ends_df, x='Char', y='Count', palette='magma')
plt.title("Top 10 Ending Characters")

plt.tight_layout()
plt.show()

# 5. DATASET INTEGRITY CHECK
duplicates = len(names) - len(set(names))
print(f"\n⚠️ Duplicates found: {duplicates} ({duplicates/len(names)*100:.1f}%)")