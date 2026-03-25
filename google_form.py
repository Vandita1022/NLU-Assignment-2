import collections
import re

# File path to your corpus
corpus_file = "corpus.txt"

def get_top_10_formatted(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read and lowercase the entire text
            text = f.read().lower()
        
        # Use regex to find all words (ignoring numbers and punctuation)
        words = re.findall(r'\b[a-z]+\b', text)
        
        # Count occurrences
        word_counts = collections.Counter(words)
        
        # Get the top 10 most common
        top_10 = word_counts.most_common(10)
        
        # Format as: word1, frequency, word2, frequency...
        formatted_output = []
        for word, count in top_10:
            formatted_output.append(f"{word}, {count}")
        
        return ", ".join(formatted_output)

    except FileNotFoundError:
        return f"Error: '{filepath}' not found. Please check the filename."

# Execute and print result
result = get_top_10_formatted(corpus_file)

print("="*40)
print("COPY THIS FOR YOUR GOOGLE FORM:")
print("="*40)
print(result)
print("="*40)