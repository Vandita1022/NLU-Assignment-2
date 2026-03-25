import requests
from bs4 import BeautifulSoup
import re
import io
import PyPDF2
from collections import Counter
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -------------------------------
# STEP 1: URL & PDF COLLECTION
# -------------------------------

departments = [
    "bioscience-and-bioengineering", "chemistry", "chemical-engineering",
    "civil-and-infrastructure-engineering", "department-of-computer-science-and-engineering",
    "electrical-engineering", "mathematics", "mechanical-engineering",
    "materials-engineering", "physics"
]

urls = [f"https://www.iitj.ac.in/{dept}/" for dept in departments]

# Adding rich content and the MANDATORY academic regulations
urls += [
    "https://www.iitj.ac.in/",
    "https://www.iitj.ac.in/office-of-academics/en/list-of-academic-programs",
    "https://www.iitj.ac.in/main/en/introduction",
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    "https://www.iitj.ac.in/main/en/vision-and-mission",
    "https://www.iitj.ac.in/main/en/history",
    "https://www.iitj.ac.in/main/en/institute-reports",
    "https://www.iitj.ac.in/aiot-fab-facility/en/aiot-fab-facility?",
    "https://www.iitj.ac.in/office-of-director/en/office-of-director",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/1_Academic_Regulations_Final_03_09_2019.pdf",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/6_2024-04-17-661f605b54457-1713332315.pdf",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/2_Academic_Regulations_Final_03_09_2019.pdf"
]

# -------------------------------
# STEP 2: SCRAPING & PDF FUNCTION
# -------------------------------

def extract_content(url):
    try:
        header = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=header, timeout=15)
        
        if response.status_code != 200:
            return ""

        # Handle PDF files
        if url.endswith('.pdf'):
            with io.BytesIO(response.content) as f:
                reader = PyPDF2.PdfReader(f)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text() + " "
                print(f"✅ Scraped PDF: {url}")
                return pdf_text

        # Handle HTML pages
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose() # Remove boilerplate

        text = soup.get_text(separator=" ")
        
        if "error" in text.lower() or len(text.strip()) < 100:
            return ""

        print(f"✅ Scraped HTML: {url}")
        return text

    except Exception as e:
        print(f"❌ Error in {url}: {e}")
        return ""

# -------------------------------
# STEP 3: BUILD CORPUS & STATS
# -------------------------------

raw_corpus = ""
successful_docs = 0

for url in urls:
    content = extract_content(url)
    if content.strip():
        raw_corpus += content + " "
        successful_docs += 1

with open("raw_corpus.txt", "w", encoding="utf-8") as f:
    f.write(raw_corpus)

# -------------------------------
# STEP 4: CLEAN TEXT (PREPROCESSING)
# -------------------------------

def clean_text(text):
    text = text.lower()
    # Remove non-English chars (keeping only a-z and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_data = clean_text(raw_corpus)

# -------------------------------
# STEP 5: TOKENIZATION & STOPWORDS
# -------------------------------

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Added 'iit', 'jodhpur', 'department' to bad_words so the Word2Vec 
# focuses on more interesting semantic relationships.
bad_words = {"view", "home", "page", "go", "click", "iit", "jodhpur", "department", "university", "institute"}
stop_words.update(bad_words)

tokens = [w for w in cleaned_data.split() if w not in stop_words and len(w) > 2]
final_corpus = " ".join(tokens)

with open("cleaned_corpus.txt", "w", encoding="utf-8") as f:
    f.write(final_corpus)

# -------------------------------
# STEP 6: DATASET STATISTICS
# -------------------------------

num_tokens = len(tokens)
vocab_size = len(set(tokens))

print("\n" + "="*30)
print("📊 DATASET STATISTICS")
print("="*30)
print(f"Total Documents Scraped: {successful_docs}")
print(f"Total Tokens (Cleaned):  {num_tokens}")
print(f"Vocabulary Size:        {vocab_size}")
print("="*30)

# -------------------------------
# STEP 7: WORD CLOUD
# -------------------------------

wc = WordCloud(width=1000, height=500, background_color='white', colormap='viridis')
wc.generate(final_corpus)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("IIT Jodhpur Corpus Word Cloud", fontsize=15)
plt.show()
