from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 1. Sample Data (Simulating a Student CV and a Job Description)
# As noted in your report, student CVs are often short and unstructured[cite: 231].
resume = "I am a IT student with experience in Python, Java, and web development. I have worked on SQL databases."
job_description = "Looking for a Software Engineer intern skilled in Python, backend development, and relational databases like PostgreSQL."

# 2. TF-IDF Baseline Implementation [cite: 233]
# This model relies on exact keyword matching.
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([resume, job_description])
tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# 3. Sentence-BERT (SBERT) Implementation [cite: 240]
# This captures semantic meaning (e.g., 'SQL' matching 'PostgreSQL').
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([resume, job_description])
sbert_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# 4. Comparative Results Output [cite: 220, 226]
print(f"--- Comparative Similarity Results ---")
print(f"TF-IDF Similarity (Baseline): {tfidf_sim:.4f}")
print(f"Sentence-BERT Similarity (Advanced): {sbert_sim:.4f}")

# 5. Brief Analysis for your Report
if sbert_sim > tfidf_sim:
    print("\nAnalysis: SBERT scored higher because it recognized the semantic relationship ")
    print("between 'web development'/'backend' and 'SQL'/'PostgreSQL', which TF-IDF missed.")