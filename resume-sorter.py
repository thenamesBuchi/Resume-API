import numpy as np  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import gensim.downloader as api  # type: ignore

print("--- Loading Light Models (Please Wait) ---")

# 1. Sample Data
resume = "I am a student skilled in Python and SQL databases"
job_desc = "Looking for a developer with experience in Python and PostgreSQL"

# 2. TF-IDF (Baseline) - Fast, no download needed
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform([resume, job_desc])
tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# 3. Word2Vec (The 'Semantic' middle ground)
# This uses a tiny pre-trained model (glove-twitter-25) which is much smaller than BERT
try:
    w2v_model = api.load("glove-twitter-25")

    def get_vector(text):
        words = text.lower().split()
        vectors = [w2v_model[w] for w in words if w in w2v_model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(25)

    res_vec = get_vector(resume).reshape(1, -1)
    job_vec = get_vector(job_desc).reshape(1, -1)
    w2v_sim = cosine_similarity(res_vec, job_vec)[0][0]
except Exception as e:
    w2v_sim = "Error loading Word2Vec"

print(f"\nRESULTS:")
print(f"TF-IDF Similarity: {tfidf_sim:.4f}")
print(f"Word2Vec Similarity: {w2v_sim:.4f}")
