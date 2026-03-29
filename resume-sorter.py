import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy (This is your Word2Vec alternative)
nlp = spacy.load("en_core_web_md")

resume = "I am a student skilled in Python and SQL databases"
job_desc = "Looking for a developer with experience in Python and PostgreSQL"

# 1. TF-IDF
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform([resume, job_desc])
tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# 2. SpaCy (Word2Vec)
doc1 = nlp(resume)
doc2 = nlp(job_desc)
spacy_sim = doc1.similarity(doc2)

print(f"TF-IDF Similarity: {tfidf_sim:.4f}")
print(f"SpaCy (Word2Vec) Similarity: {spacy_sim:.4f}")