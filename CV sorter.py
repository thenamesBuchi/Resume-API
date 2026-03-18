import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("resume_data.csv")

# FIX column names
df.columns = df.columns.str.strip()

# Fill missing values
df = df.fillna("")

# Combine CV fields
df["cv_text"] = (
    df["skills"] + " " +
    df["responsibilities"] + " " +
    df["positions"] + " " +
    df["degree_names"]
)

# Combine Job fields
df["job_text"] = (
    df["job_position_name"] + " " +
    df["skills_required"] + " " +
    df["responsibilities.1"]
)

# Sample
cv_texts = df["cv_text"].tolist()[:5]
job_texts = df["job_text"].tolist()[:5]

# Vectorization
all_texts = cv_texts + job_texts
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(all_texts)

# Similarity
similarity_matrix = cosine_similarity(vectors)

print("\n--- Fixed Similarity Results ---\n")

num_cvs = len(cv_texts)

for i in range(num_cvs):
    for j in range(len(job_texts)):
        score = similarity_matrix[i][num_cvs + j]
        print(f"CV {i+1} vs Job {j+1}: {score:.2f}")