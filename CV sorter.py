import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("resume_data.csv")

# Check columns (IMPORTANT)
print("Columns in dataset:", df.columns)

# Use correct column name from your dataset
# Change "Resume" if your column is named differently
cv_texts = df["responsibilities"].dropna().tolist()

# Limit data (keeps it fast and clean)
cv_texts = cv_texts[:5]

# Sample job descriptions (you can expand later)
jobs = [
    "data analyst internship python sql",
    "web developer html css javascript",
    "machine learning internship python",
    "finance assistant internship",
    "software engineering internship java"
]

# Combine CVs and jobs
all_texts = cv_texts + jobs

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(all_texts)

# Compute similarity
similarity_matrix = cosine_similarity(vectors)

# Display readable results
print("\n--- Similarity Results ---\n")

num_cvs = len(cv_texts)

for i in range(num_cvs):
    for j in range(len(jobs)):
        score = similarity_matrix[i][num_cvs + j]
        print(f"CV {i+1} vs Job {j+1}: {score:.2f}")

# Optional: show dataset preview
print("\n--- Dataset Preview ---\n")
print(df.head())