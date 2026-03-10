import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("data/news_sentiment.csv")

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(df["text"])

# Number of clusters (topics)
num_clusters = 5

# Apply KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

df["topic"] = kmeans.fit_predict(X)

# -------- Print Keywords For Each Cluster --------

print("\nTop keywords per cluster:\n")

terms = vectorizer.get_feature_names_out()

for i in range(num_clusters):

    center = kmeans.cluster_centers_[i]

    top_words = [terms[ind] for ind in center.argsort()[-10:]]

    print(f"Cluster {i} keywords:", top_words)

# -------- Assign Topic Names --------

topic_map = {
    0: "Politics",
    1: "Technology",
    2: "Sports",
    3: "Economy",
    4: "Health"
}

df["topic_name"] = df["topic"].map(topic_map)

# Save new dataset
df.to_csv("data/news_clustered.csv", index=False)

print("\nTopic clustering completed!")
print("File saved as data/news_clustered.csv")