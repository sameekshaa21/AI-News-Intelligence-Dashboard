import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

print("Loading datasets...")

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

# Shuffle dataset
data = data.sample(frac=1, random_state=42)

print("Dataset size:", data.shape)

# Use the full article text
X = data["text"]
y = data["label"]

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_vec = vectorizer.fit_transform(X)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

print("Training model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Testing model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("Saving model...")

with open("models/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Training completed successfully!")
print("Model saved in models folder.")