import pandas as pd
import pickle

model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Load news coming from API pipeline
df = pd.read_csv("data/news_clustered.csv")

# Transform text
X = vectorizer.transform(df["text"])

# Predict fake or real
predictions = model.predict(X)

df["fake_news_prediction"] = predictions

df["fake_news_prediction"] = df["fake_news_prediction"].map({
    0: "Fake",
    1: "Real"
})

df.to_csv("data/final_news.csv", index=False)

print("Fake news detection added!")
print(df[["title","fake_news_prediction"]].head())