import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("data/news_sentiment.csv")

vectorizer = CountVectorizer(stop_words="english")

X = vectorizer.fit_transform(df["text"])

word_freq = X.sum(axis=0)

words_freq = [(word, word_freq[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True)

print("Top trends:")

for word, freq in sorted_words[:10]:
    print(word, freq)