import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

df = pd.read_csv("data/clean_news.csv")

sia = SentimentIntensityAnalyzer()

df["sentiment"] = df["text"].apply(lambda x: sia.polarity_scores(x)["compound"])

df.to_csv("data/news_sentiment.csv", index=False)

print("Sentiment analysis complete")