import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/final_news.csv")

df["sentiment"].hist()

plt.title("Sentiment Distribution")

plt.show()