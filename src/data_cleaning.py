import pandas as pd

df = pd.read_csv("data/news_data.csv") #reading csv files

df = df.dropna() #removing missing values

df["text"] = df["title"].fillna("") + " " + df["description"].fillna("") + " " + df["content"].fillna("") #combining columns

df.to_csv("data/clean_news.csv", index=False) #saving data to new csv file

print("Clean data ready")