import sqlite3
import pandas as pd

df = pd.read_csv("data/final_news.csv")

conn = sqlite3.connect("news_database.db")

df.to_sql("news_table", conn, if_exists="replace", index=False)

print("Data stored in database!")