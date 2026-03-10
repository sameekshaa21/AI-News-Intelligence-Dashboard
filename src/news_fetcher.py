import requests
import pandas as pd

API_KEY = "a374255af3974720a23dce74fb666ce9"

url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=100&apiKey={API_KEY}"

response = requests.get(url)
data = response.json()

articles = data["articles"]

news = []

for article in articles:
    news.append({
        "title": article["title"],
        "description": article["description"],
        "content": article["content"],
        "source": article["source"]["name"]
    })

df = pd.DataFrame(news)

df.to_csv("data/news_data.csv", index=False)

print("News data saved!")