import streamlit as st
import pandas as pd
import plotly.express as px

# Page settings
st.set_page_config(
    page_title="AI News Intelligence Dashboard",
    page_icon="📰",
    layout="wide"
)

# Custom theme colors
st.markdown("""
<style>
body {
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📰 AI News Intelligence Dashboard")
st.markdown("AI powered analysis of live news articles")

# Load data
df = pd.read_csv("data/final_news.csv")

# -------- Metrics --------

total_news = len(df)
real_news = (df["fake_news_prediction"] == "Real").sum()
fake_news = (df["fake_news_prediction"] == "Fake").sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Articles", total_news)
col2.metric("Real News", real_news)
col3.metric("Fake News", fake_news)

st.divider()

# -------- Topic Distribution --------

topic_counts = df["topic_name"].value_counts().reset_index()
topic_counts.columns = ["Topic", "Articles"]

fig1 = px.bar(
    topic_counts,
    x="Topic",
    y="Articles",
    title="News Topic Distribution",
    color_discrete_sequence=["#ff4d6d"]  # red
)

st.plotly_chart(fig1, use_container_width=True)

# -------- Sentiment Distribution --------

fig2 = px.histogram(
    df,
    x="sentiment",
    nbins=30,
    title="Sentiment Distribution",
    color_discrete_sequence=["#ffb6c1"]  # baby pink
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# -------- News Table --------

st.subheader("News Articles")

st.dataframe(
    df[["title", "source", "topic_name", "sentiment", "fake_news_prediction"]],
    use_container_width=True
)