import os
import time
import pandas as pd
import streamlit as st

PARQUET_DIR = "./stream_output/parquet"

st.set_page_config(page_title="Twitter Sentiment", layout="wide")
st.title("Twitter Hashtag Sentiment (Spark Streaming)")

placeholder = st.empty()

def load_data():
    if not os.path.exists(PARQUET_DIR):
        return pd.DataFrame()
    try:
        return pd.read_parquet(PARQUET_DIR)
    except Exception:
        # Parquet may be mid-write; retry next refresh
        return pd.DataFrame()

REFRESH_SEC = 5

while True:
    df = load_data()
    with placeholder.container():
        if df.empty:
            st.info("Waiting for data ...")
        else:
            # Basic stats
            total = len(df)
            by_label = df["sentiment_label"].value_counts().to_dict()
            st.subheader(f"Total tweets: {total}")
            st.write(by_label)

            # Time series by minute
            if "minute" in df.columns:
                ts = (df.groupby(["minute", "sentiment_label"])
                        .size().reset_index(name="count"))
                st.line_chart(ts.pivot(index="minute", columns="sentiment_label", values="count").fillna(0))

            # Sample table
            cols = ["timestamp","username","content","sentiment","sentiment_label"]
            show = df[cols].sort_values("timestamp", ascending=False).head(50)
            st.dataframe(show, use_container_width=True)
    time.sleep(REFRESH_SEC)
