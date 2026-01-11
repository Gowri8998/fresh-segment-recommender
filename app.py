import streamlit as st
import pandas as pd

st.set_page_config(page_title="Amazon Fresh Recommender", layout="wide")

st.title("🛒 Amazon Fresh Recommendation System")

st.write("""
This app demonstrates a **segment-based recommendation system**
built using customer behavioral segmentation.
""")

st.header("📊 Load Data")

@st.cache_data
def load_customer_segments():
    return pd.read_csv("data/customer_segments.csv")

try:
    df_segments = load_customer_segments()
    st.success("Customer segments loaded successfully")
    st.write(df_segments.head())
except Exception as e:
    st.error("Customer segments file not found or failed to load.")
    st.write(e)

