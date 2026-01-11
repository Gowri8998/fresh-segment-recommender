import streamlit as st
import pandas as pd

# ---------------------------------------------
# Page Config
# ---------------------------------------------
st.set_page_config(
    page_title="Amazon Fresh Recommendation System",
    layout="wide"
)

# ---------------------------------------------
# Title
# ---------------------------------------------
st.title("🛒 Amazon Fresh Recommendation System")
st.write(
    "This app demonstrates a **customer segmentation–based recommendation system** "
    "built using Amazon Fresh transaction data."
)

# ---------------------------------------------
# Load Customer Segments (Parquet)
# ---------------------------------------------
@st.cache_data
def load_customer_segments():
    return pd.read_parquet("data/customer_segments.parquet")

try:
    df_segments = load_customer_segments()
    st.success("Customer segments loaded successfully")
except Exception as e:
    st.error("Failed to load customer segments file.")
    st.write(e)
    st.stop()

# ---------------------------------------------
# Preview Data
# ---------------------------------------------
st.subheader("📊 Customer Segments Preview")
st.write(df_segments.head())

st.write("Total customers loaded:", df_segments.shape[0])
