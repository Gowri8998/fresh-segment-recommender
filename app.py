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
# Load Streamlit Summary Data
# ---------------------------------------------
@st.cache_data
def load_parquet(name):
    return pd.read_parquet(f"data/{name}")

eda_daily = load_parquet("eda_daily_metrics.parquet")
segment_dist = load_parquet("segment_distribution.parquet")
evaluation_summary = load_parquet("recommender_evaluation_summary.parquet")


# ---------------------------------------------
# Load Customer Segments (Parquet)
# ---------------------------------------------
@st.cache_data
def load_customer_segments():
    return pd.read_parquet("data/customer_segments.parquet")

df_segments = load_customer_segments()

# ---------------------------------------------
# Load Segment-level KPIs (Parquet)
# ---------------------------------------------
@st.cache_data
def load_segment_kpis():
    return pd.read_parquet("data/segment_kpis.parquet")

df_segment_kpis = load_segment_kpis()

# ---------------------------------------------
# Helper — Load customer features shard on demand
# ---------------------------------------------
@st.cache_data
def load_customer_feature_shard(shard_key: str):
    shard_path = f"data/customer_feature_shards/CUST{shard_key}.parquet"
    return pd.read_parquet(shard_path)

# ---------------------------------------------
# Load Segment-level Item Affinity (Recommendations)
# ---------------------------------------------
@st.cache_data
def load_segment_item_affinity():
    return pd.read_parquet("data/segment_item_affinity.parquet")

df_item_affinity = load_segment_item_affinity()

# ---------------------------------------------
# Load Item Lookup (ASIN → Item Name)
# ---------------------------------------------
@st.cache_data
def load_item_lookup():
    return pd.read_parquet("data/item_lookup2.parquet")

df_item_lookup = load_item_lookup()

# ---------------------------------------------
# Segment Persona Definitions
# ---------------------------------------------
SEGMENT_PERSONAS = {
    "Large Basket Stock-up": (
        "**Who are they?**\nHigh-spending households, often families, completing full grocery shops online.\n\n"
        "**Behavior:** Infrequent but very large baskets.\n\n"
        "**Needs:** Reliability, speed, wide selection.\n\n"
        "**Buys:** Full baskets, premium items, ready meals."
    ),
    "Habitual Replenishers": (
        "**Who are they?** Loyal, routine-driven shoppers.\n\n"
        "**Behavior:** Frequent small-to-medium baskets.\n\n"
        "**Needs:** Convenience, consistency.\n\n"
        "**Buys:** Essentials, fresh produce, dairy."
    ),
    "Fill-in Convenience Shoppers": (
        "**Who are they?** Urgency-driven top-up shoppers.\n\n"
        "**Behavior:** Small, quick baskets.\n\n"
        "**Needs:** Speed and ease.\n\n"
        "**Buys:** Snacks, ready meals, beverages."
    ),
    "Low Engagement / Trial Users": (
        "**Who are they?** New or infrequent users.\n\n"
        "**Behavior:** Sparse activity.\n\n"
        "**Needs:** Trust-building, discovery.\n\n"
        "**Buys:** Trial items."
    ),
    "Cold Start / Unsegmented": (
        "**Who are they?** Insufficient history.\n\n"
        "**Behavior:** Variable.\n\n"
        "**Needs:** Onboarding.\n\n"
        "**Buys:** Unclear."
    )
}

# ---------------------------------------------
# App Tabs
# ---------------------------------------------
tabs = st.tabs([
    "🚀 Overview",
    "📊 Transaction EDA",
    "🧪 Feature Engineering",
    "🧠 Segmentation",
    "🤖 Recommendation Systems",
    "📈 Evaluation",
    "🔍 Customer Deep Dive"
])

tab_overview = tabs[0]
tab_eda = tabs[1]
tab_features = tabs[2]
tab_segments = tabs[3]
tab_recs = tabs[4]
tab_eval = tabs[5]
tab_customer = tabs[6]


# =============================================
# TAB 1 — OVERVIEW
# =============================================
with tab_overview:

    st.title("🚀 Amazon Fresh Personalization System")

    st.markdown("""
    ### 🎯 Business Objective
    Build an **end-to-end customer personalization system** for Amazon Fresh using
    large-scale transaction data to deliver **behavior-driven recommendations**.

    This application demonstrates the complete analytics pipeline — from raw data
    exploration to customer segmentation and recommendation modeling.
    """)

    st.divider()

    # -----------------------------------------
    # Dataset Scale Metrics
    # -----------------------------------------
    st.subheader("📦 Dataset Scale")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{segment_dist.customer_count.sum():,}"
    )

    col2.metric(
        "Customer Segments",
        segment_dist.shape[0]
    )

    col3.metric(
        "Transaction Days",
        eda_daily.shape[0]
    )

    st.divider()

    # -----------------------------------------
    # Architecture Overview
    # -----------------------------------------
    st.subheader("🔗 System Architecture")

    st.code(
        """
Transactions
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Customer Segmentation
   ↓
Recommendation Models
   ↓
Model Evaluation
   ↓
Interactive Dashboard
        """,
        language="text"
    )

    st.divider()

    # -----------------------------------------
    # Segment Distribution
    # -----------------------------------------
    st.subheader("📊 Customer Segment Distribution")

    st.bar_chart(
        segment_dist.set_index("segment_name")["customer_count"]
    )

    st.info(
        "This dashboard transforms raw grocery transaction data into "
        "actionable customer intelligence. The segmentation layer enables "
        "behavior-aware recommendations tailored to different shopping missions."
    )
