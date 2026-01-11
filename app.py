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

df_segments = load_customer_segments()

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
tab_overview, tab_segments, tab_customer = st.tabs(
    ["📊 Overview", "🧠 Segment Insights", "🔍 Customer Deep Dive"]
)

# =============================================
# TAB 1 — OVERVIEW
# =============================================
with tab_overview:
    st.subheader("Overall Customer Overview")

    st.metric("Total Customers", df_segments.shape[0])

    segment_counts = (
        df_segments["segment_name"]
        .value_counts()
        .reset_index()
    )
    segment_counts.columns = ["segment_name", "customer_count"]

    segment_counts["share_pct"] = (
        segment_counts["customer_count"]
        / segment_counts["customer_count"].sum()
        * 100
    ).round(2)

    st.subheader("Segment Distribution")
    st.dataframe(segment_counts)

    st.bar_chart(
        segment_counts.set_index("segment_name")["customer_count"]
    )

    # High-level KPIs
    total_customers = df_segments.shape[0]
    num_segments = df_segments["segment_name"].nunique()

    largest_segment = (
        df_segments["segment_name"]
        .value_counts()
        .idxmax()
    )

    smallest_segment = (
        df_segments["segment_name"]
        .value_counts()
        .idxmin()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("Number of Segments", num_segments)
    col3.metric("Largest Segment", largest_segment)
    col4.metric("Smallest Segment", smallest_segment)
        
    st.info(
        "Customer behavior is dominated by routine and stock-up shopping patterns, "
        "highlighting strong opportunities for segment-aware personalization."
    )


# =============================================
# TAB 2 — SEGMENT INSIGHTS
# =============================================
with tab_segments:
    st.subheader("Segment Personas")

    for segment, description in SEGMENT_PERSONAS.items():
        st.markdown(f"### {segment}")
        st.markdown(description)

# =============================================
# TAB 3 — CUSTOMER DEEP DIVE
# =============================================
with tab_customer:
    st.subheader("Customer Lookup")

    customer_id_input = st.text_input(
        "Enter Customer ID",
        placeholder="CUSTxxxxxxx"
    )

    if customer_id_input:
        customer_row = df_segments[
            df_segments["customer_id"] == customer_id_input
        ]

        if customer_row.empty:
            st.warning("Customer ID not found.")
        else:
            segment_name = customer_row["segment_name"].values[0]

            st.success("Customer found")
            st.metric("Segment", segment_name)

            persona_text = SEGMENT_PERSONAS.get(
                segment_name,
                "No persona description available."
            )
            st.info(persona_text)
