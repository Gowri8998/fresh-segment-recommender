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

        
    st.info(
        "Customer behavior is dominated by routine and stock-up shopping patterns, "
        "highlighting strong opportunities for segment-aware personalization."
    )


# =============================================
# TAB 2 — SEGMENT INSIGHTS
# =============================================
# =============================================
# TAB 2 — SEGMENT INSIGHTS
# =============================================
with tab_segments:
    st.subheader("Segment Insights & Behavioral KPIs")

    st.subheader("📈 Segment-wise KPIs")
    st.dataframe(df_segment_kpis)

    st.info(
        "These KPIs summarize average behavioral patterns at the segment level. "
        "Stock-up segments typically show higher spend, while habitual segments "
        "exhibit higher engagement frequency."
    )

    st.divider()
    st.subheader("🧠 Segment Personas")

    for segment, description in SEGMENT_PERSONAS.items():
        st.markdown(f"### {segment}")
        st.markdown(description)


# =============================================
# TAB 3 — CUSTOMER DEEP DIVE
# =============================================
with tab_customer:
    st.subheader("Customer Deep Dive")

    customer_id_input = st.text_input(
        "Enter Customer ID",
        placeholder="CUSTxxxxxxx"
    )

    if customer_id_input:
        # -----------------------------------------
        # Segment lookup
        # -----------------------------------------
        customer_row = df_segments[
            df_segments["customer_id"] == customer_id_input
        ]

        if customer_row.empty:
            st.warning("Customer ID not found.")
            st.stop()

        segment_name = customer_row["segment_name"].values[0]

        st.success("Customer found")
        st.metric("Customer Segment", segment_name)

        # Persona
        persona_text = SEGMENT_PERSONAS.get(
            segment_name,
            "No persona description available."
        )
        st.info(persona_text)

        # -----------------------------------------
        # Load customer-level KPIs (on demand)
        # -----------------------------------------
        shard_key = customer_id_input[4]  # first digit after 'CUST'
        shard_df = load_customer_feature_shard(shard_key)

        customer_feat = shard_df[
            shard_df["customer_id"] == customer_id_input
        ]

        if customer_feat.empty:
            st.warning("Customer feature data not available.")
            st.stop()

        customer_feat = customer_feat.iloc[0]

        # -----------------------------------------
        # Segment KPI row
        # -----------------------------------------
        segment_kpi = df_segment_kpis[
            df_segment_kpis["segment_name"] == segment_name
        ].iloc[0]

        # -----------------------------------------
        # Customer vs Segment comparison
        # -----------------------------------------
        comparison_df = pd.DataFrame({
            "Metric": [
                "Orders",
                "Total Spend",
                "Avg Order Value",
                "Recency (Days)"
            ],
            "Customer": [
                round(customer_feat["orders"], 2),
                round(customer_feat["total_spend"], 2),
                round(customer_feat["avg_order_value"], 2),
                round(customer_feat["days_since_last_order"], 2)
            ],
            "Segment Average": [
                round(segment_kpi["avg_orders"], 2),
                round(segment_kpi["avg_total_spend"], 2),
                round(segment_kpi["avg_order_value"], 2),
                round(segment_kpi["avg_recency_days"], 2)
            ]
        })

        st.subheader("📊 Customer vs Segment Comparison")
        st.dataframe(comparison_df)
        
        # -----------------------------------------
        # Segment-aware Recommendations
        # -----------------------------------------
        st.subheader("🎯 Recommended for You")

        TOP_N = 5

        recs = (
            df_item_affinity[
                df_item_affinity["segment_name"] == segment_name
            ]
            .sort_values("rank")
            .head(TOP_N)
        )

        if recs.empty:
            st.warning("No recommendations available for this segment.")
        else:
            for i, row in recs.iterrows():
                st.write(f"• **{row['asin']}**")

        st.caption(
            "Recommendations are generated using segment-aware collaborative "
            "filtering based on item co-occurrence among similar customers."
        )
