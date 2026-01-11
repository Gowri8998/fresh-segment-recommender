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

    # -----------------------------------------
    # Title
    # -----------------------------------------
    st.subheader("🧠 Segment Insights & Behavioral KPIs")

    # -----------------------------------------
    # Segment KPI Snapshot (Cards)
    # -----------------------------------------
    st.subheader("📌 Segment KPI Snapshot")

    col1, col2, col3 = st.columns(3)

    top_spend_segment = (
        df_segment_kpis
        .sort_values("avg_total_spend", ascending=False)
        .iloc[0]
    )

    most_frequent_segment = (
        df_segment_kpis
        .sort_values("avg_orders", ascending=False)
        .iloc[0]
    )

    lowest_recency_segment = (
        df_segment_kpis
        .sort_values("avg_recency_days")
        .iloc[0]
    )

    col1.metric(
        "Highest Avg Spend Segment",
        top_spend_segment["segment_name"],
        f"{top_spend_segment['avg_total_spend']:.0f}"
    )

    col2.metric(
        "Most Frequent Shoppers",
        most_frequent_segment["segment_name"],
        f"{most_frequent_segment['avg_orders']:.1f} orders"
    )

    col3.metric(
        "Most Recently Active Segment",
        lowest_recency_segment["segment_name"],
        f"{lowest_recency_segment['avg_recency_days']:.0f} days"
    )

    st.divider()

    # -----------------------------------------
    # Segment-wise KPI Table
    # -----------------------------------------
    st.subheader("📈 Segment-wise KPIs")
    st.dataframe(df_segment_kpis)

    st.info(
        "These KPIs summarize average behavioral patterns at the segment level. "
        "Different segments contribute value either through higher spend per order "
        "or through higher purchase frequency."
    )

    # -----------------------------------------
    # Avg Spend by Segment
    # -----------------------------------------
    st.subheader("💰 Average Total Spend by Segment")

    st.bar_chart(
        df_segment_kpis.set_index("segment_name")["avg_total_spend"]
    )

    # -----------------------------------------
    # Avg Orders by Segment
    # -----------------------------------------
    st.subheader("🛒 Average Number of Orders by Segment")

    st.bar_chart(
        df_segment_kpis.set_index("segment_name")["avg_orders"]
    )

    st.info(
        "Stock-up oriented segments typically drive higher spend per customer, "
        "while habitual segments drive engagement through frequent purchases. "
        "These behavioral differences motivate segment-aware recommendation strategies."
    )

    st.divider()

    # -----------------------------------------
    # Segment Personas
    # -----------------------------------------
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

        # -----------------------------------------
        # Customer Profile Snapshot
        # -----------------------------------------

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
        # Customer Profile Snapshot
        # -----------------------------------------
        st.subheader("🧑 Customer Profile Snapshot")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Orders", int(customer_feat["orders"]))
        col2.metric("Total Spend", f"₹{customer_feat['total_spend']:.0f}")
        col3.metric("Avg Order Value", f"₹{customer_feat['avg_order_value']:.0f}")
        col4.metric("Days Since Last Order", int(customer_feat["days_since_last_order"]))


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

        # Visual comparison
        import altair as alt
        
        viz_df = comparison_df.melt(
            id_vars="Metric",
            value_vars=["Customer", "Segment Average"],
            var_name="Type",
            value_name="Value"
        )
        
        chart = alt.Chart(viz_df).mark_bar().encode(
            x=alt.X("Metric:N", title=None),
            y=alt.Y("Value:Q"),
            color="Type:N",
            column="Type:N"
        )
        
        st.altair_chart(chart, use_container_width=True)

        
        # -----------------------------------------
        # Segment-aware Recommendations
        # -----------------------------------------
        st.subheader("🎯 Recommended for You")

        TOP_N = 5
        CANDIDATES = 50  # take top 50, then diversify

        # Top candidates by affinity
        candidates = (
            df_item_affinity[
                df_item_affinity["segment_name"] == segment_name
            ]
            .sort_values("rank")
            .head(CANDIDATES)
        )

        # Join with item metadata
        candidates = candidates.merge(
            df_item_lookup,
            on="asin",
            how="left"
        )

        # Diversity-aware selection
        selected = []
        seen_categories = set()

        for _, row in candidates.iterrows():
            category = row.get("uphl1")
        
            # Skip items with no category info
            if pd.isna(category):
                continue
        
            if category not in seen_categories:
                selected.append(row)
                seen_categories.add(category)
        
            if len(selected) == TOP_N:
                break


        if not selected:
            st.warning("No recommendations available.")
        else:
            for row in selected:
                item_name = row.get("item_name", "Unknown Item")
                category = row.get("uphl1", "Unknown Category")
            
                st.write(
                    f"• **{item_name}**  \n"
                    f"  _Category: {category}_"
                )


        st.caption(
            "Recommendations are generated using segment-aware collaborative "
            "filtering with category-level diversification to improve variety."
        )

        with st.expander("🤔 Why am I seeing these recommendations?"):
            st.markdown(f"""
            These recommendations are generated using a **segment-aware collaborative filtering** approach.
        
            **How it works:**
            - You belong to the **{segment_name}** customer segment
            - Customers in this segment exhibit similar shopping behavior
            - Items are recommended based on **frequent co-occurrence** in their baskets
            - Results are **diversified across categories** to improve discovery
        
            This approach balances **relevance, scalability, and interpretability**.
            """)

        
