import streamlit as st
import pandas as pd
import altair as alt

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
    "This application demonstrates an **end-to-end customer segmentation and "
    "recommendation system** built using Amazon Fresh transaction data."
)

# ---------------------------------------------
# Load Parquet Helper
# ---------------------------------------------
@st.cache_data
def load_parquet(name):
    return pd.read_parquet(f"data/{name}")

# ---------------------------------------------
# Load Data
# ---------------------------------------------
eda_daily = load_parquet("eda_daily_metrics.parquet")
eda_category_summary = load_parquet("eda_category_summary.parquet")
eda_weekday = load_parquet("eda_weekday_summary.parquet")
eda_orders_per_customer = load_parquet("eda_orders_per_customer.parquet")

segment_dist = load_parquet("segment_distribution.parquet")
feature_dist = load_parquet("feature_distribution_summary.parquet")
feature_corr = load_parquet("feature_correlation.parquet")

segment_kpis = load_parquet("segment_kpis.parquet")
segment_affinity = load_parquet("segment_item_affinity.parquet")
baseline_items = load_parquet("baseline_top_items.parquet")

df_segments = load_parquet("customer_segments.parquet")
df_item_lookup = load_parquet("item_lookup2.parquet")

# ---------------------------------------------
# Customer feature shard loader
# ---------------------------------------------
@st.cache_data
def load_customer_feature_shard(shard_key):
    return pd.read_parquet(
        f"data/customer_feature_shards/CUST{shard_key}.parquet"
    )

# ---------------------------------------------
# Segment Personas
# ---------------------------------------------
SEGMENT_PERSONAS = {
    "Large Basket Stock-up": (
        "**Who are they?**\nHigh-spending family households.\n\n"
        "**Behavior:** Infrequent but very large baskets.\n\n"
        "**Buys:** Full grocery shops, premium and bulk items."
    ),
    "Habitual Replenishers": (
        "**Who are they?** Loyal routine shoppers.\n\n"
        "**Behavior:** Frequent medium baskets.\n\n"
        "**Buys:** Dairy, produce, staples."
    ),
    "Fill-in Convenience Shoppers": (
        "**Who are they?** Urgency-driven shoppers.\n\n"
        "**Behavior:** Small baskets.\n\n"
        "**Buys:** Snacks, beverages, ready meals."
    ),
    "Low Engagement / Trial Users": (
        "**Who are they?** New or infrequent customers.\n\n"
        "**Behavior:** Sparse transactions.\n\n"
        "**Needs:** Onboarding and discovery."
    ),
    "Cold Start / Unsegmented": (
        "**Who are they?** Insufficient history.\n\n"
        "**Behavior:** Unknown."
    )
}

# ---------------------------------------------
# Tabs
# ---------------------------------------------
tabs = st.tabs([
    "🚀 Overview",
    "📊 Transaction EDA",
    "🧪 Feature Engineering",
    "🧠 Segmentation",
    "🤖 Recommendation Systems",
    "🔍 Customer Deep Dive"
])

# =====================================================
# TAB 4 — SEGMENTATION (ENRICHED)
# =====================================================
with tabs[3]:

    st.header("🧠 Customer Segmentation Insights")

    st.markdown(
        "Customers were segmented using unsupervised learning on behavioral features. "
        "This section interprets cluster quality, business meaning, and behavioral KPIs."
    )

    st.divider()

    # -----------------------------------------
    # KPI CARDS (from first app)
    # -----------------------------------------
    st.subheader("📌 Segment KPI Snapshot")

    col1, col2, col3 = st.columns(3)

    top_spend = segment_kpis.sort_values(
        "avg_total_spend", ascending=False
    ).iloc[0]

    most_freq = segment_kpis.sort_values(
        "avg_orders", ascending=False
    ).iloc[0]

    most_recent = segment_kpis.sort_values(
        "avg_recency_days"
    ).iloc[0]

    col1.metric(
        "Highest Avg Spend Segment",
        top_spend["segment_name"],
        f"{top_spend['avg_total_spend']:.0f}"
    )

    col2.metric(
        "Most Frequent Shoppers",
        most_freq["segment_name"],
        f"{most_freq['avg_orders']:.1f} orders"
    )

    col3.metric(
        "Most Recently Active Segment",
        most_recent["segment_name"],
        f"{most_recent['avg_recency_days']:.0f} days"
    )

    st.divider()

    # -----------------------------------------
    # Segment distribution
    # -----------------------------------------
    st.subheader("📊 Segment Distribution")

    st.bar_chart(
        segment_dist.set_index("segment_name")["customer_count"]
    )

    # -----------------------------------------
    # Segment KPIs
    # -----------------------------------------
    st.subheader("📈 Segment-Level KPIs")

    st.dataframe(segment_kpis)

    st.bar_chart(
        segment_kpis
        .set_index("segment_name")[
            ["avg_orders", "avg_total_spend", "avg_order_value"]
        ]
    )

    st.info(
        "Distinct behavioral differences validate the effectiveness of clustering. "
        "Each segment represents a unique shopping mission."
    )

    st.divider()

    # -----------------------------------------
    # Personas (from first app)
    # -----------------------------------------
    st.subheader("🧠 Segment Personas")

    for seg, desc in SEGMENT_PERSONAS.items():
        st.markdown(f"### {seg}")
        st.markdown(desc)

# =====================================================
# TAB 6 — CUSTOMER DEEP DIVE (ENRICHED)
# =====================================================
with tabs[5]:

    st.header("🔍 Customer Deep Dive")

    customer_id = st.text_input(
        "Enter Customer ID",
        placeholder="CUST1234567"
    )

    if not customer_id:
        st.stop()

    cust_row = df_segments[
        df_segments["customer_id"] == customer_id
    ]

    if cust_row.empty:
        st.warning("Customer not found.")
        st.stop()

    segment_name = cust_row.iloc[0]["segment_name"]

    st.success(f"Customer Segment: {segment_name}")

    # -----------------------------------------
    # Persona
    # -----------------------------------------
    st.info(
        SEGMENT_PERSONAS.get(
            segment_name,
            "Persona not available."
        )
    )

    # -----------------------------------------
    # Load customer features
    # -----------------------------------------
    shard_key = customer_id[4]
    shard_df = load_customer_feature_shard(shard_key)

    cust_feat = shard_df[
        shard_df["customer_id"] == customer_id
    ]

    if cust_feat.empty:
        st.warning("Customer feature data unavailable.")
        st.stop()

    cust_feat = cust_feat.iloc[0]

    # -----------------------------------------
    # KPI cards
    # -----------------------------------------
    st.subheader("📊 Customer Profile Snapshot")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Orders", int(cust_feat["orders"]))
    col2.metric("Total Spend", f"${cust_feat['total_spend']:.0f}")
    col3.metric("Avg Order Value", f"${cust_feat['avg_order_value']:.0f}")
    col4.metric(
        "Days Since Last Order",
        int(cust_feat["days_since_last_order"])
    )

    # -----------------------------------------
    # Segment comparison table
    # -----------------------------------------
    seg_kpi = segment_kpis[
        segment_kpis["segment_name"] == segment_name
    ].iloc[0]

    comparison_df = pd.DataFrame({
        "Metric": [
            "Orders",
            "Total Spend",
            "Avg Order Value",
            "Recency (Days)"
        ],
        "Customer": [
            cust_feat["orders"],
            cust_feat["total_spend"],
            cust_feat["avg_order_value"],
            cust_feat["days_since_last_order"]
        ],
        "Segment Average": [
            seg_kpi["avg_orders"],
            seg_kpi["avg_total_spend"],
            seg_kpi["avg_order_value"],
            seg_kpi["avg_recency_days"]
        ]
    })

    st.subheader("⚖️ Customer vs Segment Comparison")
    st.dataframe(comparison_df)

    # -----------------------------------------
    # Visual comparison
    # -----------------------------------------
    viz_df = comparison_df.melt(
        id_vars="Metric",
        var_name="Type",
        value_name="Value"
    )

    chart = alt.Chart(viz_df).mark_bar().encode(
        x="Metric:N",
        y="Value:Q",
        color="Type:N",
        column="Type:N"
    )

    st.altair_chart(chart, use_container_width=True)

    # -----------------------------------------
    # Recommendations
    # -----------------------------------------
    st.subheader("🎯 Personalized Recommendations")

    recs = (
        segment_affinity[
            segment_affinity["segment_name"] == segment_name
        ]
        .sort_values("rank")
        .head(40)
        .merge(df_item_lookup, on="asin", how="left")
    )

    selected = []
    seen = set()

    for _, row in recs.iterrows():
        cat = row.get("uphl1")
        if pd.isna(cat):
            continue
        if cat not in seen:
            selected.append(row)
            seen.add(cat)
        if len(selected) == 5:
            break

    for r in selected:
        st.markdown(
            f"• **{r['item_name']}**  \n"
            f"_Category: {r.get('uphl1','Unknown')}_"
        )

    with st.expander("🤔 Why am I seeing these recommendations?"):
        st.markdown(f"""
        These recommendations are generated using a
        **segment-aware collaborative filtering approach**.

        **Explanation:**
        - You belong to the **{segment_name}** segment  
        - Customers in this segment share similar shopping behavior  
        - Products frequently bought together are identified  
        - Results are diversified across categories  

        This ensures a balance between **relevance, diversity and interpretability**.
        """)

