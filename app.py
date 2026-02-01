import streamlit as st
import pandas as pd
import altair as alt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Amazon Fresh Recommendation System",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================
st.title("🛒 Amazon Fresh Recommendation System")
st.write(
    "An **end-to-end customer segmentation and recommendation system** "
    "built using Amazon Fresh transaction data."
)

# =====================================================
# LOAD HELPERS
# =====================================================
@st.cache_data
def load_parquet(name):
    return pd.read_parquet(f"data/{name}")

@st.cache_data
def load_customer_feature_shard(shard_key):
    return pd.read_parquet(
        f"data/customer_feature_shards/CUST{shard_key}.parquet"
    )

# =====================================================
# LOAD DATA
# =====================================================
eda_daily = load_parquet("eda_daily_metrics.parquet")
eda_category_summary = load_parquet("eda_category_summary.parquet")
eda_weekday = load_parquet("eda_weekday_summary.parquet")
eda_orders_per_customer = load_parquet("eda_orders_per_customer.parquet")

segment_distribution = load_parquet("segment_distribution.parquet")
feature_dist = load_parquet("feature_distribution_summary.parquet")
feature_corr = load_parquet("feature_correlation.parquet")

segment_kpis = load_parquet("segment_kpis.parquet")
segment_affinity = load_parquet("segment_item_affinity.parquet")
baseline_items = load_parquet("baseline_top_items.parquet")

df_segments = load_parquet("customer_segments.parquet")
df_item_lookup = load_parquet("item_lookup2.parquet")

# =====================================================
# SEGMENT PERSONAS
# =====================================================
SEGMENT_PERSONAS = {
    "Large Basket Stock-up": (
        "**Who are they?**\nHigh-spending family households.\n\n"
        "**Behavior:** Infrequent but very large baskets.\n\n"
        "**Buys:** Bulk grocery and premium items."
    ),
    "Habitual Replenishers": (
        "**Who are they?** Loyal routine shoppers.\n\n"
        "**Behavior:** Frequent medium baskets.\n\n"
        "**Buys:** Staples, dairy, produce."
    ),
    "Fill-in Convenience Shoppers": (
        "**Who are they?** Urgent top-up shoppers.\n\n"
        "**Behavior:** Small baskets.\n\n"
        "**Buys:** Snacks and beverages."
    ),
    "Low Engagement / Trial Users": (
        "**Who are they?** New or inactive users.\n\n"
        "**Behavior:** Low engagement."
    ),
    "Cold Start / Unsegmented": (
        "**Who are they?** Insufficient data."
    )
}

# =====================================================
# TABS
# =====================================================
tabs = st.tabs([
    "🚀 Overview",
    "📊 Transaction EDA",
    "🧪 Feature Engineering",
    "🧠 Segmentation",
    "🤖 Recommendation Systems",
    "🔍 Customer Deep Dive"
])

# =====================================================
# TAB 1 — OVERVIEW
# =====================================================
with tabs[0]:

    st.header("🚀 Project Overview")

    st.markdown("""
    ### 🎯 Objective
    Build an intelligent **personalization system** for online grocery retail by:

    - Understanding customer purchasing behavior  
    - Segmenting customers using unsupervised learning  
    - Delivering segment-aware product recommendations  
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{segment_distribution.customer_count.sum():,}"
    )
    col2.metric(
        "Customer Segments",
        segment_distribution.shape[0]
    )
    col3.metric(
        "Transaction Days",
        eda_daily.shape[0]
    )

    st.divider()

    st.subheader("🔗 System Architecture")

    st.code("""
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
Interactive Dashboard
""")

    st.subheader("📊 Segment Distribution")
    st.bar_chart(
        segment_distribution.set_index("segment_name")["customer_count"]
    )

# =====================================================
# TAB 2 — TRANSACTION EDA
# =====================================================
with tabs[1]:

    st.header("📊 Transaction-Level EDA")

    eda_daily["order_date"] = pd.to_datetime(eda_daily["order_date"])

    min_d = eda_daily["order_date"].min()
    max_d = eda_daily["order_date"].max()

    start, end = st.date_input(
        "Date Range",
        value=(min_d, max_d)
    )

    mask = (
        (eda_daily["order_date"] >= pd.to_datetime(start)) &
        (eda_daily["order_date"] <= pd.to_datetime(end))
    )

    df = eda_daily.loc[mask]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Orders", f"{df.orders.sum():,}")
    col2.metric("Revenue", f"${df.revenue.sum():,.0f}")
    col3.metric("Customers", f"{df.customers.sum():,}")
    col4.metric("Avg Basket", f"{df.avg_basket_size.mean():.2f}")

    st.subheader("📈 Orders & Revenue Trend")
    st.line_chart(df.set_index("order_date")[["orders", "revenue"]])

    st.subheader("🏷 Category Contribution")
    st.bar_chart(
        eda_category_summary
        .set_index("category")["total_revenue"]
    )

    st.subheader("📆 Orders by Weekday")
    st.bar_chart(
        eda_weekday.set_index("weekday")["orders"]
    )

    st.subheader("👥 Orders per Customer Distribution")
    st.bar_chart(
        eda_orders_per_customer["orders"]
        .value_counts()
        .sort_index()
    )

# =====================================================
# TAB 3 — FEATURE ENGINEERING
# =====================================================
with tabs[2]:

    st.header("🧪 Feature Engineering")

    st.markdown("""
    Features were derived from transaction data to capture:

    - Recency, frequency, monetary behavior  
    - Basket composition  
    - Shopping mission patterns  
    - Category diversity
    """)

    st.subheader("📊 Feature Distributions")
    st.dataframe(feature_dist)
    st.bar_chart(feature_dist.set_index("feature")["mean"])

    st.subheader("🔗 Feature Correlation")
    st.dataframe(feature_corr)

# =====================================================
# TAB 4 — SEGMENTATION (ENRICHED)
# =====================================================
with tabs[3]:

    st.header("🧠 Customer Segmentation")

    col1, col2, col3 = st.columns(3)

    top_spend = segment_kpis.sort_values(
        "avg_total_spend", ascending=False
    ).iloc[0]

    most_freq = segment_kpis.sort_values(
        "avg_orders", ascending=False
    ).iloc[0]

    recent = segment_kpis.sort_values(
        "avg_recency_days"
    ).iloc[0]

    col1.metric("Highest Spend Segment", top_spend.segment_name)
    col2.metric("Most Frequent Segment", most_freq.segment_name)
    col3.metric("Most Recent Segment", recent.segment_name)

    st.subheader("📊 Segment Distribution")
    st.bar_chart(
        segment_distribution.set_index("segment_name")["customer_count"]
    )

    st.subheader("📈 Segment KPIs")
    st.dataframe(segment_kpis)

    st.subheader("🧠 Segment Personas")
    for s, p in SEGMENT_PERSONAS.items():
        st.markdown(f"### {s}")
        st.markdown(p)

# =====================================================
# TAB 5 — SEGMENTATION-DRIVEN RECOMMENDATIONS
# =====================================================
with tabs[4]:

    st.header("🤖 Segmentation-Driven Recommendations")

    # =================================================
    # SECTION 1 — WHY SEGMENTATION IS NEEDED
    # =================================================
    st.subheader("📌 Why Do Recommendations Need Segmentation?")

    st.markdown("""
    In online grocery retail, customers shop with **different intentions**, such as:

    - Monthly stock-up shopping  
    - Routine weekly replenishment  
    - Urgent convenience purchases  

    Traditional recommendation systems treat all customers the same,
    ignoring these behavioral differences.

    As a result, recommendations may be **popular**, but not necessarily
    **relevant to the customer’s shopping mission**.

    Customer segmentation helps address this problem by grouping customers
    with similar behavioral patterns before generating recommendations.
    """)

    st.divider()

    # =================================================
    # SECTION 2 — WITHOUT SEGMENTATION
    # =================================================
    st.subheader("❌ Without Segmentation (Baseline Recommender)")

    st.markdown("""
    The baseline recommender uses **global item popularity**.

    - Same items are recommended to all customers  
    - Ignores purchase frequency, basket size, and shopping intent  
    - Does not adapt to customer behavior  
    """)

    st.markdown("**Top 5 Globally Popular Items:**")

    baseline_top5 = baseline_items.head(5)

    for _, row in baseline_top5.iterrows():
        st.markdown(
            f"• **{row.get('item_name', 'Unknown Item')}**"
        )

    st.info(
        "These recommendations remain identical regardless of whether "
        "the customer is a stock-up shopper, a frequent replenisher, "
        "or a convenience buyer."
    )

    st.divider()

    # =================================================
    # SECTION 3 — WITH SEGMENTATION
    # =================================================
    st.subheader("✅ With Segmentation (Segment-Aware Recommender)")

    st.markdown("""
    In the segment-aware approach:

    - Customers are first grouped using behavioral clustering  
    - Item co-occurrence patterns are learned **within each segment**  
    - Recommendations align with the dominant shopping mission  
    """)

    selected_segment = st.selectbox(
        "Select Customer Segment",
        segment_affinity["segment_name"].unique()
    )

    segment_recs = (
        segment_affinity[
            segment_affinity["segment_name"] == selected_segment
        ]
        .sort_values("rank")
        .head(30)
        .merge(
            df_item_lookup,
            on="asin",
            how="left"
        )
    )

    # simple category diversification
    selected = []
    seen_categories = set()

    for _, row in segment_recs.iterrows():
        category = row.get("uphl1")

        if category not in seen_categories:
            selected.append(row)
            seen_categories.add(category)

        if len(selected) == 5:
            break

    st.markdown(f"**Top 5 Recommendations for _{selected_segment}_ Segment:**")

    for row in selected:
        st.markdown(
            f"• **{row.get('item_name', 'Unknown Item')}**  \n"
            f"_Category: {row.get('uphl1', 'Unknown')}_"
        )

    st.success(
        "Recommendations differ across segments, reflecting differences "
        "in shopping intent and purchasing behavior."
    )

    st.divider()

    # =================================================
    # SECTION 4 — KEY TAKEAWAY
    # =================================================
    st.subheader("🎯 Key Takeaway")

    st.markdown("""
    **Customer segmentation enables behavior-aware personalization.**

    - Without segmentation → generic popularity-based recommendations  
    - With segmentation → mission-driven, relevant recommendations  

    This demonstrates how unsupervised customer clustering directly improves
    downstream personalization systems in online retail.
    """)


# =====================================================
# TAB 6 — CUSTOMER DEEP DIVE (ENRICHED)
# =====================================================
with tabs[5]:

    st.header("🔍 Customer Deep Dive")

    customer_id = st.text_input("Enter Customer ID")

    if not customer_id:
        st.stop()

    cust_seg = df_segments[
        df_segments.customer_id == customer_id
    ]

    if cust_seg.empty:
        st.warning("Customer not found")
        st.stop()

    segment_name = cust_seg.iloc[0]["segment_name"]

    st.success(f"Segment: {segment_name}")
    st.info(SEGMENT_PERSONAS.get(segment_name))

    shard = customer_id[4]
    df_feat = load_customer_feature_shard(shard)

    cust = df_feat[df_feat.customer_id == customer_id].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Orders", int(cust.orders))
    col2.metric("Spend", f"${cust.total_spend:.0f}")
    col3.metric("AOV", f"${cust.avg_order_value:.0f}")
    col4.metric("Recency", int(cust.days_since_last_order))

    seg_kpi = segment_kpis[
        segment_kpis.segment_name == segment_name
    ].iloc[0]

    comp = pd.DataFrame({
        "Metric": ["Orders", "Spend", "AOV", "Recency"],
        "Customer": [
            cust.orders,
            cust.total_spend,
            cust.avg_order_value,
            cust.days_since_last_order
        ],
        "Segment Avg": [
            seg_kpi.avg_orders,
            seg_kpi.avg_total_spend,
            seg_kpi.avg_order_value,
            seg_kpi.avg_recency_days
        ]
    })

    st.subheader("⚖️ Customer vs Segment")
    st.dataframe(comp)

    melt = comp.melt("Metric", var_name="Type", value_name="Value")

    chart = alt.Chart(melt).mark_bar().encode(
        x="Metric",
        y="Value",
        color="Type",
        column="Type"
    )

    st.altair_chart(chart, use_container_width=True)

    st.subheader("🎯 Recommendations")

    recs = (
        segment_affinity[
            segment_affinity.segment_name == segment_name
        ]
        .sort_values("rank")
        .head(40)
        .merge(df_item_lookup, on="asin", how="left")
    )

    seen = set()
    final = []

    for _, r in recs.iterrows():
        cat = r.get("uphl1")
        if cat not in seen:
            final.append(r)
            seen.add(cat)
        if len(final) == 5:
            break

    for r in final:
        st.markdown(
            f"• **{r['item_name']}**  \n"
            f"_Category: {r.get('uphl1','Unknown')}_"
        )

    with st.expander("🤔 Why am I seeing this?"):
        st.markdown(f"""
        - You belong to **{segment_name}**
        - Recommendations are learned from customers with similar behavior
        - Items are co-purchased frequently within this segment
        - Results are category diversified
        """)
