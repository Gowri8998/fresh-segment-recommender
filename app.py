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
eda_category_summary = load_parquet("eda_category_summary.parquet")

eda_weekday = load_parquet("eda_weekday_summary.parquet")
eda_orders_per_customer = load_parquet("eda_orders_per_customer.parquet")


segment_dist = load_parquet("segment_distribution.parquet")
evaluation_summary = load_parquet("recommender_evaluation_summary.parquet")

feature_dist = load_parquet("feature_distribution_summary.parquet")
feature_corr = load_parquet("feature_correlation.parquet")

segment_kpis = load_parquet("segment_kpis.parquet")
segment_distribution = load_parquet("segment_distribution.parquet")

baseline_items = load_parquet("baseline_top_items.parquet")
segment_affinity = load_parquet("segment_item_affinity.parquet")


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
    "🔍 Customer Deep Dive"
])

tab_overview = tabs[0]
tab_eda = tabs[1]
tab_features = tabs[2]
tab_segments = tabs[3]
tab_recs = tabs[4]
tab_customer = tabs[5]


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


# =============================================
# TAB 2 — TRANSACTION EDA
# =============================================
with tab_eda:

    st.header("📊 Transaction-Level Exploratory Data Analysis")

    st.markdown(
        "This section explores high-level customer purchasing behavior "
        "using transaction-level summaries derived from the raw dataset."
    )

    st.divider()

    # -----------------------------------------
    # Date Range Filter
    # -----------------------------------------
    st.subheader("📅 Select Analysis Period")

    eda_daily["order_date"] = pd.to_datetime(eda_daily["order_date"])

    min_date = eda_daily["order_date"].min()
    max_date = eda_daily["order_date"].max()

    start_date, end_date = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    mask = (
        (eda_daily["order_date"] >= pd.to_datetime(start_date)) &
        (eda_daily["order_date"] <= pd.to_datetime(end_date))
    )

    filtered_daily = eda_daily.loc[mask]

    st.divider()

    # -----------------------------------------
    # Key Metrics
    # -----------------------------------------
    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Orders", f"{filtered_daily.orders.sum():,}")
    col2.metric("Total Revenue", f"${filtered_daily.revenue.sum():,.0f}")
    col3.metric("Active Customers", f"{filtered_daily.customers.sum():,}")
    col4.metric(
        "Avg Basket Size",
        f"{filtered_daily.avg_basket_size.mean():.2f}"
    )

    st.divider()

    # -----------------------------------------
    # Trends Over Time
    # -----------------------------------------
    st.subheader("📈 Behavioral Trends Over Time")

    st.line_chart(
        filtered_daily.set_index("order_date")[["orders", "revenue"]]
    )

    st.divider()

    # -----------------------------------------
    # Basket Size Trend
    # -----------------------------------------
    st.subheader("🧺 Average Basket Size Trend")

    st.line_chart(
        filtered_daily.set_index("order_date")["avg_basket_size"]
    )

    st.divider()

    # -----------------------------------------
    # Category Contribution
    # -----------------------------------------
    st.subheader("🏷 Category Contribution")

    st.dataframe(
        eda_category_summary
        .sort_values("total_revenue", ascending=False)
    )

    st.bar_chart(
        eda_category_summary
        .set_index("category")["total_revenue"]
    )

    st.info(
        "Category-level analysis helps identify major revenue drivers "
        "and informs segment-specific merchandising strategies."
    )

    st.divider()
    st.subheader("📆 Orders by Day of Week")
    
    st.bar_chart(
        eda_weekday.set_index("weekday")["orders"]
    )
    
    st.info(
        "Shopping activity varies across the week, "
        "with noticeable differences between weekday "
        "and weekend behavior."
    )
    
    st.divider()
    st.subheader("👥 Orders per Customer Distribution")
    
    st.bar_chart(
        eda_orders_per_customer["orders"]
        .value_counts()
        .sort_index()
    )
    
    st.info(
        "Customer activity is highly skewed — a small "
        "set of customers place many orders while the "
        "majority transact infrequently. This motivates "
        "customer segmentation and personalized recommendations."
    )

# =============================================
# TAB 3 — FEATURE ENGINEERING
# =============================================
with tab_features:

    st.header("🧪 Feature Engineering Insights")

    st.markdown(
        "Customer segmentation was driven by behavioral features engineered "
        "from raw transaction data. This section explains feature construction "
        "and validates their analytical usefulness."
    )

    st.divider()

    # -----------------------------------------
    # Feature Overview
    # -----------------------------------------
    st.subheader("🔍 Engineered Feature Groups")

    st.markdown("""
    **1. RFM & Engagement Features**
    - Orders  
    - Total spend  
    - Days since last order  

    **2. Basket Behavior Features**
    - Average units per order  
    - Average order value  

    **3. Trip Mission Composition**
    - Fill-in shopping %  
    - Routine shopping %  
    - Large basket %  

    **4. Category Breadth**
    - Category diversity
    """)

    st.divider()

    # -----------------------------------------
    # Feature Distribution Summary
    # -----------------------------------------
    st.subheader("📊 Feature Distributions")

    st.dataframe(feature_dist)

    st.bar_chart(
        feature_dist.set_index("feature")["mean"]
    )

    st.info(
        "Feature distributions reveal strong skewness and heterogeneity "
        "across customers, reinforcing the need for clustering-based "
        "segmentation instead of rule-based grouping."
    )

    st.divider()

    # -----------------------------------------
    # Feature Correlation
    # -----------------------------------------
    st.subheader("🔗 Feature Correlation Analysis")

    st.dataframe(feature_corr)

    st.info(
        "Correlation analysis was used to remove redundant features prior "
        "to clustering, improving stability and interpretability of segments."
    )


# =============================================
# TAB 4 — SEGMENTATION INSIGHTS
# =============================================
with tab_segments:

    st.header("🧠 Customer Segmentation Insights")

    st.markdown(
        "Customers were segmented using unsupervised learning on behavioral "
        "features derived from transaction history. This section interprets "
        "the resulting segments and highlights key behavioral differences."
    )

    st.divider()

    # -----------------------------------------
    # Segment Distribution
    # -----------------------------------------
    st.subheader("📊 Segment Size Distribution")

    st.bar_chart(
        segment_distribution
        .set_index("segment_name")["customer_count"]
    )

    st.divider()

    # -----------------------------------------
    # Segment KPI Overview
    # -----------------------------------------
    st.subheader("📈 Segment-Level Behavioral KPIs")

    st.dataframe(segment_kpis)

    st.info(
        "Segment KPIs summarize average behavioral patterns within each cluster, "
        "revealing clear differences in spend, engagement, and shopping missions."
    )

    st.divider()

    # -----------------------------------------
    # KPI Comparison Chart
    # -----------------------------------------
    st.subheader("📊 Key KPI Comparison")

    comparison_cols = [
        "avg_orders",
        "avg_total_spend",
        "avg_order_value",
        "avg_recency_days"
    ]

    st.bar_chart(
        segment_kpis
        .set_index("segment_name")[comparison_cols]
    )

    st.info(
        "These comparisons demonstrate why a single global recommendation "
        "strategy is suboptimal. Different customer segments exhibit "
        "distinct shopping behaviors requiring tailored personalization."
    )

# =============================================
# TAB 5 — RECOMMENDATION SYSTEMS
# =============================================
with tab_recs:

    st.header("🤖 Recommendation Systems")

    st.markdown("""
    This section demonstrates two recommendation approaches:

    **1️⃣ Baseline Recommender**  
    Popular items recommended uniformly to all customers.

    **2️⃣ Segment-Aware Recommender**  
    Personalized recommendations based on customer shopping behavior segments.
    """)

    st.divider()

    # -----------------------------------------
    # Baseline recommender
    # -----------------------------------------
    st.subheader("⭐ Baseline: Most Popular Items")

    st.dataframe(
        baseline_items.head(10)
    )

    st.info(
        "The baseline recommender ignores customer behavior and recommends "
        "globally popular items. While simple, it lacks personalization."
    )

    st.divider()

    # -----------------------------------------
    # Segment-aware recommender
    # -----------------------------------------
    st.subheader("🎯 Segment-Aware Recommendations")

    selected_segment = st.selectbox(
        "Select customer segment",
        segment_affinity["segment_name"].unique()
    )

    top_recs = (
        segment_affinity[
            segment_affinity["segment_name"] == selected_segment
        ]
        .sort_values("rank")
        .head(10)
    )

    st.dataframe(top_recs)

    st.success(
        "Segment-aware recommendations capture differences in "
        "shopping missions such as stock-up, routine replenishment, "
        "and convenience trips."
    )

    st.divider()

    # -----------------------------------------
    # Explanation
    # -----------------------------------------
    st.subheader("🧠 How Segment-Aware Recommendation Works")

    st.markdown("""
    **Step 1 — Item Co-occurrence Learning**  
    Products frequently purchased together are identified using transaction data.

    **Step 2 — Segment Filtering**  
    Item affinities are computed separately within each customer segment.

    **Step 3 — Diversification**  
    Recommendations are diversified across categories to improve discovery.

    This approach balances **relevance**, **diversity**, and **interpretability**.
    """)


# =============================================
# TAB 6 — EVALUATION
# =============================================
with tab_eval:

    st.header("📈 Recommendation Evaluation")

    st.markdown("""
    Recommendation models were evaluated using offline metrics
    on a held-out customer test set.
    """)

    st.divider()

    # -----------------------------------------
    # Show raw evaluation table
    # -----------------------------------------
    st.subheader("📊 Evaluation Summary")

    st.dataframe(evaluation_summary)

    st.divider()

    # -----------------------------------------
    # Auto-detect metric columns
    # -----------------------------------------
    possible_metric_cols = [
        "precision_at_k", "precision",
        "recall_at_k", "recall",
        "hit_rate", "hit_rate_at_k", "hit"
    ]

    metric_cols = [
        c for c in evaluation_summary.columns
        if c.lower() in possible_metric_cols
        or any(x in c.lower() for x in ["precision", "recall", "hit"])
    ]

    # detect model column
    model_col = None
    for c in evaluation_summary.columns:
        if c.lower() in ["model", "model_name", "recommender", "algorithm"]:
            model_col = c
            break

    if model_col is None:
        st.error("Could not detect model column in evaluation file.")
        st.stop()

    # -----------------------------------------
    # Comparison chart
    # -----------------------------------------
    st.subheader("📈 Model Comparison")

    chart_df = evaluation_summary.set_index(model_col)[metric_cols]

    st.bar_chart(chart_df)

    st.success(
        "Segment-aware recommender outperforms the baseline "
        "across multiple evaluation metrics."
    )

    st.divider()

    # -----------------------------------------
    # Interpretation
    # -----------------------------------------
    st.subheader("🧠 Interpretation")

    st.markdown("""
    **Key takeaways:**

    • Personalized models outperform popularity-based baselines  
    • Behavioral segmentation improves recommendation relevance  
    • Offline evaluation confirms measurable uplift  

    These results justify the use of segment-aware recommendation
    strategies in retail personalization systems.
    """)

# =============================================
# TAB 7 — CUSTOMER DEEP DIVE
# =============================================
with tab_customer:

    st.header("🔍 Customer Deep Dive")

    st.markdown(
        "Explore individual customer behavior, segment assignment, "
        "and personalized product recommendations."
    )

    st.divider()

    # -----------------------------------------
    # Customer Input
    # -----------------------------------------
    customer_id = st.text_input(
        "Enter Customer ID",
        placeholder="CUST1234567"
    )

    if not customer_id:
        st.stop()

    # -----------------------------------------
    # Segment Lookup
    # -----------------------------------------
    cust_seg = df_segments[
        df_segments["customer_id"] == customer_id
    ]

    if cust_seg.empty:
        st.warning("Customer not found in segmentation data.")
        st.stop()

    segment_name = cust_seg.iloc[0]["segment_name"]

    st.success(f"Customer Segment: {segment_name}")

    # -----------------------------------------
    # Persona
    # -----------------------------------------
    persona = SEGMENT_PERSONAS.get(
        segment_name,
        "Persona not defined."
    )

    st.info(persona)

    # -----------------------------------------
    # Load customer features shard
    # -----------------------------------------
    shard_key = customer_id[4]  # CUSTX...
    cust_features_df = load_customer_feature_shard(shard_key)

    cust_feat = cust_features_df[
        cust_features_df["customer_id"] == customer_id
    ]

    if cust_feat.empty:
        st.warning("Customer feature data not available.")
        st.stop()

    cust_feat = cust_feat.iloc[0]

    # -----------------------------------------
    # KPIs
    # -----------------------------------------
    st.subheader("📊 Customer Behavioral KPIs")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Orders", int(cust_feat["orders"]))
    col2.metric("Total Spend", f"${cust_feat['total_spend']:.0f}")
    col3.metric("Avg Order Value", f"${cust_feat['avg_order_value']:.2f}")
    col4.metric("Recency (Days)", int(cust_feat["days_since_last_order"]))
    col5.metric("Category Diversity", int(cust_feat["category_diversity"]))

    # -----------------------------------------
    # Segment comparison
    # -----------------------------------------
    st.subheader("⚖️ Customer vs Segment Average")

    seg_kpi = segment_kpis[
        segment_kpis["segment_name"] == segment_name
    ].iloc[0]

    comparison_df = pd.DataFrame({
        "Metric": [
            "Orders",
            "Total Spend",
            "Avg Order Value",
            "Recency",
            "Category Diversity"
        ],
        "Customer": [
            cust_feat["orders"],
            cust_feat["total_spend"],
            cust_feat["avg_order_value"],
            cust_feat["days_since_last_order"],
            cust_feat["category_diversity"]
        ],
        "Segment Avg": [
            seg_kpi["avg_orders"],
            seg_kpi["avg_total_spend"],
            seg_kpi["avg_order_value"],
            seg_kpi["avg_recency_days"],
            seg_kpi["avg_category_diversity"]
        ]
    })

    st.dataframe(comparison_df)

    # -----------------------------------------
    # Recommendations
    # -----------------------------------------
    st.subheader("🎯 Personalized Recommendations")

    recs = (
        segment_affinity[
            segment_affinity["segment_name"] == segment_name
        ]
        .sort_values("rank")
        .head(30)
        .merge(
            item_lookup,
            on="asin",
            how="left"
        )
    )

    # diversify by category
    selected = []
    seen = set()

    for _, row in recs.iterrows():
        cat = row.get("uphl1")

        if cat not in seen:
            selected.append(row)
            seen.add(cat)

        if len(selected) == 5:
            break

    if not selected:
        st.warning("No recommendations available.")
    else:
        for r in selected:
            st.markdown(
                f"• **{r['item_name']}**  \n"
                f"_Category: {r.get('uphl1', 'Unknown')}_"
            )

    st.caption(
        "Recommendations are segment-aware, co-occurrence based, "
        "and category-diversified to improve discovery."
    )
