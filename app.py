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

# ---------------------------------------------
# Load Customer Segments (Parquet)
# ---------------------------------------------
@st.cache_data
def load_customer_segments():
    return pd.read_parquet("data/customer_segments.parquet")

df_segments = load_customer_segments()


# ---------------------------------------------
# Segment Persona Definitions  ✅ MUST BE HERE
# ---------------------------------------------
# ---------------------------------------------
# Segment Persona Definitions (Detailed)
# ---------------------------------------------
SEGMENT_PERSONAS = {

    "Large Basket Stock-up": (
        "**Who are they?**\n"
        "High-spending households, often families, who complete their full or weekly grocery shopping online. "
        "They typically have higher disposable incomes and a strong affinity towards Amazon Fresh.\n\n"

        "**General shopping behavior:**\n"
        "These customers place infrequent but very large orders, often exceeding a full weekly shop. "
        "They usually know exactly what they want to buy and rely on Amazon Fresh for both routine and "
        "special-occasion shopping.\n\n"

        "**Underlying needs & motivations:**\n"
        "They value reliability, fast delivery, and wide selection. Grocery shopping online helps them "
        "save time, stay organized, and spend more time with family. They enjoy discovering new products, "
        "meal kits, and cuisines, but prefer experiences that are seamless and efficient.\n\n"

        "**What they buy:**\n"
        "A complete grocery basket across all major categories. They tend to over-index on premium items "
        "such as wine, spirits, and ready-to-eat meals."
    ),

    "Habitual Replenishers": (
        "**Who are they?**\n"
        "Regular, loyal customers who rely on Amazon Fresh for frequent grocery replenishment. "
        "Often families or steady households with predictable consumption patterns.\n\n"

        "**General shopping behavior:**\n"
        "They place frequent orders with smaller-to-medium basket sizes, typically restocking everyday essentials. "
        "Their shopping is routine-driven and planned.\n\n"

        "**Underlying needs & motivations:**\n"
        "They value convenience, consistency, and trust. Shopping online gives them a sense of control "
        "and helps them manage daily life efficiently. They appreciate good deals, ease of reordering, "
        "and reliable availability.\n\n"

        "**What they buy:**\n"
        "Fresh produce, dairy, pantry staples, and household essentials. They tend to repurchase the same "
        "set of items regularly."
    ),

    "Fill-in Convenience Shoppers": (
        "**Who are they?**\n"
        "Convenience-driven customers making quick, top-up purchases for immediate needs or forgotten items.\n\n"

        "**General shopping behavior:**\n"
        "They place small, frequent orders, often driven by urgency rather than routine planning. "
        "These trips are usually short and focused.\n\n"

        "**Underlying needs & motivations:**\n"
        "Speed and ease are critical. They use Amazon Fresh to quickly solve a need — saving time, "
        "avoiding physical store visits, and ensuring immediate availability.\n\n"

        "**What they buy:**\n"
        "Top-up items such as snacks, ready meals, beverages, and last-minute essentials."
    ),

    "Low Engagement / Trial Users": (
        "**Who are they?**\n"
        "New or infrequent users with limited interaction history on Amazon Fresh.\n\n"

        "**General shopping behavior:**\n"
        "They place very few orders and have not yet formed a consistent shopping habit on the platform.\n\n"

        "**Underlying needs & motivations:**\n"
        "These customers are still exploring the service. They may be testing delivery quality, pricing, "
        "or product availability before committing to regular usage.\n\n"

        "**What they buy:**\n"
        "Small baskets with limited category diversity, often focused on trial or one-off purchases."
    ),

    "Cold Start / Unsegmented": (
        "**Who are they?**\n"
        "Customers with insufficient historical data to confidently assign a behavioral segment.\n\n"

        "**General shopping behavior:**\n"
        "Either very new customers or those with extremely sparse purchase activity.\n\n"

        "**Underlying needs & motivations:**\n"
        "Not enough data to infer stable preferences yet. These customers represent an opportunity "
        "for onboarding, discovery, and engagement strategies.\n\n"

        "**What they buy:**\n"
        "Highly variable — depends on initial interaction with the platform."
    )
}


# ---------------------------------------------
# Customer Lookup
# ---------------------------------------------
st.divider()
st.header("🔍 Customer Segment Lookup")

customer_id_input = st.text_input(
    "Enter Customer ID (e.g. CUST123456)",
    placeholder="CUSTxxxxxxx"
)

if customer_id_input:
    customer_row = df_segments[
        df_segments["customer_id"] == customer_id_input
    ]

    if customer_row.empty:
        st.warning("Customer ID not found in segmentation data.")
    else:
        segment_name = customer_row["segment_name"].values[0]

        st.success("Customer found!")
        st.metric("Segment", segment_name)
        
        persona_text = SEGMENT_PERSONAS.get(
            segment_name,
            "No persona description available."
        )
        
        st.info(persona_text)

# ---------------------------------------------
# Segment Distribution Overview
# ---------------------------------------------
st.divider()
st.header("📊 Segment Distribution Overview")

# Compute distribution
segment_counts = (
    df_segments["segment_name"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["segment_name", "customer_count"]

# Percentage share
total_customers = segment_counts["customer_count"].sum()
segment_counts["share_pct"] = (
    segment_counts["customer_count"] / total_customers * 100
).round(2)

# Display table
st.subheader("Segment-wise Customer Counts")
st.dataframe(segment_counts)

# Bar chart
st.subheader("Customer Share by Segment")
st.bar_chart(
    segment_counts.set_index("segment_name")["customer_count"]
)

