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

# ---------------------------------------------
# Segment Persona Definitions
# ---------------------------------------------
SEGMENT_PERSONAS = {
    "Large Basket Stock-up": (
        "Customers who place infrequent but large orders, "
        "typically stocking up on groceries and household essentials."
    ),
    "Habitual Replenishers": (
        "Regular shoppers who place frequent orders with smaller baskets, "
        "often replenishing everyday essentials."
    ),
    "Fill-in Convenience Shoppers": (
        "Customers who make quick, convenience-driven purchases, "
        "often for immediate consumption or missing items."
    ),
    "Low Engagement / Trial Users": (
        "Customers with limited or infrequent activity, "
        "often new or trial users of the platform."
    ),
    "Cold Start / Unsegmented": (
        "Customers with insufficient purchase history for behavioral segmentation."
    )
}

        st.success("Customer found!")
        st.metric("Segment", segment_name)
        
        persona_text = SEGMENT_PERSONAS.get(
            segment_name,
            "No persona description available."
        )
        
        st.info(persona_text)
