import streamlit as st
import joblib
import numpy as np

# Load models
sentiment_model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf.pkl")
rating_model = joblib.load("rating_model.pkl")

st.title("🧴 AI Skincare Product Analyzer")

# -------------------------
# SENTIMENT ANALYSIS
# -------------------------
st.header("Sentiment Analysis")

review = st.text_area("Enter Review Text")

if st.button("Predict Sentiment"):
    review_vec = tfidf.transform([review])
    sentiment = sentiment_model.predict(review_vec)[0]
    st.success(f"Predicted Sentiment: {sentiment}")

# -------------------------
# RATING PREDICTION
# -------------------------
st.header("Rating Prediction")

# Fixed category options (safe)
brand = st.selectbox(
    "Brand",
    ["GlowCare", "SkinPure", "DermaLux", "BeautyPlus"]
)

category = st.selectbox(
    "Category",
    ["Moisturizer", "Serum", "Cleanser", "Sunscreen"]
)

price_range = st.selectbox(
    "Price Range",
    ["Low", "Medium", "High"]
)

verified = st.selectbox(
    "Verified Purchase",
    ["Yes", "No"]
)

# Manual encoding (SAFE)
brand_map = {
    "GlowCare": 0,
    "SkinPure": 1,
    "DermaLux": 2,
    "BeautyPlus": 3
}

category_map = {
    "Moisturizer": 0,
    "Serum": 1,
    "Cleanser": 2,
    "Sunscreen": 3
}

price_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

verified_map = {
    "Yes": 1,
    "No": 0
}

if st.button("Predict Rating"):
    input_features = np.array([[
        brand_map[brand],
        category_map[category],
        price_map[price_range],
        verified_map[verified]
    ]])

    rating = rating_model.predict(input_features)[0]
    st.success(f"Predicted Rating: {round(rating, 1)} ⭐")
