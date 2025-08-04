import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Step 1: Import and run the data fetcher
from fetch_files import download_files
download_files()

# Step 2: Continue with the main app
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure Streamlit
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-bottom: 1rem;
    }
    .book-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        book_df = pd.read_csv('data/Books.csv', encoding='latin-1')
        user_df = pd.read_csv('data/Users.csv', encoding='latin-1')
        rating_df = pd.read_csv('data/Ratings.csv', encoding='latin-1')

        book_df['ISBN'] = book_df['ISBN'].str.upper()
        book_df['Book-Author'] = book_df['Book-Author'].str.upper()
        user_df['Country'] = user_df['Location'].str.extract(r',\s*([^,]+)\s*$')
        user_df['Country'] = user_df['Country'].str.strip().str.upper()
        user_df['Age'].fillna(user_df['Age'].mean(), inplace=True)

        rating_with_name = rating_df.merge(book_df, on='ISBN')
        explicit_rating = rating_df[rating_df['Book-Rating'] != 0]
        merged_df = book_df.merge(explicit_rating, on='ISBN')
        df_new = merged_df.merge(user_df, on='User-ID')

        return book_df, user_df, rating_df, df_new, rating_with_name
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# RECOMMENDER AND UI UTILS – You can paste your existing functions like:
# - most_popular()
# - country_popular()
# - get_collaborative_recommendations()
# - display_books()

# Load data
book_df, user_df, rating_df, df_new, rating_with_name = load_data()

if book_df is not None:
    st.markdown("<h1 class='main-header'>📚 Book Recommendation System</h1>", unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox(
        "Choose Recommendation Type:",
        ["Dashboard", "Most Popular Books", "Country-based Recommendations", "Collaborative Filtering"]
    )

    # Your logic from here remains same as in your app (dashboard, charts, filtering, etc.)

else:
    st.error("Failed to load data. Please check if CSV files were fetched successfully.")
