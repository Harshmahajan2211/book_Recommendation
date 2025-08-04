import gdown
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Download CSVs from Drive
def fetch_csv_from_drive():
    if not os.path.exists("data"):
        os.makedirs("data")

    file_ids = {
        "Books.csv": "1EeMLA8mBSLTXJcszy6JLv1qFnGUY89UP",
        "Ratings.csv": "10rOVJ7t2-nMCjBixo3Uiw4qABhC3vMol",
        "Users.csv": "1wwN3NmAvdWBK9ILY9F68KRjty8UbZC0o"
    }

    for filename, file_id in file_ids.items():
        output_path = os.path.join("data", filename)
        if not os.path.exists(output_path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Run the fetch before anything else
fetch_csv_from_drive()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
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
    """Load and preprocess the data"""
    try:
        # Load data
        book_df = pd.read_csv('Books.csv', encoding='latin-1')
        user_df = pd.read_csv('Users.csv', encoding='latin-1')
        rating_df = pd.read_csv('Ratings.csv', encoding='latin-1')
        
        # Basic preprocessing
        book_df['ISBN'] = book_df['ISBN'].str.upper()
        book_df['Book-Author'] = book_df['Book-Author'].str.upper()
        
        # Extract country from location
        user_df['Country'] = user_df['Location'].str.extract(r',\s*([^,]+)\s*$')
        user_df['Country'] = user_df['Country'].str.strip().str.upper()
        
        # Fill missing ages
        mean_age = user_df['Age'].mean()
        user_df['Age'].fillna(mean_age, inplace=True)
        
        # Create merged dataframes
        rating_with_name = rating_df.merge(book_df, on='ISBN')
        explicit_rating = rating_df[rating_df['Book-Rating'] != 0]
        merged_df = book_df.merge(explicit_rating, on='ISBN')
        df_new = merged_df.merge(user_df, on='User-ID')
        
        return book_df, user_df, rating_df, df_new, rating_with_name
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

def most_popular(df_new, n=5):
    """Get most popular books based on rating count"""
    popular = df_new.groupby('ISBN')['Book-Rating'].count().reset_index()
    popular = popular.sort_values(by='Book-Rating', ascending=False).head(n)
    return popular.merge(book_df, on='ISBN')

def country_popular(df_new, country, n=5):
    """Get popular books by country"""
    country_upper = country.upper()
    if country_upper in df_new['Country'].values:
        country_data = df_new[df_new['Country'] == country_upper]
        return most_popular(country_data, n)
    else:
        return None

def get_collaborative_recommendations(book_name, df_new, rating_with_name):
    """Get collaborative filtering recommendations"""
    try:
        # More aggressive filtering for large datasets
        # Filter users with more than 20 ratings (increased from 5)
        df_collab = rating_with_name.groupby('User-ID').count()['Book-Title'] > 20
        df_user = df_collab[df_collab].index
        
        if len(df_user) == 0:
            return None
            
        # Limit to top active users to reduce memory usage
        user_counts = rating_with_name.groupby('User-ID').count()['Book-Title']
        top_users = user_counts.nlargest(1000).index  # Limit to top 1000 users
        df_user = df_user.intersection(top_users)
        
        if len(df_user) == 0:
            return None
            
        filtered_rating = rating_with_name[rating_with_name['User-ID'].isin(df_user)]
        
        # Filter books with more ratings and limit to popular books
        df_famous = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 10
        famous_book = df_famous[df_famous].index
        
        # Limit to top 2000 most rated books to control memory
        book_counts = filtered_rating.groupby('Book-Title').count()['Book-Rating']
        top_books = book_counts.nlargest(2000).index
        famous_book = famous_book.intersection(top_books)
        
        if len(famous_book) == 0:
            return None
            
        final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_book)]
        
        # Check if we have enough data
        if final_ratings.empty:
            return None
            
        # Check if the selected book is in our filtered data
        if book_name not in famous_book:
            # Try to find the book in the original dataset and add it
            book_ratings = rating_with_name[rating_with_name['Book-Title'] == book_name]
            if book_ratings.empty:
                return None
            
            # Add the selected book to the analysis even if it doesn't meet criteria
            final_ratings = pd.concat([final_ratings, book_ratings], ignore_index=True)
            final_ratings = final_ratings.drop_duplicates()
        
        # Create pivot table with memory optimization
        df_pt = final_ratings.pivot_table(
            index='Book-Title', 
            columns='User-ID', 
            values='Book-Rating',
            fill_value=0
        )
        
        # Further reduce size if still too large
        if df_pt.shape[0] * df_pt.shape[1] > 5000000:  # If matrix > 5M elements
            # Keep only books with most ratings
            book_rating_counts = df_pt.sum(axis=1)
            top_books_final = book_rating_counts.nlargest(1000).index
            
            # Always include the selected book
            if book_name not in top_books_final:
                top_books_final = top_books_final.union([book_name])
            
            df_pt = df_pt.loc[top_books_final]
        
        # Check if the selected book is still in our data
        if book_name not in df_pt.index:
            return None
            
        # Check if we have enough books for comparison
        if len(df_pt) < 2:
            return None
        
        # Use memory-efficient similarity calculation
        try:
            from scipy.sparse import csr_matrix
            # Convert to sparse matrix for memory efficiency
            sparse_matrix = csr_matrix(df_pt.values)
            
            # Calculate similarity only for the selected book
            book_index = df_pt.index.get_loc(book_name)
            book_vector = sparse_matrix[book_index:book_index+1]
            
            # Calculate similarity with all other books
            similarities = cosine_similarity(book_vector, sparse_matrix).flatten()
            
        except ImportError:
            # Fallback: Calculate similarity without scipy (less memory efficient)
            st.warning("Using fallback similarity calculation. Install scipy for better performance.")
            
            # Get the selected book's ratings
            book_index = df_pt.index.get_loc(book_name)
            book_ratings = df_pt.iloc[book_index:book_index+1]
            
            # Calculate similarity with a subset of books at a time to manage memory
            similarities = np.zeros(len(df_pt))
            batch_size = 500  # Process books in batches
            
            for i in range(0, len(df_pt), batch_size):
                end_idx = min(i + batch_size, len(df_pt))
                batch_data = df_pt.iloc[i:end_idx]
                batch_similarities = cosine_similarity(book_ratings, batch_data).flatten()
                similarities[i:end_idx] = batch_similarities
        
        # Get indices of most similar books (excluding the book itself)
        similar_indices = similarities.argsort()[::-1]
        similar_indices = similar_indices[similar_indices != book_index][:5]
        
        recommendations = []
        for idx in similar_indices:
            similarity_score = similarities[idx]
            if similarity_score > 0:  # Only include books with positive similarity
                book_title = df_pt.index[idx]
                book_info = df_new[df_new['Book-Title'] == book_title].drop_duplicates('Book-Title')
                if not book_info.empty:
                    recommendations.append({
                        'Book-Title': book_title,
                        'Book-Author': book_info['Book-Author'].iloc[0],
                        'Similarity': f"{similarity_score:.3f}"
                    })
        
        return recommendations if recommendations else None
        
    except MemoryError:
        st.error("Dataset is too large for collaborative filtering. Try using other recommendation types.")
        return None
    except Exception as e:
        st.error(f"Error in collaborative filtering: {e}")
        return None

def display_books(books_df, title="Books"):
    """Display books in a nice format"""
    if books_df is not None and not books_df.empty:
        st.markdown(f"<div class='sub-header'>{title}</div>", unsafe_allow_html=True)
        
        for idx, book in books_df.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    try:
                        st.image(book.get('Image-URL-M', ''), width=100)
                    except:
                        st.write("📚")
                
                with col2:
                    st.markdown(f"""
                    <div class='book-card'>
                        <h4>{book['Book-Title']}</h4>
                        <p><strong>Author:</strong> {book['Book-Author']}</p>
                        <p><strong>Publisher:</strong> {book.get('Publisher', 'N/A')}</p>
                        <p><strong>Year:</strong> {book.get('Year-Of-Publication', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("No books found for the given criteria.")

# Load data
book_df, user_df, rating_df, df_new, rating_with_name = load_data()

if book_df is not None:
    # Main header
    st.markdown("<h1 class='main-header'>📚 Book Recommendation System</h1>", unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox(
        "Choose Recommendation Type:",
        ["Dashboard", "Most Popular Books", "Country-based Recommendations", "Collaborative Filtering"]
    )
    
    if option == "Dashboard":
        st.markdown("<div class='sub-header'>📊 Dataset Overview</div>", unsafe_allow_html=True)
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{len(book_df):,}</h3>
                <p>Total Books</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{len(user_df):,}</h3>
                <p>Total Users</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{len(rating_df):,}</h3>
                <p>Total Ratings</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{book_df['Book-Author'].nunique():,}</h3>
                <p>Unique Authors</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Top publishers chart
        st.markdown("<div class='sub-header'>📈 Top Publishers</div>", unsafe_allow_html=True)
        top_publishers = book_df['Publisher'].value_counts().head(10)
        st.bar_chart(top_publishers)
        
        # Top authors chart
        st.markdown("<div class='sub-header'>✍ Top Authors</div>", unsafe_allow_html=True)
        top_authors = book_df['Book-Author'].value_counts().head(10)
        st.bar_chart(top_authors)
        
        # Rating distribution
        if not rating_df.empty:
            st.markdown("<div class='sub-header'>⭐ Rating Distribution</div>", unsafe_allow_html=True)
            rating_dist = rating_df['Book-Rating'].value_counts().sort_index()
            st.bar_chart(rating_dist)
    
    elif option == "Most Popular Books":
        st.markdown("<div class='sub-header'>🔥 Most Popular Books</div>", unsafe_allow_html=True)
        
        num_books = st.slider("Number of books to display:", 1, 20, 5)
        
        if st.button("Get Popular Books"):
            popular_books = most_popular(df_new, num_books)
            display_books(popular_books, f"Top {num_books} Most Popular Books")
    
    elif option == "Country-based Recommendations":
        st.markdown("<div class='sub-header'>🌍 Country-based Recommendations</div>", unsafe_allow_html=True)
        
        # Get available countries
        available_countries = sorted(df_new['Country'].dropna().unique())
        
        country = st.selectbox("Select a country:", available_countries)
        num_books = st.slider("Number of books to display:", 1, 10, 5)
        
        if st.button("Get Country Recommendations"):
            country_books = country_popular(df_new, country, num_books)
            if country_books is not None:
                display_books(country_books, f"Popular Books in {country}")
            else:
                st.warning(f"No data found for {country}")
    
    elif option == "Collaborative Filtering":
        st.markdown("<div class='sub-header'>🤝 Collaborative Filtering Recommendations</div>", unsafe_allow_html=True)
        
        # Get available books
        available_books = sorted(df_new['Book-Title'].unique())
        
        selected_book = st.selectbox("Select a book you like:", available_books)
        
        if st.button("Get Similar Books"):
            with st.spinner("Finding similar books..."):
                recommendations = get_collaborative_recommendations(selected_book, df_new, rating_with_name)
                
                if recommendations:
                    st.success(f"Books similar to '{selected_book}':")
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"""
                        <div class='book-card'>
                            <h4>{i}. {rec['Book-Title']}</h4>
                            <p><strong>Author:</strong> {rec['Book-Author']}</p>
                            <p><strong>Similarity Score:</strong> {rec['Similarity']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Could not find similar books. Try selecting a different book.")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤ using Streamlit | Book Recommendation System")

else:
    st.error("Failed to load data. Please check if the CSV files are in the correct location.")