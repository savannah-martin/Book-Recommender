import streamlit as st
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import load_npz
from joblib import load
import pickle
import difflib

# Load the model using a relative path
try:
    model = load('model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please ensure the model file is in the same directory as your Streamlit app.")

# Load the matrix
with open('rating_pivot.pickle', 'rb') as f:
    rating_pivot = pickle.load(f)

# Function for book recommendation
def recommended_book(book_name):
    if book_name in rating_pivot.index:  # Checking if the book title exists in the matrix index
        book_id = np.where(rating_pivot.index == book_name)[0][0]
        distances, suggestions = model.kneighbors(rating_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
        recommendations = []
        for i in range(len(suggestions)):
            recommendations.extend(rating_pivot.index[suggestions[i]])
        recommendations.remove(book_name)
        return recommendations
    else:
        similar_books = difflib.get_close_matches(book_name, rating_pivot.index, n=5, cutoff=0.5)
        if similar_books != []:
            st.write(f'Book "{book_name}" not found in the dataset. Did you mean one of these books instead?')
            return similar_books
        else:
            st.write(f'Book "{book_name}" not found in the dataset. Please try another book.')

# Streamlit UI
st.title('Book Recommendation System')

# Text input for entering book title
book_input = st.text_input('Enter a book title:', 'The Great Gatsby')

# Button to trigger recommendations
if st.button('Get Recommendations'):
    recommended_books = recommended_book(book_input)
    if isinstance(recommended_books, list):
        for book in recommended_books:
            st.write(book)
