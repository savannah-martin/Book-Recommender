import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import pickle
import difflib

model = load('model.joblib')

# Load the matrix
with open('rating_pivot.pickle', 'rb') as f:
    rating_pivot = pickle.load(f)

books = pd.read_csv('data/Books.csv')

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
        if similar_books:
            st.write(f'Book "{book_name}" not found in the dataset. Did you mean one of these books instead?')
            return similar_books
        else:
            st.write(f'Book "{book_name}" not found in the dataset. Please try another book.')

st.title('Book Recommendation System')

book_input = st.text_input('Enter a book title:', 'The Great Gatsby')

if st.button('Get Recommendations'):
    recommended_books = recommended_book(book_input)
    if isinstance(recommended_books, list):
        for book in recommended_books:
            st.subheader(book)
            # Displaying image for the entered book title
            selected_book = books[books['Book-Title'] == book]
            if not selected_book.empty:
                book_author = selected_book['Book-Author'].values[0]
                yop = selected_book['Year-Of-Publication'].values[0]
                pub = selected_book['Publisher'].values[0]
                
                st.write('Book Details:')
                st.write(f"- **Author:** {book_author}")
                st.write(f"- **Year of Publication:** {yop}")
                st.write(f"- **Publisher:** {pub}")
            else:
                st.write(f"Book '{book}' not found.")

