import streamlit as st
import pickle
import pandas as pd
import difflib
from pymongo import MongoClient
from datetime import datetime

# MongoDB connection
client = MongoClient(st.secrets["mongo"]["uri"])
db = client[st.secrets["mongo"]["db"]]
collection = db[st.secrets["mongo"]["collection"]]

# Load movie list and similarity matrix
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Set page config
st.set_page_config(layout="wide")

# Function to recommend movies
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# Sidebar layout
with st.sidebar:
    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.username:
        st.markdown(f"## Welcome {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.username = None
            st.rerun()
    else:
        st.markdown("## Login")
        username_input = st.text_input("Enter your username")
        if st.button("Login") and username_input:
            st.session_state.username = username_input
            st.success("Logged in successfully!")
            st.rerun()

# Main layout
if st.session_state.username:
    st.title("Movie Recommendation System")
    selected_movie = st.selectbox("Choose a movie to get recommendations:", movies['title'].values)

    if st.button("Recommend"):
        recommendations = recommend(selected_movie)
        st.subheader("Top 5 Recommendations")

        for movie_title in recommendations:
            st.markdown(f"- {movie_title}")

            # Log to MongoDB
            try:
                collection.insert_one({
                    "username": st.session_state.username,
                    "recommended_movie": movie_title,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                st.error(f"MongoDB Error: {e}")
else:
    st.warning("Please login to continue.")
