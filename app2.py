# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from rapidfuzz import process
import requests
from pymongo import MongoClient

st.set_page_config(page_title="Mood-Time Based Recommender", layout="wide")



# MongoDB connection
MONGO_URI = st.secrets["mongo"]["uri"]
DB_NAME = st.secrets["mongo"]["db"]
COLLECTION_NAME = st.secrets["mongo"]["collection"]

client = MongoClient(MONGO_URI)
db = client[MovieRecoDB]
collection = db[interaction_logs]

MOODS = ["Happy", "Sad", "Excited", "Bored", "Angry", "Calm"]
GENRES = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi", "Horror", "Animation"]

MOVIES = {
    "Action": [
        {"title": "Mad Max", "rating": 8.1, "url": "https://www.imdb.com/title/tt1392190/"},
        {"title": "John Wick", "rating": 7.4, "url": "https://www.imdb.com/title/tt2911666/"}
    ],
    "Comedy": [
        {"title": "The Hangover", "rating": 7.7, "url": "https://www.imdb.com/title/tt1119646/"},
        {"title": "Superbad", "rating": 7.6, "url": "https://www.imdb.com/title/tt0829482/"}
    ],
    "Drama": [
        {"title": "The Shawshank Redemption", "rating": 9.3, "url": "https://www.imdb.com/title/tt0111161/"},
        {"title": "The Godfather", "rating": 9.2, "url": "https://www.imdb.com/title/tt0068646/"}
    ],
    "Thriller": [
        {"title": "Gone Girl", "rating": 8.1, "url": "https://www.imdb.com/title/tt2267998/"},
        {"title": "Se7en", "rating": 8.6, "url": "https://www.imdb.com/title/tt0114369/"}
    ],
    "Romance": [
        {"title": "The Notebook", "rating": 7.8, "url": "https://www.imdb.com/title/tt0332280/"},
        {"title": "La La Land", "rating": 8.0, "url": "https://www.imdb.com/title/tt3783958/"}
    ],
    "Sci-Fi": [
        {"title": "Inception", "rating": 8.8, "url": "https://www.imdb.com/title/tt1375666/"},
        {"title": "Interstellar", "rating": 8.6, "url": "https://www.imdb.com/title/tt0816692/"}
    ],
    "Horror": [
        {"title": "The Conjuring", "rating": 7.5, "url": "https://www.imdb.com/title/tt1457767/"},
        {"title": "Hereditary", "rating": 7.3, "url": "https://www.imdb.com/title/tt7784604/"}
    ],
    "Animation": [
        {"title": "Coco", "rating": 8.4, "url": "https://www.imdb.com/title/tt2380307/"},
        {"title": "Up", "rating": 8.3, "url": "https://www.imdb.com/title/tt1049413/"}
    ]
}

# Sidebar login/logout UI
if "username" not in st.session_state:
    st.session_state.username = None

with st.sidebar:
    if st.session_state.username:
        st.markdown(f"### Welcome {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.username = None
            st.rerun()
    else:
        username_input = st.text_input("Enter Username")
        if st.button("Login") and username_input:
            st.session_state.username = username_input
            st.rerun()

st.title("🎬 Mood-Time Based Movie Recommender")

if not st.session_state.username:
    st.warning("Please login from the sidebar to continue.")
    st.stop()

mood = st.selectbox("How are you feeling now?", MOODS)
current_hour = datetime.now().hour
sim_hour = st.slider("Select Time (simulate time-of-day)", 0, 23, current_hour)

# Placeholder for JustWatch API
def fetch_streaming_link(movie_title):
    return f"https://www.justwatch.com/in/search?q={movie_title.replace(' ', '%20')}"

# Logging to MongoDB
def log_interaction(username, mood, genre, movie, hour):
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "mood": mood,
        "genre": genre,
        "movie": movie,
        "hour": hour
    }
    collection.insert_one(interaction)

# Model training
def train_model():
    records = list(collection.find())
    if len(records) < 10:
        return None, None, None
    df = pd.DataFrame(records)

    le_mood = LabelEncoder()
    le_genre = LabelEncoder()
    df['mood_encoded'] = le_mood.fit_transform(df['mood'])
    df['genre_encoded'] = le_genre.fit_transform(df['genre'])

    X = df[['hour', 'mood_encoded']]
    y = df['genre_encoded']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le_genre, le_mood

model, le_genre, le_mood = train_model()

def recommend_movie_ml(mood, hour):
    if model is None:
        return recommend_movie_rule(mood, hour)
    try:
        mood_encoded = le_mood.transform([mood])[0]
        pred_encoded = model.predict([[hour, mood_encoded]])[0]
        genre = le_genre.inverse_transform([pred_encoded])[0]
        movie_info = np.random.choice(MOVIES[genre])
        return movie_info, genre
    except:
        return recommend_movie_rule(mood, hour)

def recommend_movie_rule(mood, hour):
    genre_map = {
        "Happy": "Comedy",
        "Sad": "Drama",
        "Excited": "Action",
        "Bored": "Thriller",
        "Angry": "Sci-Fi",
        "Calm": "Animation"
    }
    genre = genre_map.get(mood, np.random.choice(GENRES))
    movie_info = np.random.choice(MOVIES[genre])
    return movie_info, genre

if st.button("🎥 Recommend me a movie"):
    movie_info, genre = recommend_movie_ml(mood, sim_hour)
    st.success(f"**Recommended Movie**: {movie_info['title']}")
    st.write(f"**Genre**: {genre} | **Mood**: {mood} | **Hour**: {sim_hour}")
    st.markdown(f"[🔗 Watch Now]({fetch_streaming_link(movie_info['title'])})")

    if st.session_state.username:
        log_interaction(st.session_state.username, mood, genre, movie_info['title'], sim_hour)
