# app2.py

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
import requests

st.set_page_config(page_title="Mood-Time Recommender", layout="wide")

# ------------------ Constants -------------------
DATA_FILE = "interaction_logs.csv"
POSTER_API = "https://api.themoviedb.org/3/search/movie"
POSTER_IMG = "https://image.tmdb.org/t/p/w500"
TMDB_API_KEY = "f2ea015394e9dc19b850c7dbd745eca9"
MOODS = ["Happy", "Sad", "Excited", "Bored", "Angry", "Calm"]
GENRES = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi", "Horror", "Animation"]

MOVIES = {
    "Action": [{"title": "Mad Max"}, {"title": "John Wick"}],
    "Comedy": [{"title": "The Hangover"}, {"title": "Superbad"}],
    "Drama": [{"title": "The Shawshank Redemption"}, {"title": "The Godfather"}],
    "Thriller": [{"title": "Gone Girl"}, {"title": "Se7en"}],
    "Romance": [{"title": "The Notebook"}, {"title": "La La Land"}],
    "Sci-Fi": [{"title": "Inception"}, {"title": "Interstellar"}],
    "Horror": [{"title": "The Conjuring"}, {"title": "Hereditary"}],
    "Animation": [{"title": "Coco"}, {"title": "Up"}]
}

RATINGS = {
    "Mad Max": 8.1, "John Wick": 7.4, "The Hangover": 7.7, "Superbad": 7.6,
    "The Shawshank Redemption": 9.3, "The Godfather": 9.2,
    "Gone Girl": 8.1, "Se7en": 8.6, "The Notebook": 7.8,
    "La La Land": 8.0, "Inception": 8.8, "Interstellar": 8.6,
    "The Conjuring": 7.5, "Hereditary": 7.3, "Coco": 8.4, "Up": 8.3
}


# ------------------ Sidebar -------------------
if "username" not in st.session_state:
    st.session_state.username = None

with st.sidebar:
    st.title("🎬 Movie Recommender")

    if st.session_state.username:
        st.success(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.username = None
            st.rerun()
    else:
        username = st.text_input("Enter username:")
        if st.button("Login"):
            if username:
                st.session_state.username = username
                st.rerun()
            else:
                st.warning("Username cannot be empty.")


# ------------------ Auth Gate -------------------
if not st.session_state.username:
    st.stop()


# ------------------ Helper Functions -------------------
def fetch_streaming_link(title):
    return f"https://www.justwatch.com/in/search?q={title.replace(' ', '%20')}"

def fetch_poster(title):
    params = {"api_key": TMDB_API_KEY, "query": title}
    response = requests.get(POSTER_API, params=params)
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results and results[0].get("poster_path"):
            return POSTER_IMG + results[0]["poster_path"]
    return None

def train_model():
    if not os.path.exists(DATA_FILE):
        return None, None, None
    df = pd.read_csv(DATA_FILE)
    if len(df) < 10:
        return None, None, None

    le_mood, le_genre = LabelEncoder(), LabelEncoder()
    df['mood_encoded'] = le_mood.fit_transform(df['mood'])
    df['genre_encoded'] = le_genre.fit_transform(df['genre'])
    X, y = df[['hour', 'mood_encoded']], df['genre_encoded']
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    return model, le_genre, le_mood

model, le_genre, le_mood = train_model()

def recommend_movie(mood, hour):
    if model:
        try:
            mood_code = le_mood.transform([mood])[0]
            genre_code = model.predict([[hour, mood_code]])[0]
            genre = le_genre.inverse_transform([genre_code])[0]
        except:
            genre = rule_based_genre(mood)
    else:
        genre = rule_based_genre(mood)

    movie = np.random.choice(MOVIES[genre])['title']
    rating = RATINGS.get(movie, "N/A")
    poster = fetch_poster(movie)
    stream = fetch_streaming_link(movie)
    return movie, genre, rating, poster, stream

def rule_based_genre(mood):
    return {
        "Happy": "Comedy", "Sad": "Drama", "Excited": "Action",
        "Bored": "Thriller", "Angry": "Sci-Fi", "Calm": "Animation"
    }.get(mood, np.random.choice(GENRES))


# ------------------ Main Section -------------------
st.markdown("## 🧠 Mood-Time Based Recommendation")

mood = st.selectbox("How are you feeling now?", MOODS)
hour = st.slider("🕒 Choose a time to simulate your mood", 0, 23, datetime.now().hour)

if st.button("🎥 Recommend Movie"):
    movie, genre, rating, poster, stream = recommend_movie(mood, hour)

    st.markdown(f"### 🎬 {movie}")
    st.write(f"**Genre**: {genre} | **Mood**: {mood} | **Time**: {hour}:00 hrs")
    st.write(f"⭐ **Rating**: {rating}")
    st.markdown(f"[🔗 Where to Watch]({stream})")

    if poster:
        st.image(poster, width=300)
    else:
        st.info("No poster found.")

    # Logging user data
    log = pd.DataFrame([{
        "username": st.session_state.username,
        "timestamp": datetime.now(),
        "hour": hour,
        "mood": mood,
        "genre": genre,
        "movie": movie
    }])

    if os.path.exists(DATA_FILE):
        logs = pd.read_csv(DATA_FILE)
        logs = pd.concat([logs, log], ignore_index=True)
    else:
        logs = log

    logs.to_csv(DATA_FILE, index=False)


# ------------------ Heatmap + Insights -------------------
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    df = df[df['username'] == st.session_state.username]

    if not df.empty:
        st.subheader("📊 Your Mood-Time Genre Heatmap")

        pivot = df.groupby(['hour', 'genre']).size().reset_index(name='count')
        heatmap_data = pivot.pivot_table(index='hour', columns='genre', values='count', aggfunc='sum').fillna(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".0f")
        st.pyplot(fig)

else:
    st.info("Use the app to build up your recommendation history.")
