import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Mood-Time Movie Recommender", layout="centered")

# Constants
DATA_FILE = "interaction_logs.csv"
MOODS = ["Happy", "Sad", "Excited", "Bored", "Angry", "Calm"]
GENRES = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi", "Horror", "Animation"]
TMDB_API_KEY = "f2ea015394e9dc19b850c7dbd745eca9"

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

def fetch_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        res = requests.get(url).json()
        if res['results']:
            poster_path = res['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w300{poster_path}"
    except:
        pass
    return None

def fetch_streaming_link(title):
    query = title.replace(" ", "+")
    return f"https://www.justwatch.com/search?q={query}"

def train_model():
    if not os.path.exists(DATA_FILE):
        return None, None, None

    df = pd.read_csv(DATA_FILE)
    if len(df) < 10:
        return None, None, None

    le_mood = LabelEncoder()
    le_genre = LabelEncoder()

    df['mood_encoded'] = le_mood.fit_transform(df['mood'])
    df['genre_encoded'] = le_genre.fit_transform(df['genre'])

    X = df[['hour', 'mood_encoded']]
    y = df['genre_encoded']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le_genre, le_mood

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
    if mood == "Happy": genre = "Comedy"
    elif mood == "Sad": genre = "Drama"
    elif mood == "Excited": genre = "Action"
    elif mood == "Bored": genre = "Thriller"
    elif mood == "Angry": genre = "Sci-Fi"
    elif mood == "Calm": genre = "Animation"
    else: genre = np.random.choice(GENRES)
    movie_info = np.random.choice(MOVIES[genre])
    return movie_info, genre

def login():
    if "username" not in st.session_state:
        username = st.text_input("👤 Enter your username to personalize recommendations", "")
        if username:
            st.session_state["username"] = username
            st.experimental_rerun()  # rerun to load session state with username

if "username" not in st.session_state:
    login()
else:
    st.sidebar.markdown(f"**Welcome, {st.session_state['username']}!** 🎉")

    # Auto-play chill audio once user is logged in
    st.audio('https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3', format='audio/mp3', start_time=0)

    st.title("🎬 Mood-Time Based Movie Recommender")

    mood = st.selectbox("How are you feeling now?", MOODS)
    current_hour = datetime.now().hour
    sim_hour = st.slider("Select Time (simulate time-of-day)", 0, 23, current_hour)

    model, le_genre, le_mood = train_model()

    if st.button("🎥 Recommend me a movie"):
        movie_info, genre = recommend_movie_ml(mood, sim_hour)
        poster_url = fetch_poster(movie_info['title'])
        streaming_url = fetch_streaming_link(movie_info['title'])

        st.success(f"**Recommended Movie**: {movie_info['title']}")
        if poster_url:
            st.image(poster_url, width=200, caption=movie_info['title'])
        st.write(f"**Genre**: {genre} | **Mood**: {mood} | **Hour**: {sim_hour}")
        st.write(f"⭐ **Rating**: {movie_info['rating']}")
        st.markdown(f"🔗 [More Info / Trailer]({movie_info['url']})")
        st.markdown(f"📺 [Where to Watch]({streaming_url})")

        # Save interaction log
        new_data = pd.DataFrame([{
            'timestamp': datetime.now(),
            'username': st.session_state['username'],
            'hour': sim_hour,
            'mood': mood,
            'genre': genre,
            'movie': movie_info['title']
        }])
        if os.path.exists(DATA_FILE):
            existing = pd.read_csv(DATA_FILE)
            updated = pd.concat([existing, new_data], ignore_index=True)
        else:
            updated = new_data
        updated.to_csv(DATA_FILE, index=False)

    st.subheader("📊 Mood vs Time vs Genre Heatmap")

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        user_df = df[df['username'] == st.session_state["username"]]
        if len(user_df) > 0:
            pivot = user_df.groupby(['hour', 'genre']).size().reset_index(name='count')
            heatmap_data = pivot.pivot_table(index='hour', columns='genre', values='count', aggfunc='sum').fillna(0)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=.5)
            plt.title(f"Your Genre Popularity by Time (User: {st.session_state['username']})")
            st.pyplot(fig)
        else:
            st.info("No user data yet to show heatmap. Start getting recommendations!")
    else:
        st.info("Start generating recommendations to see heatmap insights.")
