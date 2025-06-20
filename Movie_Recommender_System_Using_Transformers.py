import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
import streamlit as st
import time


st.set_page_config(page_title="Movie Recommendation Engine", layout="wide")


st.sidebar.title(" Movie Filters")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datum-oracle/netflix-movie-titles/main/titles.csv"
    return pd.read_csv(url)


movies_df = load_data()


movies_df['combined_text'] = movies_df[['title', 'description', 'genres']].fillna('').agg(' '.join, axis=1)


os.environ['HUGGINGFACE_TOKEN'] = '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'  
hf_token = os.getenv('HUGGINGFACE_TOKEN')


@st.cache_resource
def init_embed_model():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'token': hf_token})

embed_model = init_embed_model()


@st.cache_resource
def create_faiss_index(texts):
    return FAISS.from_texts(texts, embed_model)


if 'recommendation_count' not in st.session_state:
    st.session_state.recommendation_count = 0

if 'filtered_faiss_index' not in st.session_state:
    st.session_state.filtered_faiss_index = None


st.title(" Movie Recommendation Engine")


progress_bar = st.progress(0)
status_text = st.empty()


for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i + 1)
    status_text.text(f"Initializing... {i + 1}%")

status_text.text("Initialization complete!")
time.sleep(1)
progress_bar.empty()
status_text.empty()


user_input = st.text_input(" Describe the kind of movie you're looking for:",
                           placeholder="e.g., 'heartfelt romantic comedy'")


min_imdb_score = st.sidebar.slider('Minimum IMDb Score', 0.0, 10.0, 5.0, 0.1)
min_votes = st.sidebar.slider('Minimum Number of Votes', 0, 1000000, 10000, 1000)

if user_input:
    st.write(f" Searching for movies related to: **{user_input}**")


    filtered_df = movies_df[
        (movies_df['imdb_score'].astype(float) >= min_imdb_score) &
        (movies_df['imdb_votes'].astype(float) >= min_votes)
    ]

    if filtered_df.empty:
        st.warning(" No movies match the filter criteria.")
    else:
        
        filtered_texts = filtered_df['combined_text'].tolist()

        if st.session_state.filtered_faiss_index is None:
            st.session_state.filtered_faiss_index = create_faiss_index(filtered_texts)

        
        similar_docs = st.session_state.filtered_faiss_index.similarity_search(user_input, k=1)

        if similar_docs:
            top_match_index = filtered_texts.index(similar_docs[0].page_content)
            movie = filtered_df.iloc[top_match_index]

            
            st.success(" We found a movie you might like!")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image("https://via.placeholder.com/200x300?text=Movie+Poster", caption=movie['title'])

            with col2:
                st.subheader(movie['title'])
                st.write(f"**Description:** {movie['description']}")
                st.write(f"**Genres:** {movie['genres']}")
                st.write(f"**IMDb Score:** {movie['imdb_score']:.1f} ")
                st.write(f"**IMDb Votes:** {movie['imdb_votes']:,}")

            
            if st.button(" Get Another Recommendation"):
                
                st.session_state.recommendation_count += 1
                
                st.session_state.filtered_faiss_index = None
                
                st.experimental_rerun()
        else:
            st.warning(" No similar movies found.")


st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses FAISS for efficient similarity search and Hugging Face's sentence transformers for text embedding.")
st.sidebar.text(" 2024 Movie Recommender")
