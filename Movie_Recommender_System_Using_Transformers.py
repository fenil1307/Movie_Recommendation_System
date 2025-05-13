# # import pandas as pd
# # import os
# # from sentence_transformers import SentenceTransformer
# # from langchain.vectorstores import FAISS
# # from langchain.embeddings import HuggingFaceEmbeddings
# # import numpy as np
# # import streamlit as st
# # import time
# #
# # # Step 1: Extract and Prepare the Dataset
# # url = "https://raw.githubusercontent.com/datum-oracle/netflix-movie-titles/main/titles.csv"
# # movies_df = pd.read_csv(url)
# #
# # # Step 2: Transform the Data - Combine relevant columns
# # movies_df['combined_text'] = movies_df[['title', 'description', 'genres']].fillna('').agg(' '.join, axis=1)
# #
# # # Set your Hugging Face token
# # os.environ['HUGGINGFACE_TOKEN'] = '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'  # Replace with your actual token
# # hf_token = os.getenv('HUGGINGFACE_TOKEN')
# #
# # # Step 3: Initialize the embedding model
# # embed_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'token': hf_token})
# #
# # # Generate embeddings and create FAISS index
# # texts = movies_df['combined_text'].tolist()
# # faiss_index = FAISS.from_texts(texts, embed_model)
# #
# # # Step 4: Query Interface - Streamlit Application
# # st.title("Movie Recommendation Engine")
# # st.write("Initializing...")
# #
# # start_time = time.time()
# # time.sleep(2)  # Simulate delay
# # st.write(f"Initialization took {time.time() - start_time} seconds")
# #
# # # Get user input
# # user_input = st.text_input("Describe the kind of movie you're looking for (e.g., 'heartfelt romantic comedy'):")
# # if user_input:
# #     st.write(f"Searching for movies related to: {user_input}")
# #
# #     # Add additional filters for IMDb score and votes
# #     min_imdb_score = st.slider('Minimum IMDb Score', 0.0, 10.0, 5.0)
# #     min_votes = st.slider('Minimum Number of Votes', 0, 1000000, 10000)
# #
# #     # Convert IMDb score and votes to numeric and drop NaN values
# #     movies_df['imdb_score'] = pd.to_numeric(movies_df['imdb_score'], errors='coerce')
# #     movies_df['imdb_votes'] = pd.to_numeric(movies_df['imdb_votes'], errors='coerce')
# #     movies_df = movies_df.dropna(subset=['imdb_score', 'imdb_votes'])
# #
# #     # Apply filters
# #     filtered_df = movies_df[(movies_df['imdb_score'] >= min_imdb_score) & (movies_df['imdb_votes'] >= min_votes)]
# #
# #     if filtered_df.empty:
# #         st.write("No movies match the filter criteria.")
# #     else:
# #         # Create a new FAISS index for filtered movies
# #         filtered_texts = filtered_df['combined_text'].tolist()
# #         filtered_faiss_index = FAISS.from_texts(filtered_texts, embed_model)
# #
# #         # Search for the most similar movie
# #         similar_docs = filtered_faiss_index.similarity_search(user_input, k=1)
# #
# #         if similar_docs:
# #             top_match_index = filtered_texts.index(similar_docs[0].page_content)
# #
# #             # Show the top-matching movie
# #             st.write("You might like this movie:", filtered_df.iloc[top_match_index]['title'])
# #             st.write("Description:", filtered_df.iloc[top_match_index]['description'])
# #             st.write("Genres:", filtered_df.iloc[top_match_index]['genres'])
# #             st.write("IMDb Score:", filtered_df.iloc[top_match_index]['imdb_score'])
# #             st.write("IMDb Votes:", filtered_df.iloc[top_match_index]['imdb_votes'])
# #         else:
# #             st.write("No similar movies found.")
#
#
# import pandas as pd
# import os
# from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# import numpy as np
# import streamlit as st
# import time
#
# # Set page config
# st.set_page_config(page_title="Movie Recommendation Engine", page_icon="", layout="wide")
#
# # Custom CSS
# st.markdown("""
# <style>
#     .reportview-container {
#         background: #f0f2f6
#     }
#     .sidebar .sidebar-content {
#         background: #ffffff
#     }
#     .Widget>label {
#         color: #31333F;
#         font-weight: bold;
#     }
#     .stButton>button {
#         color: #ffffff;
#         background-color: #FF4B4B;
#         border-radius: 5px;
#     }
# </style>
# """, unsafe_allow_html=True)
#
# # Sidebar
# st.sidebar.title("🎥 Movie Filters")
#
#
# @st.cache_data
# def load_data():
#     url = "https://raw.githubusercontent.com/datum-oracle/netflix-movie-titles/main/titles.csv"
#     return pd.read_csv(url)
#
#
# # Load data
# movies_df = load_data()
#
# # Transform the Data
# movies_df['combined_text'] = movies_df[['title', 'description', 'genres']].fillna('').agg(' '.join, axis=1)
#
# # Set Hugging Face token
# os.environ['HUGGINGFACE_TOKEN'] = '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'  # Replace with your actual token
# hf_token = os.getenv('HUGGINGFACE_TOKEN')
#
#
# # Initialize the embedding model
# @st.cache_resource
# def init_embed_model():
#     return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'token': hf_token})
#
#
# embed_model = init_embed_model()
#
#
# # Generate embeddings and create FAISS index
# @st.cache_resource
# def create_faiss_index(texts):
#     return FAISS.from_texts(texts, embed_model)
#
#
# # Main content
# st.title(" Movie Recommendation Engine")
#
# # Initialize progress
# progress_bar = st.progress(0)
# status_text = st.empty()
#
# # Simulate initialization process
# for i in range(100):
#     time.sleep(0.01)
#     progress_bar.progress(i + 1)
#     status_text.text(f"Initializing... {i + 1}%")
#
# status_text.text("Initialization complete!")
# time.sleep(1)
# progress_bar.empty()
# status_text.empty()
#
# # User input
# user_input = st.text_input(" Describe the kind of movie you're looking for:",
#                            placeholder="e.g., 'heartfelt romantic comedy'")
#
# # Sidebar filters
# min_imdb_score = st.sidebar.slider('Minimum IMDb Score', 0.0, 10.0, 5.0, 0.1)
# min_votes = st.sidebar.slider('Minimum Number of Votes', 0, 1000000, 10000, 1000)
#
# if user_input:
#     st.write(f" Searching for movies related to: **{user_input}**")
#
#     # Apply filters
#     filtered_df = movies_df[
#         (movies_df['imdb_score'].astype(float) >= min_imdb_score) &
#         (movies_df['imdb_votes'].astype(float) >= min_votes)
#         ]
#
#     if filtered_df.empty:
#         st.warning("⚠ No movies match the filter criteria.")
#     else:
#         # Create a new FAISS index for filtered movies
#         filtered_texts = filtered_df['combined_text'].tolist()
#         filtered_faiss_index = create_faiss_index(filtered_texts)
#
#         # Search for the most similar movie
#         similar_docs = filtered_faiss_index.similarity_search(user_input, k=1)
#
#         if similar_docs:
#             top_match_index = filtered_texts.index(similar_docs[0].page_content)
#             movie = filtered_df.iloc[top_match_index]
#
#             # Show the top-matching movie
#             st.success(" We found a movie you might like!")
#
#             col1, col2 = st.columns([1, 2])
#
#             with col1:
#                 st.image("https://via.placeholder.com/200x300?text=Movie+Poster", caption=movie['title'])
#
#             with col2:
#                 st.subheader(movie['title'])
#                 st.write(f"**Description:** {movie['description']}")
#                 st.write(f"**Genres:** {movie['genres']}")
#                 st.write(f"**IMDb Score:** {movie['imdb_score']:.1f} ")
#                 st.write(f"**IMDb Votes:** {movie['imdb_votes']:,}")
#
#             # Add a button to get more recommendations
#             if st.button(" Get Another Recommendation"):
#                 st.experimental_rerun()
#         else:
#             st.warning(" No similar movies found.")
#
# # Footer
# st.sidebar.markdown("---")
# st.sidebar.info(
#     "This app uses FAISS for efficient similarity search and Hugging Face's sentence transformers for text embedding.")
# st.sidebar.text("© 2024 Movie Recommender")










# =========================================================================================================================================

import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
import streamlit as st
import time


st.set_page_config(page_title="Movie Recommendation Engine", layout="wide")


st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #FF4B4B;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


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
