import pandas as pd
import requests
import streamlit as st
import json
import pickle

movies_list = pickle.load(open('/Users/fenilkheni/Downloads/Movie_Recommender_System-master/Models/movies_dict.pkl', 'rb'))
# movies_list = pd.DataFrame(movies_list)

similarity = pickle.load(open('/Users/fenilkheni/Downloads/Movie_Recommender_System-master/Models/similarity.pkl', 'rb'))
credit = pickle.load(open('/Users/fenilkheni/Downloads/Movie_Recommender_System-master/Models/credits_dict.pkl', 'rb'))
credit_list = pd.DataFrame(credit)
column_names = ['movie_id', 'title', 'tags']
movies_list_df = pd.DataFrame(movies_list, columns=column_names)
st.title('Movie Recommender System')



# Load the pickle files
movies_list = pickle.load(open('/Users/fenilkheni/Downloads/Movie_Recommender_System-master/Models/movies_dict.pkl', 'rb'))

# Check the loaded data type and structure
# st.write(type(movies_list))
st.write(movies_list_df.head())  # This will print the entire data loaded from the pickle file


option = st.selectbox("Which movie would you like to watch!", movies_list_df['title'].values)


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()

    # Check if 'poster_path' is present in the API response
    if 'poster_path' in data and data['poster_path'] is not None:
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    else:
        return None


def recommend(movie):
    index = movies_list[movies_list['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies_list.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies_list.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters


def fetch_info(selected_movie_id):
    selected_movie_crew = credit_list[credit_list['movie_id'] == selected_movie_id]['cast'].values

    cast = selected_movie_crew[0]

    cast_info = json.loads(cast)  # Convert the string to a Python object
    crew_df = pd.DataFrame(cast_info)
    top_crew = crew_df.head(3)  # Get the top 3 crew members

    return top_crew




# Fetch poster of the selected movie
selected_movie_id = movies_list[movies_list['title'] == option]['movie_id'].values[0]
selected_movie_poster = fetch_poster(selected_movie_id)

col1, col2 = st.columns([0.7,1])
# Display the poster above the button
with col1:
    st.image(selected_movie_poster, caption=option, width=200)

# Display top 3 crew details in the second column
with col2:
    st.title("Top Crew:")
    selected_movie_info = fetch_info(selected_movie_id)
    for index, row in selected_movie_info.iterrows():
        st.text(f"Name: {row['character']}")

if st.button('Recommend🤝'):
    recommended_movie_names, recommended_movie_posters = recommend(option)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

print(movies_list)