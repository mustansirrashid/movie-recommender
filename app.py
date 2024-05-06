import pickle
import streamlit as st 
import requests 

class MovieRecommender:
    def __init__(self, movies_path, similarity_path):
        self.movies = pickle.load(open(movies_path, 'rb'))
        self.similarity = pickle.load(open(similarity_path, 'rb'))

    def fetch_poster(self, movie_id):
        url = "https://api.themoviedb.org/3/movie/{}?api_key=7e80d2686fbd0916ffad5e27b5d682a7&language=en-US".format(movie_id)
        data = requests.get(url).json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path

    def recommend(self, movie):
        index = self.movies[self.movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(self.similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movies_name = []
        recommended_movies_poster = []
        for i in distances[1:6]:
            movie_id = self.movies.iloc[i[0]].movie_id
            recommended_movies_poster.append(self.fetch_poster(movie_id))
            recommended_movies_name.append(self.movies.iloc[i[0]].title)
        return recommended_movies_name, recommended_movies_poster


recommender = MovieRecommender('artifacts/movie_list.pkl', 'artifacts/similarity.pkl')

st.header("METEORA MOVIES RECOMMENDER")

# Create a row with a selectbox and a help button
col1, col2 = st.columns([4, 1])
with col1:
    movie_list = recommender.movies['title'].values
    selected_movie = st.selectbox(
        'Type or Select a movie to get you recommendation',
        movie_list
    )
with col2:
    if st.button('❓'):
        st.popover("""
        **How to use this app:**
        1. Use the dropdown menu to select a movie from the list.
        2. Click the 'Show Recommendation' button.
        3. The app will display 5 movies that are similar to the selected movie.
        4. Use the 2nd dropdown menu to see important graphs
        """)

if st.button('Show Recommendation'):
    recommended_movies_name, recommended_movies_poster = recommender.recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movies_name[0])
        st.image(recommended_movies_poster[0])
    with col2:
        st.text(recommended_movies_name[1])
        st.image(recommended_movies_poster[1])
    with col3:
        st.text(recommended_movies_name[2])
        st.image(recommended_movies_poster[2])
    with col4:
        st.text(recommended_movies_name[3])
        st.image(recommended_movies_poster[3])
    with col5:
        st.text(recommended_movies_name[4])
        st.image(recommended_movies_poster[4])

graphs = ['Graph for Highest Budget', 'Highest Movie Numbers Around The World','Highest Revenue Generated Movies','Most Preffered Langugages In A Movie','Number Of Movies Realeased Each Year','Most Popular Movies ']

# Create a dictionary mapping graph names to file paths
graph_files = {
    'Graph for Highest Budget': 'artifacts/HighestBudget.jpg',
    'Highest Movie Numbers Around The World': 'artifacts/HighestMovieNumber.jpg',
    'Highest Revenue Generated Movies': 'artifacts/HighestRevenue.jpg',
    'Most Preffered Langugages In A Movie': 'artifacts/MostLanguage.jpg',
    'Number Of Movies Realeased Each Year': 'artifacts/MovieNumber.jpg',
    'Most Popular Movies ': 'artifacts/PopularMovies.jpg',
}

# Create a selectbox for the graphs
selected_graph = st.selectbox('Select a graph:', graphs)

# Display the selected graph
if st.button('Show Graph'):
    # Load the image file
    img_path = graph_files[selected_graph]
    img = open(img_path, 'rb').read()

    
    st.image(img, caption=selected_graph)
