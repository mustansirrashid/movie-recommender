import pickle
import streamlit as st 
import requests 
import matplotlib.pyplot as plt

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

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
movies_df = pd.read_csv("mergedpandas.csv")
class MovieDataVisualizer:
    def __init__(self, movies_df):
        self.movies_df = movies_df

    def plot_movies_per_year(self):
        


        movie_counts = movies_df['year'].value_counts().sort_index()

# Create the Plotly figure
        fig = go.Figure(data=go.Bar(x=movie_counts.index, y=movie_counts.values))
        fig.update_layout(
            plot_bgcolor='rgb(17, 17, 17)',  
            paper_bgcolor='rgb(17, 17, 17)',  
            font_color='white', 
            title='Number of Movies Released Each Year',  
            xaxis=dict(title='Year'),  
            yaxis=dict(title='Number of Movies')
            )
        fig.update_traces(marker_color='red')

# Display the figure with Streamlit
        st.plotly_chart(fig)

    def plot_top_countries(self):
        import ast

# Function to extract country name from dictionary string
        def extract_country(dict_string):
            dict_list = ast.literal_eval(dict_string)
            if dict_list:
                return dict_list[0]['name']
            else:
                return None

# Apply the function to the 'production_countries' column
        movies_df['country'] = movies_df['production_countries'].apply(extract_country)

# Now create your treemap with the 'country' column
        top_countries = movies_df['country'].value_counts().head(10)
        fig = px.treemap(names=top_countries.index, parents=["" for _ in top_countries.index], values=top_countries.values)
        fig.update_layout(
            plot_bgcolor='rgb(17, 17, 17)',  
            paper_bgcolor='rgb(17, 17, 17)', 
            font_color='white',  
            title='Top Countries with Highest Number of Movies',
            )
        st.plotly_chart(fig)


    def plot_top_languages(self):
        top_language = movies_df['original_language'].value_counts().head(10)


        fig = px.treemap(names=top_language.index, parents=["" for _ in top_language.index], values=top_language.values)
        fig.update_layout(
        plot_bgcolor='rgb(17, 17, 17)',  
        paper_bgcolor='rgb(17, 17, 17)', 
        font_color='white',  
        title='Most Preferred Languages',
        )

        st.plotly_chart(fig)

    def plotPopularMovies(self):
        
        C = movies_df['vote_average'].mean()
        m = movies_df['vote_count'].quantile(0.9)

        q_movies = movies_df.copy().loc[movies_df['vote_count'] >= m]


        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            return (v/(v+m) * R) + (m/(m+v) * C)

        q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
#Sort movies based on score calculated above
        q_movies = q_movies.sort_values('score', ascending=False)

# Print the top 15 movies
        q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)
        pop= movies_df.sort_values('popularity', ascending=False)

        plt.figure(figsize=(25,10))

        plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
        plt.gca().invert_yaxis()
        plt.xlabel("Popularity")
        plt.title("Popular Movies (IMDB metrics system )")
        st.pyplot(plt)

    def HighestRevenue(self):
        revenue= movies_df.sort_values('revenue', ascending=False)

        plt.figure(figsize=(25,10))

        plt.barh(revenue['title'].head(10),revenue['revenue'].head(10), align='center',
        color='green')
        plt.xlabel('Revenue(USD in Billion)')
        plt.title('Movies with the Highest Revenue')
        plt.gca().invert_yaxis()  # Invert y-axis to display the movie with highest revenue at the top
        st.pyplot(plt)

    def HighestBudget(self):
        budget= movies_df.sort_values('budget', ascending=False)

        plt.figure(figsize=(25,10))
        plt.barh(budget['title'].head(8),budget['budget'].head(8), align='center',
        color='violet')
        plt.xlabel('Budget(USD in Billion)')

        plt.title('Movies with the Highest Budget')
        plt.gca().invert_yaxis()  # Invert y-axis to display the movie with highest revenue at the top
        st.pyplot(plt)



visualizer = MovieDataVisualizer(movies_df)

graph_options = ['Number of Movies Released Each Year', 'Top Countries with Highest Number of Movies', 'Most Preferred Languages','Popular Movies of All Time','High Revenue Generated Movies','High Budget Movies']  # Add more options as needed
selected_graph = st.selectbox('Select a graph to display', graph_options)

if st.button('Show Graph'):
    if selected_graph == 'Number of Movies Released Each Year':
        visualizer.plot_movies_per_year()
    elif selected_graph == 'Top Countries with Highest Number of Movies':
        visualizer.plot_top_countries()
    elif selected_graph == 'Most Preferred Languages':
        visualizer.plot_top_languages()
    elif selected_graph =='Popular Movies of All Time':
        visualizer.plotPopularMovies()
    elif selected_graph == 'High Revenue Generated Movies':
        visualizer.HighestRevenue()
    elif selected_graph == 'High Budget Movies':
        visualizer.HighestBudget()


