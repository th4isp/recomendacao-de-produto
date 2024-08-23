from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Carregar dados e preparar
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
movie_titles = movies.set_index('movieId')['title'].to_dict()
user_similarity = cosine_similarity(user_movie_ratings)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)

def recommend_movies(user_id, num_recommendations=5):
    if user_id not in user_similarity_df.index:
        return "User ID not found"
    
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    recommended_movies = []
    
    for sim_user in similar_users:
        user_movies = user_movie_ratings.loc[sim_user]
        for movie_id in user_movies[user_movies > 0].index:
            if user_movie_ratings.loc[user_id, movie_id] == 0:
                recommended_movies.append(movie_id)
                
                if len(recommended_movies) >= num_recommendations:
                    return [movie_titles[movie_id] for movie_id in recommended_movies]
    
    return [movie_titles[movie_id] for movie_id in recommended_movies] if recommended_movies else "No recommendations available"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = recommend_movies(user_id)
    return render_template('recommendations.html', user_id=user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
