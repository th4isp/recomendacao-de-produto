import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Caminho para os arquivos CSV
movies_path = 'ml-latest-small/movies.csv'
ratings_path = 'ml-latest-small/ratings.csv'

# Carregar os dados
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Mostrar as primeiras linhas dos dados
print("Movies Data:")
print(movies.head())

print("\nRatings Data:")
print(ratings.head())

# Exploração Adicional
print("\nMovies Data Info:")
print(movies.info())

print("\nRatings Data Info:")
print(ratings.info())

print("\nRatings Data Statistics:")
print(ratings.describe())

print("\nNumber of Ratings per Movie:")
print(ratings['movieId'].value_counts().head())

print("\nNumber of Ratings per User:")
print(ratings['userId'].value_counts().head())

# Preparar os Dados
# Criar a matriz de usuário-item
user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating')
user_movie_ratings = user_movie_ratings.fillna(0)

# Criar um dicionário de IDs de filmes para títulos
movie_titles = movies.set_index('movieId')['title'].to_dict()

# Calcular a similaridade entre usuários
user_similarity = cosine_similarity(user_movie_ratings)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)

print("\nUser Similarity Matrix:")
print(user_similarity_df.head())

# Função para recomendar filmes
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

# Testar a função com um ID de usuário
user_id = 1  # Substitua pelo ID de usuário que deseja testar
print("\nRecommendations for User ID", user_id)
print(recommend_movies(user_id))
