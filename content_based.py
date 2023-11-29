import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

movies = pd.read_csv("./data/clean_movies.csv", converters={"genres": literal_eval})
ratings_train = pd.read_csv("./data/train.csv", converters={"genres": literal_eval, "tag": literal_eval})
unique_genre = movies['genres'].explode().unique()
genre_distribution = ratings_train['genres'].explode().value_counts()
genre_dict = {k: v for v, k in enumerate(unique_genre)}

plt.pie(genre_distribution.values, labels = genre_distribution.keys())
plt.show()

movies['movie_vector'] = movies['genres'].apply(lambda x: [genre_dict[g] for g in x])
movies['movie_vector'] = movies['movie_vector'].apply(lambda x: np.bincount(x, minlength=len(genre_dict)))

movies.to_csv("./data/movie_vector.csv")
user_ids = ratings_train['userId'].unique()
user_df = pd.DataFrame(columns=['userId', 'user_vector', 'avg_rating', 'num_movies_rated'])
                       
for user_id in user_ids:

    user_rating_df = ratings_train[(ratings_train['userId'] == user_id)]

    user_vector = np.zeros(len(genre_dict))
    count_vector = np.zeros(len(genre_dict))
    
    user_avg_rating = 0
    movies_rated_count = 0
    
    for _, row in user_rating_df.iterrows():
        user_avg_rating += row.rating 
        movies_rated_count += 1
        genres = row.genres

        user_movie_vector = np.zeros(len(genre_dict))
        
        for g in genres:
            user_movie_vector[genre_dict[g]] = 1
            count_vector[genre_dict[g]] += 1
            
        user_vector += user_movie_vector*row.rating

    count_vector = np.where(count_vector==0, 1, count_vector)
    user_vector = np.divide(user_vector, count_vector)
    user_avg_rating /= movies_rated_count
    row_df = pd.DataFrame([[user_id, user_vector, user_avg_rating, movies_rated_count]], 
                          columns=['userId', 'user_vector', 'avg_rating', 'num_movies_rated'])
    user_df = pd.concat([user_df, row_df], ignore_index=True)

user_df.to_csv("./data/user_info.csv")

ratings_test = pd.read_csv("./data/test.csv", converters={"genres": literal_eval, "tag": literal_eval}) 

algo_predictions = pd.DataFrame(columns=['userId', 'movieId', 'user_vector', 'movie_vector', 'og_rating', 'pred_rating'])
for ind, row in ratings_test.iterrows():
    userId = row['userId']
    movieId = row['movieId']
    og_rating = row['rating']
    
    try:
        user_vector = user_df[user_df['userId'] == int(userId)].user_vector.values[0]
        movie_vector = movies[movies['movieId'] == int(movieId)].movie_vector.values[0]
        predicted_rating = user_vector*movie_vector
  
        if predicted_rating.any():
            predicted_rating = np.nanmean(np.where(predicted_rating!=0, predicted_rating, np.nan)) 
        else:
            predicted_rating = 0

        row_df = pd.DataFrame([[userId, movieId, user_vector, movie_vector, og_rating, predicted_rating]], 
                    columns=['userId', 'movieId', 'user_vector', 'movie_vector', 'og_rating', 'pred_rating'])
        algo_predictions = pd.concat([algo_predictions, row_df], ignore_index=True)
    except:
        print("User not found: ", userId)

algo_predictions.to_csv("./data/content_based_genre_ratings.csv") 
rmse = ((algo_predictions.og_rating - algo_predictions.pred_rating) ** 2).mean() ** .5
mae = (abs(algo_predictions.og_rating - algo_predictions.pred_rating)).mean()
print(f"rmse: {rmse}")
print(f"mae: {mae}")