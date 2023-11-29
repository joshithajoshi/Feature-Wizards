import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import pdb

genre_user_vector = pd.read_csv("./data/user_info.csv")
genre_user_vector = genre_user_vector[['userId', 'user_vector']]

genre_user_vector['user_vector'] = genre_user_vector['user_vector'].apply(lambda x: x.replace('[', ' ').replace(']', ' ').strip().split())
genre_user_vector['user_vector'] = genre_user_vector['user_vector'].apply(lambda x: np.asarray(x).astype(float))

era_user_vector = pd.read_csv("./data/user_era_vector.csv")
era_user_vector = era_user_vector[['userId', 'user_era_vector']]

era_user_vector['user_era_vector'] = era_user_vector['user_era_vector'].apply(lambda x: x.replace('[', ' ').replace(']', ' ').strip().split())
era_user_vector['user_era_vector'] = era_user_vector['user_era_vector'].apply(lambda x: np.asarray(x).astype(float))

merged_user = genre_user_vector.join(era_user_vector['user_era_vector'])

merged_user['final_user_vector'] = merged_user.apply(lambda x: np.concatenate((2*x['user_vector'], x['user_era_vector'])), axis=1)

movie_genre_vector = pd.read_csv("./data/movie_vector.csv")
movie_genre_vector = movie_genre_vector[['movieId', 'movie_vector']]

movie_genre_vector['movie_vector'] = movie_genre_vector['movie_vector'].apply(lambda x: x.replace('[', ' ').replace(']', ' ').strip().split())
movie_genre_vector['movie_vector'] = movie_genre_vector['movie_vector'].apply(lambda x: np.asarray(x).astype(float))

movie_era_vector = pd.read_csv("./data/movie_era_vector.csv")
movie_era_vector = movie_era_vector[['movieId', 'era_vector']]

movie_era_vector['era_vector'] = movie_era_vector['era_vector'].apply(lambda x: x.replace('[', ' ').replace(']', ' ').strip().split())
movie_era_vector['era_vector'] = movie_era_vector['era_vector'].apply(lambda x: np.asarray(x).astype(float))

merged_movie = movie_genre_vector.join(movie_era_vector['era_vector'])
merged_movie['final_movie_vector'] = merged_movie.apply(lambda x: np.concatenate((2*np.atleast_1d(x['movie_vector']), np.atleast_1d(x['era_vector']))), axis=1)
ratings_test = pd.read_csv("./data/test.csv", converters={"genres": literal_eval, "tag": literal_eval}) 

algo_predictions = pd.DataFrame(columns=['userId', 'movieId', 'user_vector', 'movie_vector', 'og_rating', 'pred_rating'])
error_count = 0
for ind, row in ratings_test.iterrows():
    userId = row['userId']
    movieId = row['movieId']
    og_rating = row['rating']
    
    user_vector = merged_user[merged_user['userId'] == int(userId)].final_user_vector.values[0]
    if len(merged_movie[merged_movie['movieId'] == int(movieId)].final_movie_vector.values):
        movie_vector = merged_movie[merged_movie['movieId'] == int(movieId)].final_movie_vector.values[0]
    else:
        error_count += 1
        print("Movie vector not found!", movieId)

    # Pad movie_vector with zeros to make it compatible for element-wise multiplication
    padded_movie_vector = np.pad(movie_vector, (0, len(user_vector) - len(movie_vector)))

    # Perform element-wise multiplication
    predicted_rating = user_vector * padded_movie_vector

    # predicted_rating = user_vector*movie_vector
    if predicted_rating.any():
        predicted_rating = np.nanmean(np.where(predicted_rating!=0, predicted_rating, np.nan))
    else:
        predicted_rating = 0
    
    row_df = pd.DataFrame([[userId, movieId, user_vector, movie_vector, og_rating, predicted_rating]], 
                columns=['userId', 'movieId', 'user_vector', 'movie_vector', 'og_rating', 'pred_rating'])
    algo_predictions = pd.concat([algo_predictions, row_df], ignore_index=True)

algo_predictions.to_csv("./data/movie_era_based_genre_ratings.csv")
rmse = ((algo_predictions.og_rating - algo_predictions.pred_rating/3) ** 2).mean() ** .5
print(rmse)
mae = (((algo_predictions.og_rating - algo_predictions.pred_rating/3) ** 2) ** .5).mean()
print(mae)