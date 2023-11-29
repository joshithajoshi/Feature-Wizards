import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import pdb

# Load movies data
md = pd.read_csv("data/movies_tmdb.csv")

# mask = md['year'] == 'NaT'
# md_refined = md[~mask]
md_refined = md

plt.hist(md_refined['year'].astype(int), bins=15)
plt.xlabel('Year')
plt.ylabel('Number of movies')
plt.title('Histogram')
plt.show()

def get_era(x):
    year = int(x['year'])
    era = ''
    if year > 2010:
        era = '10s' 
    elif year >= 2000:
        era = '2000s'
    elif year > 1980: 
        era = '90s'
    else:
        era = 'Old'
    return era

def get_era_vector(x):
    year = int(x['year'])
    era_vector = np.zeros(4)
    if year > 2010:
        era_vector[3] = 1
    elif year >= 2000:
        era_vector[2] = 1
    elif year > 1980: 
        era_vector[1] = 1
    else:
        era_vector[0] = 1
    return era_vector

md_refined['era'] = md_refined.apply(get_era, axis=1)
md_refined['era_vector'] = md_refined.apply(get_era_vector, axis=1)

md_refined[['movieId', 'era_vector']].to_csv('./data/movie_era_vector.csv')

plt.pie(md_refined['era'].value_counts(), labels = md_refined['era'].unique())
plt.show()

ratings_train = pd.read_csv("./data/train.csv", converters={"genres": literal_eval, "tag": literal_eval})

user_ids = ratings_train['userId'].unique()
user_df = pd.DataFrame(columns=['userId', 'user_era_vector'])
error_count = 0
for user_id in user_ids:
    user_rating_df = ratings_train[(ratings_train['userId'] == user_id)]
    user_vector = np.zeros(4)
    count_vector = np.zeros(4)
    for _, row in user_rating_df.iterrows():
        if len(md_refined['era_vector'][md_refined['movieId'] == row['movieId']].values):
            user_movie_vector = md_refined['era_vector'][md_refined['movieId'] == row['movieId']].values[0]
            count_vector += user_movie_vector
            user_vector += user_movie_vector*row.rating
        else:
            error_count += 1
    count_vector = np.where(count_vector==0, 1, count_vector)
    user_vector = np.divide(user_vector, count_vector)
    row_df = pd.DataFrame([[user_id, user_vector]], columns=['userId', 'user_era_vector'])
    user_df = pd.concat([user_df, row_df], ignore_index=True)

user_df.to_csv("./data/user_era_vector.csv")

ratings_test = pd.read_csv("data/test.csv", converters={"genres": literal_eval, "tag": literal_eval}) 
ratings_test.head()

algo_predictions = pd.DataFrame(columns=['userId', 'movieId', 'user_vector', 'movie_vector', 'og_rating', 'pred_rating'])
error_count = 0
for ind, row in ratings_test.iterrows():
    userId = row['userId']
    movieId = row['movieId']
    og_rating = row['rating']
    
    user_vector = user_df[user_df['userId'] == int(userId)].user_era_vector.values[0]
    if len(md_refined[md_refined['movieId'] == int(movieId)].era_vector.values):
        movie_vector = md_refined[md_refined['movieId'] == int(movieId)].era_vector.values[0]
    else:
        error_count += 1
        print("Movie vector not found!", movieId)
    predicted_rating = user_vector*movie_vector

    if predicted_rating.any():
        predicted_rating = np.nanmean(np.where(predicted_rating!=0, predicted_rating, np.nan))
    else:
        predicted_rating = 0

    row_df = pd.DataFrame([[userId, movieId, user_vector, movie_vector, og_rating, predicted_rating]], 
                columns=['userId', 'movieId', 'user_vector', 'movie_vector', 'og_rating', 'pred_rating'])
    algo_predictions = pd.concat([algo_predictions, row_df], ignore_index=True)

algo_predictions.to_csv("./data/movie_year_based_genre_ratings.csv")
rmse = ((algo_predictions.og_rating - algo_predictions.pred_rating) ** 2).mean() ** .5
mae = (((algo_predictions.og_rating - algo_predictions.pred_rating) ** 2) ** .5).mean()

print(rmse)
print(mae)

algo_predictions.to_csv("./data/era_predictions.csv")