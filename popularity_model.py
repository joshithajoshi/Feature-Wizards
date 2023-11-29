import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ast import literal_eval

movies = pd.read_csv("data/movies_tmdb.csv")
movies.sort_values(by='popularity', ascending=False)

genres = {'Adventure': 0,
 'Animation': 1,
 'Children': 2,
 'Comedy': 3,
 'Fantasy': 4,
 'Romance': 5,
 'Drama': 6,
 'Action': 7,
 'Crime': 8,
 'Thriller': 9,
 'Horror': 10,
 'Mystery': 11,
 'Sci-Fi': 12,
 'War': 13,
 'Musical': 14,
 'Documentary': 15,
 'IMAX': 16,
 'Western': 17,
 'Film-Noir': 18,
 '(no genres listed)': 19}

def genre_based_popularity(genre):
    mask = movies.genres.apply(lambda x: genre in x)
    filtered_movie = movies[mask]
    filtered_movie = filtered_movie.sort_values(by='popularity', ascending=False)
    return filtered_movie

vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

m = vote_counts.quantile(0.95)

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

movies['wr'] = movies.apply(weighted_rating, axis=1)

def genre_based_popularity_PT(genre):
    mask = movies.genres.apply(lambda x: genre in x)
    filtered_movie = movies[mask]
    filtered_movie = filtered_movie.sort_values(by='wr', ascending=False)
    return filtered_movie

print(genre_based_popularity_PT('Animation')[['title', 'wr', 'popularity']].head(10))