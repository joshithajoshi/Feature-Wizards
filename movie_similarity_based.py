import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

movies = pd.read_csv("./data/movies_tmdb.csv")

movies['description'] = movies['overview'].fillna('')

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english', min_df=1)
tfidf_matrix = tf.fit_transform(movies['description'])

print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

movies = movies.reset_index()
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

def get_recommendations(title):
    idx = indices[title]
    if type(idx) != np.int64:
        if len(idx)>1:
            print("ALERT: Multiple values")
            idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

print("Recomendations Based on overview for movie Doctor Who: Last Christmas")
print(get_recommendations('Doctor Who: Last Christmas').head(10))

popularity_df = movies[['popularity', 'vote_average', 'vote_count']]

movies['description_genre'] = movies['overview'] + 2*movies['genres']
movies['description_genre'] = movies['description_genre'].fillna('')

tf_new = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix_new = tf_new.fit_transform(movies['description_genre'])

cosine_sim_new = linear_kernel(tfidf_matrix_new, tfidf_matrix_new)

# tf_new.vocabulary_['scifi']

movies = movies.reset_index()
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])
indices.head(2)

def get_recommendations_new(title):
    idx = indices[title]
    if type(idx) != np.int64:
        if len(idx)>1:
            print("ALERT: Multiple values")
            idx = idx[0]
    sim_scores = list(enumerate(cosine_sim_new[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]
print("Recomendations Based on overview and genre for movie Doctor Who: Last Christmas")
print(get_recommendations_new('Doctor Who: Last Christmas').head(10))

vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

m = vote_counts.quantile(0.95)

def weighted_rating(x):
    v = int(x['vote_count'])
    R = int(x['vote_average'])
    return (v/(v+m) * R) + (m/(m+v) * C)

def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_new[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies_x = movies.iloc[movie_indices][['title', 'vote_count', 'vote_average']]
    vote_counts = movies_x[movies_x['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies_x[movies_x['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies_x[(movies_x['vote_count'] >= m) & (movies_x['vote_count'].notnull()) &
                       (movies_x['vote_average'].notnull())]
    qualified.loc[:, 'wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified

print("Recomendations Based on weighted rating, overview and genre for movie Doctor Who: Last Christmas")
print(improved_recommendations('Doctor Who: Last Christmas').head(10))
