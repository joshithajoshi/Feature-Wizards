import pandas as pd
import numpy as np
from surprise import SVD, BaselineOnly, SVDpp, NMF, SlopeOne, CoClustering, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

def convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe):
    reader = Reader(rating_scale=(0, 5))
    trainset = Dataset.load_from_df(training_dataframe[['userId', 'movieId', 'rating']], reader)
    testset = Dataset.load_from_df(testing_dataframe[['userId', 'movieId', 'rating']], reader)
    trainset = trainset.construct_trainset(trainset.raw_ratings)
    testset = testset.construct_testset(testset.raw_ratings)
    return trainset, testset

file_path_train = './data/train.csv'
file_path_test = './data/test.csv'
traindf = pd.read_csv(file_path_train)
testdf = pd.read_csv(file_path_test)
trainset, testset = convert_traintest_dataframe_forsurprise(traindf, testdf)

sim_options = {'name': 'cosine',
               'user_based': False
               }
knnbaseline_algo = KNNBaseline(sim_options=sim_options)
knnbaseline_algo.fit(trainset)

svd_algo = SVD()
svd_algo.fit(trainset)

svdpp_algo = SVDpp()
svdpp_algo.fit(trainset)

slopeone_algo = SlopeOne()
slopeone_algo.fit(trainset)

baseline_algo = BaselineOnly()
baseline_algo.fit(trainset)

movies = pd.read_csv("./data/movies_tmdb.csv")

genre_to_idx = {'Adventure': 0,
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

idx_to_genre = {0: 'Adventure',
 1: 'Animation',
 2: 'Children',
 3: 'Comedy',
 4: 'Fantasy',
 5: 'Romance',
 6: 'Drama',
 7: 'Action',
 8: 'Crime',
 9: 'Thriller',
 10: 'Horror',
 11: 'Mystery',
 12: 'Sci-Fi',
 13: 'War',
 14: 'Musical',
 15: 'Documentary',
 16: 'IMAX',
 17: 'Western',
 18: 'Film-Noir',
 19: '(no genres listed)'}

movies['description_genre'] = movies['overview'] + 2*movies['genres']
movies['description_genre'] = movies['description_genre'].fillna('')

tf_new = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix_new = tf_new.fit_transform(movies['description_genre'])

cosine_sim_new = linear_kernel(tfidf_matrix_new, tfidf_matrix_new)

movies = movies.reset_index()
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

def get_recommendations_new(title):
    idx = indices[title]
    if type(idx) != np.int64:
        if len(idx)>1:
            print("ALERT: Multiple values")
            idx = idx[0]
    sim_scores = list(enumerate(cosine_sim_new[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['movieId'].iloc[movie_indices]

def genre_based_popularity(genre):
    mask = movies.genres.apply(lambda x: genre in x)
    filtered_movie = movies[mask]
    filtered_movie = filtered_movie.sort_values(by='popularity', ascending=False)
    return filtered_movie['movieId'].head(10).values.tolist() 

user_info = pd.read_csv('./data/user_info.csv')

user_info['user_vector'] = user_info['user_vector'].apply(lambda x: x.replace('[', ' ').replace(']', ' ').strip().split())
user_info['user_vector'] = user_info['user_vector'].apply(lambda x: np.asarray(x).astype(float))

def user_top_genre(userId):
    user_vec = user_info['user_vector'][user_info['userId'] == userId].values[0].copy()
    top_genre_indices = np.flip(np.argsort(user_vec))
    genre_list = []
    for i in top_genre_indices[:3]:
        genre_list.append(idx_to_genre[i])
    return genre_list

user_list = testdf['userId'].unique()
svd_wt = 1.504
knn_wt = 0.712
svdpp_wt = 0.08
slopeone_wt = 0.88
baseline_wt = -2.53

def hybrid(userId):
    user_movies = testdf[testdf['userId'] == userId]
    user_movies['est'] = user_movies['movieId'].apply(lambda x: knn_wt*knnbaseline_algo.predict(userId, x).est + svdpp_wt*svdpp_algo.predict(userId, x).est\
              + svd_wt*svd_algo.predict(userId, x).est + baseline_wt*baseline_algo.predict(userId, x).est\
                +slopeone_wt*slopeone_algo.predict(userId,x).est)    
    user_movies = user_movies.sort_values(by ='est', ascending=False).head(4)
    user_movies['Model'] = 'SVD + CF'
    
    recommend_list = user_movies[['movieId', 'est', 'Model']]
    
    movie_list = recommend_list['movieId'].values.tolist()
    print(movie_list)
    sim_movies_list = []
    for movie_id in movie_list:
        # Call content based 
        movie_title = movies['title'][movies['movieId'] == movie_id].values[0]
        sim_movies = get_recommendations_new(movie_title)
        sim_movies_list.extend(sim_movies)
        
    for movie_id in sim_movies_list:
        pred_rating = knn_wt*knnbaseline_algo.predict(userId, movie_id).est + svdpp_wt*svdpp_algo.predict(userId, movie_id).est\
              + svd_wt*svd_algo.predict(userId, movie_id).est + baseline_wt*baseline_algo.predict(userId, movie_id).est\
                +slopeone_wt*slopeone_algo.predict(userId,movie_id).est
        row_df = pd.DataFrame([[movie_id, pred_rating, 'Movie similarity']], columns=['movieId', 'est','Model'])
        recommend_list = pd.concat([recommend_list, row_df], ignore_index=True)
    
    # Popular based movies
    top_genre_list = user_top_genre(userId)
    
    popular_movies = []
    for top_genre in top_genre_list:
        popular_movies.extend(genre_based_popularity(top_genre))
    
    # Compute ratings for the popular movies
    for movie_id in popular_movies:
        pred_rating = knn_wt*knnbaseline_algo.predict(userId, movie_id).est + svdpp_wt*svdpp_algo.predict(userId, movie_id).est\
              + svd_wt*svd_algo.predict(userId, movie_id).est + baseline_wt*baseline_algo.predict(userId, movie_id).est\
                +slopeone_wt*slopeone_algo.predict(userId,movie_id).est
        row_df = pd.DataFrame([[movie_id, pred_rating, 'Popularity']], columns=['movieId', 'est','Model'])
        recommend_list = pd.concat([recommend_list, row_df], ignore_index=True)
    recommend_list = recommend_list.drop_duplicates(subset=['movieId'])
    train_movie_list = traindf[traindf['userId']==userId]['movieId'].values.tolist()
    
    # Remove movies in training for this user
    mask = recommend_list.movieId.apply(lambda x: x not in train_movie_list)
    recommend_list = recommend_list[mask]
    
    return recommend_list

def get_title(x):
    mid = x['movieId']
    return movies['title'][movies['movieId'] == mid].values

def get_genre(x):
    mid = x['movieId']
    return movies['genres'][movies['movieId'] == mid].values

movie_ids = hybrid(1)
movie_ids['title'] = movie_ids.apply(get_title,axis=1)
movie_ids['genre'] = movie_ids.apply(get_genre,axis=1)
print(movie_ids['title'].head())











