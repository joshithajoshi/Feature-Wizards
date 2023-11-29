import pandas as pd
import numpy as np
from surprise import SVD, BaselineOnly, SVDpp, NMF, SlopeOne, CoClustering, Reader
from surprise import Dataset
from surprise.prediction_algorithms import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import dump
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn import linear_model

def convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe):
    reader = Reader(rating_scale=(0, 5))
    trainset = Dataset.load_from_df(training_dataframe[['userId', 'movieId', 'rating']], reader)
    testset = Dataset.load_from_df(testing_dataframe[['userId', 'movieId', 'rating']], reader)
    testset = testset.construct_testset(trainset.raw_ratings)
    trainset = trainset.construct_trainset(trainset.raw_ratings)
    return trainset, testset

file_path_train = './data/train.csv'
file_path_test = './data/test.csv'
traindf = pd.read_csv(file_path_train)
testdf = pd.read_csv(file_path_test)
trainset, testset = convert_traintest_dataframe_forsurprise(traindf, testdf)

def train_predictions(algo):
    algo.fit(trainset)
    return algo.test(testset)

sim_options = {'name': 'cosine',
               'user_based': False
               }
algo = KNNBaseline(sim_options=sim_options)
train_knn_pred = train_predictions(algo)

algo = SVD()
train_svd_pred = train_predictions(algo)

algo = SVDpp()
train_svdpp_pred = train_predictions(algo) 

algo = SlopeOne()
train_slopeone_pred = train_predictions(algo)

algo = BaselineOnly()
train_base_pred = train_predictions(algo)

num_train = len(train_base_pred)
train_pred_df = pd.DataFrame(columns= ['uid', 'iid', 'og_rating', 'svd_rating', 'knn_rating', 'svdpp_rating', 'slopeone_rating', 'baseline_rating'])

for i in range(num_train): 
  svd = train_svd_pred[i]
  slopeone = train_slopeone_pred[i]
  knn = train_knn_pred[i]
  svdpp = train_svdpp_pred[i]
  baseline = train_base_pred[i]
  df = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, svd.est, knn.est, svdpp.est, slopeone.est, baseline.est]], columns=['uid', 'iid', 'og_rating', 'svd_rating', 'knn_rating', 'svdpp_rating', 'slopeone_rating','baseline_rating'])
  train_pred_df = pd.concat([df, train_pred_df], ignore_index=True)

train_pred_df.to_csv('./data/train_prediction.csv')

X_train = train_pred_df[['svd_rating', 'knn_rating', 'svdpp_rating', 'slopeone_rating', 'baseline_rating']]
y_train = train_pred_df['og_rating']

reg = linear_model.LinearRegression()
 
# train the model using the training sets
reg.fit(X_train, y_train)
 
# regression coefficients
print('Coefficients: ', reg.coef_)


