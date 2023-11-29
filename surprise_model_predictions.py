from surprise import SVD, BaselineOnly, SVDpp, NMF, SlopeOne, CoClustering, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import accuracy
from surprise.model_selection import train_test_split

import pandas as pd
import numpy as np

def convert_traintest_dataframe_forsurprise(training_dataframe, testing_dataframe):
    reader = Reader(rating_scale=(0, 5))
    trainset = Dataset.load_from_df(training_dataframe[['userId', 'movieId', 'rating']], reader)
    testset = Dataset.load_from_df(testing_dataframe[['userId', 'movieId', 'rating']], reader)
    trainset = trainset.construct_trainset(trainset.raw_ratings)
    testset = testset.construct_testset(testset.raw_ratings)
    return trainset, testset

traindf = pd.read_csv('./data/train.csv')
testdf = pd.read_csv('./data/test.csv')
trainset, testset = convert_traintest_dataframe_forsurprise(traindf, testdf)

def recommendation(algo, trainset, testset):
  algo.fit(trainset)
  test_predictions = algo.test(testset)
  test_rmse = accuracy.rmse(test_predictions)
  test_mae = accuracy.mae(test_predictions)
  
  return test_rmse, test_mae, test_predictions

# KNNBaseline
sim_options = {'name': 'pearson_baseline',
               'user_based': False  # compute  similarities between items
               }
print('KNNBaseline')
algo = KNNBaseline(sim_options=sim_options)
test_knn_rmse, test_knn_mae, test_knn_pred = recommendation(algo, trainset, testset)

# SlopeOne
print('SlopeOne')
algo = SlopeOne()
test_slopeone_rmse, test_slopeone_mae, test_slopeone_pred = recommendation(algo, trainset, testset)

# SVD
print('SVD')
algo = SVD()
test_svd_rmse, test_svd_mae, test_svd_pred  = recommendation(algo, trainset, testset)

# SVDpp #Best
print('SVDpp')
algo = SVDpp()
test_svdpp_rmse, test_svdpp_mae, test_svdpp_pred = recommendation(algo, trainset, testset)

print('BaselineOnly')
algo = BaselineOnly()
test_base_rmse, test_base_mae, test_base_pred  = recommendation(algo, trainset, testset)

test_pred_df = pd.DataFrame(columns= ['uid', 'iid', 'og_rating', 'svd_rating', 'knn_rating', 'svdpp_rating', 'slopeone_rating', 'baseline_rating'])

num_test = len(test_base_pred)

for i in range(num_test): 
  svd = test_svd_pred[i]
  slopeone = test_slopeone_pred[i]
  knn = test_knn_pred[i]
  svdpp = test_svdpp_pred[i]
  baseline = test_base_pred[i]
  df = pd.DataFrame([[svd.uid, svd.iid, svd.r_ui, svd.est, knn.est, svdpp.est, slopeone.est, baseline.est]], columns=['uid', 'iid', 'og_rating', 'svd_rating', 'knn_rating', 'svdpp_rating', 'slopeone_rating','baseline_rating'])
  # print(df)
  test_pred_df = pd.concat([df, test_pred_df], ignore_index=True)


test_pred_df.to_csv('./preds/test_prediction.csv')