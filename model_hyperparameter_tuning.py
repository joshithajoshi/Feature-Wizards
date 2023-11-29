from surprise import SVD, BaselineOnly, SVDpp, NMF, SlopeOne, CoClustering, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import accuracy
from surprise.model_selection import train_test_split
import time
import pandas as pd
from surprise import NormalPredictor
from surprise.model_selection import GridSearchCV

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

def recommendation(algo, trainset, testset):
  # Train the algorithm on the trainset, and predict ratings for the testset
  start_fit = time.time()
  algo.fit(trainset)
  end_fit = time.time()
  fit_time = end_fit - start_fit

  # Predictions on testing set
  start_test = time.time()
  test_predictions = algo.test(testset)
  end_test = time.time()
  test_time = end_test - start_test
  test_rmse = accuracy.rmse(test_predictions)
  test_mae = accuracy.mae(test_predictions)
  
  return test_rmse, test_mae, test_predictions, fit_time, test_time

reader = Reader(rating_scale=(0, 5))     
data = Dataset.load_from_df(traindf[['userId', 'movieId', 'rating']], reader)
param_grid = {'n_factors':[25,50,100], 'n_epochs': [5, 10, 20], 'lr_all': [0.01, 0.02],
              'reg_all': [0.01,0.02]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
gs.fit(data)
svd_rmse = gs.best_score['rmse']
svd_mae = gs.best_score['mae']
print(gs.best_params['rmse'])

param_grid = {
  'sim_options': {
      'name': ['cosine'],  # Specify the similarity measures you want to test
      'user_based': [False],  # Whether to use user-based or item-based collaborative filtering
  },
  'k': [15, 20, 25, 30, 40, 50, 60]
}

knnbasic_gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
knnbasic_gs.fit(data)
knnbasic_rmse = knnbasic_gs.best_score['rmse']
knnbasic_mae = knnbasic_gs.best_score['mae']

knnmeans_gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
knnmeans_gs.fit(data)
knnmeans_rmse = knnmeans_gs.best_score['rmse']
knnmeans_mae = knnmeans_gs.best_score['mae']

knnz_gs = GridSearchCV(KNNWithZScore, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
knnz_gs.fit(data)
knnz_rmse = knnz_gs.best_score['rmse']
knnz_mae = knnz_gs.best_score['mae']

knnbaseline_gs = GridSearchCV(KNNBaseline, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
knnbaseline_gs.fit(data)  
knnbaseline_rmse = knnbaseline_gs.best_score['rmse']
knnbaseline_mae = knnbaseline_gs.best_score['mae']


x = [15, 20, 25, 30, 40, 50, 60]
y1 = knnbasic_gs.cv_results['mean_test_rmse']
y2 = knnbasic_gs.cv_results['mean_test_mae']

y3 = knnmeans_gs.cv_results['mean_test_rmse']
y4 = knnmeans_gs.cv_results['mean_test_mae']

y5 = knnz_gs.cv_results['mean_test_rmse']
y6 = knnz_gs.cv_results['mean_test_mae']

y7 = knnbaseline_gs.cv_results['mean_test_rmse']
y8 = knnbaseline_gs.cv_results['mean_test_mae']
     

import matplotlib.pyplot as plt

plt.figure(figsize=(18,5))

plt.subplot(1, 2, 1)
plt.title('K Neighbors vs RMSE', loc='center', fontsize=15)
plt.plot(x, y1, label='KNNBasic', color='lightcoral', marker='o')
plt.plot(x, y3, label='KNNWithMeans', color='darkred', marker='o')
plt.plot(x, y5, label='KNNWithZScore', color='indianred', marker='o')
plt.plot(x, y7, label='KNNWithBaseline', color='red', marker='o')
plt.xlabel('K Neighbor', fontsize=15)
plt.ylabel('RMSE Value', fontsize=15)
plt.legend()
plt.grid(ls='dotted')

plt.subplot(1, 2, 2)
plt.title('K Neighbors vs MAE', loc='center', fontsize=15)
plt.plot(x, y2, label='KNNBasic', color='lightcoral', marker='o')
plt.plot(x, y4, label='KNNWithMeans', color='indianred', marker='o')
plt.plot(x, y6, label='KNNWithZScore', color='darkred', marker='o')
plt.plot(x, y8, label='KNNWithBaseline', color='red', marker='o')
plt.xlabel('K Neighbor', fontsize=15)
plt.ylabel('MAE Value', fontsize=15)
plt.legend()
plt.grid(ls='dotted')

plt.show()