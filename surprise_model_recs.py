from surprise import SVD, BaselineOnly, SVDpp, NMF, SlopeOne, CoClustering, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import accuracy
from surprise.model_selection import train_test_split
import math
from collections import defaultdict
import csv
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd
import time

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

def get_top_n(predictions, n):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    org_ratings = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        org_ratings[uid].append((iid, true_r))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n, org_ratings

def dcg_at_k(scores):
    return scores[0] + sum(sc/math.log(ind, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))

def ndcg_at_k(scores):
    idcg = dcg_at_k(sorted(scores, reverse=True))
    return (dcg_at_k(scores)/idcg) if idcg > 0.0 else 0.0

def precision_recall_at_k(predictions, k=5, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    precision = (sum(prec for prec in precisions.values()) / len(precisions))
    recall = (sum(rec for rec in recalls.values()) / len(recalls))

    return precision, recall


def recommendation(algo, trainset, testset):
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

  top_n, org_ratings = get_top_n(test_predictions, 5)

  precision, recall = precision_recall_at_k(test_predictions)

  f_measure = (2*precision*recall)/(precision+recall)

  ndcg_scores = dict()
  for uid, user_ratings in top_n.items():
    scores = []
    for iid, est_r in user_ratings:
        iid_found = False
        org_user_ratings = org_ratings[uid]
        for i, r in org_user_ratings:
            if iid == i:
                scores.append(r)
                iid_found = True
                break
        if not iid_found:
            scores.append(0)
    ndcg_scores[uid] = ndcg_at_k(scores)
  ndcg_score = sum(ndcg for ndcg in ndcg_scores.values())/len(ndcg_scores)

  return (test_rmse, test_mae, fit_time, test_time, precision, recall, f_measure, ndcg_score,test_predictions)

surprise_df = pd.DataFrame(columns= ['Algorithm', 'test_rmse', 'test_mae', 'fit_time', 'test_time', 'Precision', 'Recall', 'F-measure', 'NDCG'])
     
# Iterate over all algorithms
for algorithm in [KNNBasic(), SVD(), SVDpp(), SlopeOne(), KNNBaseline(), KNNWithMeans(), KNNWithZScore(), BaselineOnly()]:
    results = recommendation(algorithm,trainset,testset) 
    
    name =str(algorithm).split(' ')[0].split('.')[-1]
    print("Algorithm:", name)
    df = pd.DataFrame([[name, results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]]], columns= ['Algorithm', 'test_rmse', 'test_mae', 'fit_time', 'test_time', 'Precision', 'Recall', 'F-measure', 'NDCG'])
    surprise_df = pd.concat([df, surprise_df], ignore_index=True)


sim_options = {'name': 'pearson_baseline',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBaseline(sim_options=sim_options)
print("Algorithm: KNNBaseLine (pearson_baseline)")
results = recommendation(algo,trainset,testset)
df = pd.DataFrame([['KNNBaseline (pearson_baseline)', results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]]], columns= ['Algorithm', 'test_rmse', 'test_mae', 'fit_time', 'test_time', 'Precision', 'Recall', 'F-measure', 'NDCG'])
surprise_df = pd.concat([df, surprise_df], ignore_index=True)

# algo = CoClustering(2,5,50)
# print("Algorithm: CoClustering(2,5,50)")
# results = recommendation(algo,trainset,testset)
# df = pd.DataFrame([['CoClustering(2,5,50)', results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]]], columns= ['Algorithm', 'test_rmse', 'test_mae', 'fit_time', 'test_time', 'Precision', 'Recall', 'F-measure', 'NDCG'])
# surprise_df = pd.concat([df, surprise_df], ignore_index=True)

# surprise_df=surprise_df.round(2)

surprise_df = surprise_df.sort_values(by='NDCG', ascending=False)
surprise_df.to_csv('./preds/Surprise_results.csv')
     