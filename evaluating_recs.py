import math
from collections import defaultdict
import csv
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd


def get_top_n(predictions, algo_weights, n):
    
    top_n = defaultdict(list)
    top_n_ndcg = defaultdict(list)
    for i in range(len(predictions)):
        row = predictions.iloc[i, :]
        final_est = algo_weights['svd']*float(row['svd_rating']) + algo_weights['knn']*float(row['knn_rating']) + \
                    algo_weights['svdpp']*float(row['svdpp_rating']) + algo_weights['slope']*float(row['slopeone_rating']) + \
                    algo_weights['baseline']*float(row['baseline_rating'])
        top_n[row[0]].append((row[1], final_est))
        top_n_ndcg[row[0]].append((row[1], row[2], final_est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    for uid, user_ratings in top_n_ndcg.items():
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        top_n_ndcg[uid] = user_ratings[:n]

    return top_n, top_n_ndcg


def precision_recall_at_k(predictions, algo_weights, k, threshold):
    user_est_true = defaultdict(list)
    for i in range(len(predictions)):
        row = predictions.iloc[i, :]
        final_est = algo_weights['svd']*float(row['svd_rating']) + algo_weights['knn']*float(row['knn_rating']) + \
                    algo_weights['svdpp']*float(row['svdpp_rating']) + algo_weights['slope']*float(row['slopeone_rating']) + \
                    algo_weights['baseline']*float(row['baseline_rating'])
        user_est_true[row[0]].append((final_est, row[2]))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k/n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k/n_rel if n_rel != 0 else 1
    return precisions, recalls


def dcg_at_k(scores):
    return scores[0] + sum(sc/math.log(ind, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))


def ndcg_at_k(predicted_scores, actual_scores):
    idcg = dcg_at_k(sorted(actual_scores, reverse=True))
    return (dcg_at_k(predicted_scores)/idcg) if idcg > 0.0 else 0.0


predictions = pd.read_csv("./preds/test_prediction.csv", usecols=range(1, 9))
algo_weights = dict()
algo_weights['svd'] = 0.1
algo_weights['knn'] = 0.5
algo_weights['svdpp'] = 0.3
algo_weights['slope'] = 0
algo_weights['baseline'] = 0.1
n = 5
threshold = 3.75
top_n, top_n_ndcg = get_top_n(predictions, algo_weights, n)
with open('top5_svdpp.csv', 'w', newline="") as csv_file:
    writer = csv.writer(csv_file)
    for key, value in top_n.items():
        writer.writerow([key, value])

ndcg_scores = dict()
for uid, user_ratings in top_n_ndcg.items():
    true = []
    est = []
    for _, tru_r, est_r in user_ratings:
        true.append(tru_r)
        est.append(est_r)
    ndcg = ndcg_at_k(est, true)
    ndcg_scores[uid] = ndcg

precisions, recalls = precision_recall_at_k(predictions, algo_weights, n, threshold)
precision = sum(prec for prec in precisions.values())/len(precisions)
recall = sum(rec for rec in recalls.values())/len(recalls)
fmeasure = (2*precision*recall)/(precision + recall)
ndcg_score = sum(ndcg for ndcg in ndcg_scores.values())/len(ndcg_scores)
final_pred = algo_weights['svd']*predictions['svd_rating'] + algo_weights['knn']*predictions['knn_rating'] + \
                    algo_weights['svdpp']*predictions['svdpp_rating'] + algo_weights['slope']*predictions['slopeone_rating'] + \
                    algo_weights['baseline']*predictions['baseline_rating']
rmse = np.mean((final_pred - predictions['og_rating'])**2)**0.5
mae = np.mean(abs(final_pred - predictions['og_rating']))
print("RMSE: ",rmse)
print("Mae: ",mae)
print("Precision: ", precision)
print("Recall: ", recall)
print("F-Measure", fmeasure)
print("NDCG Score: ", ndcg_score)