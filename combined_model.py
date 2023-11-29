import pandas as pd
import numpy as np
import math
     
pred_data = pd.read_csv('./preds/test_prediction.csv')

T = pred_data.shape[0]
    
svd_wt = 1.504
knn_wt = 0.712
svdpp_wt = 0.082
slopeone_wt = 0.881
baseline_wt = -2.534

sqr_sum = 0
abs_sum = 0

for ind, row in pred_data.iterrows():
  org_r = row['og_rating']
  pred_r = svd_wt*row['svd_rating'] + knn_wt*row['knn_rating'] + svdpp_wt*row['svdpp_rating'] + slopeone_wt*row['slopeone_rating'] + baseline_wt*row['baseline_rating']
  diff = np.abs(org_r - pred_r)
  # print(diff)
  abs_sum += diff
  sqr_sum += diff**2

rmse = np.sqrt(sqr_sum/T)
print("RMSE", rmse)
mae = abs_sum/T
print("MAE", mae)
     