import matplotlib.pyplot as plt
import pandas as pd

result = pd.read_csv("./preds/Surprise_results.csv")
x_algo = result['Algorithm'].tolist()

rmse = result['test_rmse'].tolist()
mae = result['test_mae'].tolist()
fit_time = result['fit_time'].tolist()
test_time = result['test_time'].tolist()
precision = result['Precision'].tolist()
recall = result['Recall'].tolist()
f_measure = result['F-measure'].tolist()
ndcg = result['NDCG'].tolist()

plt.figure(figsize=(15,7))

# plt.subplot(1, 2, 1)
plt.title('Comparison of Algorithms on RMSE', loc='center', fontsize=15)
plt.plot(x_algo, rmse, label='RMSE', color='darkgreen', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('RMSE Value', fontsize=15)

plt.legend()
plt.grid(ls='dashed')
plt.show()

# plt.subplot(1, 2, 2)
plt.figure(figsize=(15,7))
plt.title('Comparison of Algorithms on MAE', loc='center', fontsize=15)
plt.plot(x_algo, mae, label='MAE', color='navy', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('MAE Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')

plt.show()

plt.figure(figsize=(15,6))
plt.title('Comparison of Algorithms on RMSE and MAE', loc='center', fontsize=15)
plt.plot(x_algo, rmse, label='RMSE', color='darkgreen', marker='o')
plt.plot(x_algo, mae, label='MAE', color='navy', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')
plt.show()

plt.figure(figsize=(15,6))
plt.title('Comparison of Algorithms on fit and train time', loc='center', fontsize=15)
plt.plot(x_algo, fit_time, label='Fit Time', color='navy', marker='o')
plt.plot(x_algo, test_time, label='Test Time', color='red', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('Time (s)', fontsize=15)
plt.legend()
plt.grid(ls='dashed')
plt.show()

plt.figure(figsize=(15,6))
plt.title('Comparison of Algorithms on evluation parameters', loc='center', fontsize=15)
plt.plot(x_algo, precision, label='Precision', color='darkgreen', marker='o')
plt.plot(x_algo, recall, label='Recall', color='navy', marker='o')
plt.plot(x_algo, f_measure, label='F-Measure', color='darkred', marker='o')
# plt.plot(x_algo, ndcg, label='NDCG', color='purple', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.xticks(rotation=20)
plt.ylabel('Value', fontsize=15)
plt.legend()
plt.grid(ls='dashed')
plt.show()


plt.figure(figsize=(15,6))
plt.title('Comparison of Algorithms on NDCG', loc='center', fontsize=15)
plt.plot(x_algo, ndcg, label='NDCG', color='purple', marker='o')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('Value', fontsize=15)
plt.xticks(rotation=20)
plt.legend()
plt.grid(ls='dashed')
plt.show()