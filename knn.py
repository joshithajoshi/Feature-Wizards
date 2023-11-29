from surprise import SVD, BaselineOnly, SVDpp, NMF, SlopeOne, CoClustering
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

ratings_df = pd.read_csv("./data/ratings.csv")
reader = Reader(rating_scale=(1, 5))  # Assuming ratings are on a scale of 1 to 5
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=.20)

user_rmse = [0,0,0,0]
item_rmse = [0,0,0,0]
user_mae = [0,0,0,0]
item_mae = [0,0,0,0]

def recommendation(algo, trainset, testset):
  
  algo.fit(trainset)
  predictions = algo.test(testset)  
  return accuracy.rmse(predictions),accuracy.mae(predictions)

print("Cosine Item")
sim_options = {'name': 'cosine',
               'user_based': False 
               }
algo = KNNBaseline(sim_options=sim_options)
item_rmse[0],item_mae[0] = recommendation(algo, trainset, testset)

print("Cosine User")
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNBaseline(sim_options=sim_options)
user_rmse[0],user_mae[0] = recommendation(algo, trainset, testset)

print("MSD Item")
sim_options = {'name': 'msd',
               'user_based': False  
               }
algo = KNNBaseline(sim_options=sim_options)
item_rmse[1],item_mae[1] = recommendation(algo, trainset, testset)

print("MSD User")
sim_options = {'name': 'msd', 'user_based': True}
algo = KNNBaseline(sim_options=sim_options)
user_rmse[1],user_mae[1] = recommendation(algo, trainset, testset)

print("pearson Item")
sim_options = {'name': 'pearson',
               'user_based': False  
               }
algo = KNNBaseline(sim_options=sim_options)
item_rmse[2],item_mae[2] = recommendation(algo, trainset, testset)

print("pearson User")
sim_options = {'name': 'pearson',
               'user_based': True 
               }
algo = KNNBaseline(sim_options=sim_options)
user_rmse[2],user_mae[2] = recommendation(algo, trainset, testset)

print("pearson_baseline Item")
# Below one is the best and best for item based #
sim_options = {'name': 'pearson_baseline',
               'user_based': False  
               }
algo = KNNBaseline(sim_options=sim_options)
item_rmse[3],item_mae[3] = recommendation(algo, trainset, testset)

print("pearson_baseline Item k=60")
sim_options = {'name': 'pearson_baseline',
               'user_based': False  
               }
algo = KNNBaseline(k=60,sim_options=sim_options)
recommendation(algo, trainset, testset)

# Below one is the best for user based #
print("pearson_baseline User")
sim_options = {'name': 'pearson_baseline',
               'user_based': True  
               }
algo = KNNBaseline(sim_options=sim_options)
user_rmse[3],user_mae[3] = recommendation(algo, trainset, testset)

print("pearson_baseline User k=60")
sim_options = {'name': 'pearson_baseline',
               'user_based': True  
               }
algo = KNNBaseline(k=60,sim_options=sim_options)
recommendation(algo, trainset, testset)

x_algo = ['cosine', 'msd', 'pearson', 'pearson_baseline']

plt.figure(figsize=(15,5))
plt.title('Comparison of similarity metric on RMSE and MAE', loc='center', fontsize=15)
plt.plot(x_algo, user_rmse, label='User-user RMSE', color='darkgreen', marker='o')
plt.plot(x_algo, item_rmse, label='Item-item RMSE', color='navy', marker='o')
plt.xlabel('Similarity metric', fontsize=15)
plt.ylabel('Error Value', fontsize=15)
plt.plot(x_algo, user_mae, label='User-user MAE', color='darkgreen', marker='o')
plt.plot(x_algo, item_mae, label='Item-item MAE', color='navy', marker='o')
plt.legend()
plt.grid(ls='dashed')

plt.show()
