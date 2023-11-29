import pandas as pd
from sklearn.model_selection import train_test_split

movies = pd.read_csv('../data/movies.csv')
ratings = pd.read_csv('../data/ratings.csv')
tags = pd.read_csv('../data/tags.csv')

df = pd.merge(ratings, movies, on='movieId' , how='left')
df = df.drop('title', axis=1)

df['genres'] = df['genres'].str.split('|')
tags['tag'] = tags['tag'].str.split('|')
tags.drop('timestamp', axis=1, inplace=True)
tags = tags.groupby(['userId', 'movieId'])['tag'].apply(list).reset_index()
tags['tag'] = tags['tag'].apply(lambda d: sum(d,[]))

df = pd.merge(df, tags, on=['userId','movieId'], how='left')
df['tag'] = df['tag'].apply(lambda d: d if isinstance(d, list) else [])
df['genres'] = df['genres'].apply(lambda d: d if isinstance(d, list) else [])

train_data, test_data = train_test_split(df, test_size=0.2, stratify=df.userId)
train_data = train_data.sort_values(['userId', 'movieId'])
test_data = test_data.sort_values(['userId','movieId'])

train_data.to_csv('../data/train.csv', index = False)
test_data.to_csv('../data/test.csv', index = False)

movies['genres'] = movies['genres'].str.split('|')
movies['genres'] = movies['genres'].apply(lambda d: d if isinstance(d, list) else [])
movies.to_csv('../data/clean_movies.csv', index = False)

