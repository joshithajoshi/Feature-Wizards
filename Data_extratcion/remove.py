import pandas as pd

df1 = pd.read_csv("../data/movies_tmdb.csv")
df2 = pd.read_csv("../data/movies.csv")
df3 = pd.read_csv("../data/ratings.csv")
df4 = pd.read_csv("../data/tags.csv")

# Drop rows from df1 based on the identified mask
print(df2.shape)
mask = df2['movieId'].isin(df1['movieId'])
df2_filtered = df2[mask]
print("Filtered DataFrame:")
print(df2_filtered.shape)
df2_filtered.to_csv("../data/movies.csv",index=False)

print(df3.shape)
mask = df3['movieId'].isin(df1['movieId'])
df3_filtered = df3[mask]
print("Filtered DataFrame:")
print(df3_filtered.shape)
df3_filtered.to_csv("../data/ratings.csv",index=False)

print(df4.shape)
mask = df4['movieId'].isin(df1['movieId'])
df4_filtered = df4[mask]
print("Filtered DataFrame:")
print(df4_filtered.shape)
df4_filtered.to_csv("../data/tags.csv",index=False)