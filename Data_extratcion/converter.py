import json
import csv
import pandas as pd
import numpy as np

# Replace 'input.json' with your JSON file's name
with open('./data/recommendation_metadata.json', 'r') as json_file:
    data_list = json.load(json_file)

# Assuming your JSON data is a list of dictionaries
results_list = [item.get("results", []) for item in data_list]

# Flatten the list of lists into a single list
results_list = [item for sublist in results_list for item in sublist]

# Replace 'output.csv' with the desired CSV file name
csv_file = './data/movies_metadata.csv'

# Check if the results_list is not empty
if results_list:
    # Write data to the CSV file
    with open(csv_file, 'w', newline='') as csv_output:
        writer = csv.DictWriter(csv_output, fieldnames=results_list[0].keys())

        # Write the header row
        writer.writeheader()

        # Write the data rows
        for row in results_list:
            writer.writerow(row)
else:
    print("The results_list is empty. Cannot write CSV.")

md = pd.read_csv('data/movies_metadata.csv')
md = md[["id","title","overview","popularity","release_date","vote_average","vote_count"]]
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
md = md.drop_duplicates(subset="id")
print(md.iloc[0])
df1 = pd.read_csv("../data/movies.csv")
df1['title'] = df1['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
print(df1.iloc[0])
# df1['year'] = df1['title'].str.extract(r'\((\d{4})\)')
merged_df = pd.merge(df1, md, on='title')
merged_df = merged_df.drop("id",axis=1)
merged_df = merged_df.drop_duplicates("movieId")
merged_df['genres'] = merged_df['genres'].str.split('|')
merged_df['genres'] = merged_df['genres'].apply(lambda d: d if isinstance(d, list) else [])

print(merged_df.shape)
print(merged_df.iloc[0])
merged_df.to_csv("../data/movies_tmdb.csv")
