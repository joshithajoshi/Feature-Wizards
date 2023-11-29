import requests
import time
import json
import pandas as pd

API_KEY = "b1b36c58f167ff3fbad78a116a4a0970"
base_url = "https://api.themoviedb.org/3/movie/"

json_file = open('./data/recommendation_metadata.json', 'w')

json_file.write(' [')
length = pd.read_csv('./data/links.csv').shape[0]
with open('data/links.csv', 'r') as f:
	for count,line in enumerate(f):
		line = line.split(',')
		tmdbId = line[2]
		if len(tmdbId) == 0:
			print("Movie doesn't exist is TMDB Database.")
			continue
		url = base_url + tmdbId + "/recommendations"  + "?api_key=" + API_KEY

		print("Requesting Movie Number: " + str(count))
		res = requests.get(url)

		if res.status_code == 404:
			print("Movie doesn't exist is TMDB Database.")
			continue
		elif res.status_code == 500:
			print("Internal Server Error.")
			continue
		elif res.status_code == 429:
			print("Time out. Waiting for 10 seconds")
			time.sleep(10)
			res = requests.get(url)
		else:
			res.raise_for_status()

		js = dict(json.loads(res.text))
		js["id"] = tmdbId
		json_file.write(str(json.dumps(js)))
		if count == length-1:
			break
		json_file.write(',')

json_file.write(' ]')
json_file.close()

