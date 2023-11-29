# Feature-Wizards
Movie Recommendation System

### Data Pre-Processing
#### Ordering of files to run:
1) run remove.py to remove the duplicate entries from the dataset.
2) run data_preprocess.py to get processed movies dataset.
3) run scraper.py to scrape data from tmdb website
4) run converter.py to convert json file obtained above.

### Model
#### 1) Popularity Based:
1) run popularity_model.py to get popularity based recommendations.

#### 2) Content-based:
 1) Run content_based.py 
 2) run movie_similarity_based.py
 3) run movie_year_analysis.py
 4) run movie_era_based.py

#### 3) Collaborative-Filtering:
 1) run knn.py to identify best variation of knn
 2) run surprise_model_predictions.py to test various matrix Factorization-based algorithms
 3) run weight_training.py to get best parameter values for the combined_model.py which combines the above algorithms.
 4) surprise_model_recs.py gives recomendaions using matrix factorization-based algorithms.

#### Hybrid model
run hybrid_model.py - this combines all the models to get predictions.

#### Files to run to get Recommendations:
 1) user.ipynb to get user based recommendations
 2) movie.ipynb to get movie based recommendations

#### Files to run to get analysis:
 1) plot.py
 2) plot.ipynb
 3) cold_start_analysis.ipynb
 4) model_hyperparameter_tuning.py
