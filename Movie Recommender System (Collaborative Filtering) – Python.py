import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt

# -----------------------
# 1. Load and Preprocess Data
# -----------------------

# Example data: You would replace this with actual MovieLens or custom dataset
# movies.csv: movieId, title
# ratings.csv: userId, movieId, rating

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Create user-item rating matrix
rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Fill missing values with 0 (for simplicity; alternatives include imputation)
rating_matrix_filled = rating_matrix.fillna(0)

# Convert to sparse matrix for efficiency
sparse_matrix = csr_matrix(rating_matrix_filled.values)

# -----------------------
# 2. Similarity Computation
# -----------------------

# User-based similarity
user_similarity = cosine_similarity(sparse_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# Item-based similarity
item_similarity = cosine_similarity(sparse_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=rating_matrix.columns, columns=rating_matrix.columns)

# -----------------------
# 3. Predict Ratings
# -----------------------

def predict_user_based(user_id, movie_id, k=5):
    if movie_id not in rating_matrix.columns or user_id not in rating_matrix.index:
        return np.nan
    
    # Extract similarity scores for target user
    user_sim_scores = user_similarity_df.loc[user_id]
    
    # Extract ratings by other users for this movie
    movie_ratings = rating_matrix[movie_id]
    
    # Drop users who haven't rated the movie
    valid_users = movie_ratings[movie_ratings > 0].index
    sim_scores = user_sim_scores.loc[valid_users]
    
    # Select top-k most similar users
    top_k_users = sim_scores.sort_values(ascending=False).head(k)
    top_k_ratings = movie_ratings.loc[top_k_users.index]
    
    if top_k_users.sum() == 0:
        return np.nan

    # Weighted average
    predicted_rating = np.dot(top_k_users, top_k_ratings) / top_k_users.sum()
    return predicted_rating

def predict_item_based(user_id, movie_id, k=5):
    if user_id not in rating_matrix.index or movie_id not in rating_matrix.columns:
        return np.nan

    # Extract ratings by the user
    user_ratings = rating_matrix.loc[user_id]
    
    # Extract similarity scores for the target movie
    item_sim_scores = item_similarity_df[movie_id]
    
    # Filter only movies rated by the user
    rated_movies = user_ratings[user_ratings > 0].index
    sim_scores = item_sim_scores.loc[rated_movies]
    
    # Select top-k most similar movies
    top_k_items = sim_scores.sort_values(ascending=False).head(k)
    top_k_ratings = user_ratings.loc[top_k_items.index]
    
    if top_k_items.sum() == 0:
        return np.nan
    
    # Weighted average
    predicted_rating = np.dot(top_k_items, top_k_ratings) / top_k_items.sum()
    return predicted_rating

# -----------------------
# 4. Evaluation (RMSE)
# -----------------------

# Test on a sample of ratings
test_data = ratings.sample(frac=0.05, random_state=42)

user_based_preds = []
item_based_preds = []
actual_ratings = []

for _, row in test_data.iterrows():
    uid = row['userId']
    mid = row['movieId']
    actual = row['rating']
    
    user_pred = predict_user_based(uid, mid)
    item_pred = predict_item_based(uid, mid)
    
    if not np.isnan(user_pred) and not np.isnan(item_pred):
        actual_ratings.append(actual)
        user_based_preds.append(user_pred)
        item_based_preds.append(item_pred)

# RMSE Calculation
def rmse(preds, actuals):
    return sqrt(mean_squared_error(actuals, preds))

print("User-based CF RMSE:", rmse(user_based_preds, actual_ratings))
print("Item-based CF RMSE:", rmse(item_based_preds, actual_ratings))
