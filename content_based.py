import datapac
import json
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import json


def create_soup(x: pd.Series):
    def join_feat(feat):
        if feat is None:
            parsed = ""
        else:
            v = str(feat).split(",")
            parsed = ' '.join(v)
        return parsed

    out = join_feat(x['genres'])
    out = out + ' ' + join_feat(x['production_companies'])
    out = out + ' ' + join_feat(x['title'])
    return out


def generate_matrix(df: pd.DataFrame, save=False):
    df = df[['production_companies', 'genres', 'title', 'id']]
    df['soup'] = df.apply(create_soup, axis=1)
    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(df['soup'])
    cos_sim = cosine_similarity(count_matrix, count_matrix)
    if save:
        np.save("./matrices/feature_sim_matrix.npy", cos_sim)
    return


def predict_rating(user_id:int, target_id:int, similarity_matrix, indices, ratings):
    rated_items = ratings.loc[ratings['userId'] == user_id][['movieId', 'rating']]
    
    weighted_ratings = 0
    similarity_sum = 0

    for rated in rated_items.iterrows():
        seen_mov_id = int(rated[1]['movieId'])
        seen_idx = indices.loc[indices['id']==seen_mov_id].index[0]
        target_idx = indices.loc[indices['id']==target_id].index[0]

        sim = similarity_matrix[seen_idx][target_idx]
        weighted_ratings += sim * rated[1]['rating']
        similarity_sum += sim

    if similarity_sum == 0:
        return np.nan  # No similar items with ratings
    return weighted_ratings / similarity_sum


def calculate_scores(sim_mat, indices, ratings):
    def test(row):
        out = predict_rating(int(row['userId']), int(row['movieId']), sim_mat, indices)
        # print({'predicted': out, 'actual': row['rating']})
        return pd.Series({'userId': row['userId'], 'movieId': row['movieId'], 'predicted': out, 'actual': row['rating']})
    predictions = ratings[:10000].apply(test, axis=1)
    predictions.to_csv("metrics/content_based_predictions.csv")

    rmse = root_mean_squared_error(predictions['actual'].values, predictions['predicted'].values)
    mae = mean_absolute_error(predictions['actual'].values, predictions['predicted'].values)

    with open("metrics/content_based.txt", "w+") as file:
        file.write(json.dumps({'rmse': rmse, 'mae': mae}))


def get_recommendations(movie_id: int, cosine_sim):
    sim_scores = list(enumerate(cosine_sim[movie_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    # sim = [i[1] for i in sim_scores]
    return movie_indices, sim_scores


def train_and_save():
    movies = datapac.load_movies()
    generate_matrix(movies, save=True)

    indices = pd.read_csv("data/movies_processed_cleaned.csv")[['id', 'title']].drop_duplicates()
    feat_sim_mat = np.load("matrices/feature_sim_matrix.npy")
    test_ratings = pd.read_csv("data/test_ratings.csv")
    calculate_scores(sim_mat=feat_sim_mat, indices=indices, ratings=test_ratings)

    print("### done training ###")


def load_from_saved():
    movies = pd.read_csv("data/movies_processed_cleaned.csv")
    feat_sim_mat = np.load("matrices/feature_sim_matrix.npy")
    indices = pd.DataFrame(movies[['id', 'title']]).drop_duplicates()
    test_ratings = pd.read_csv("data/test_ratings.csv")
    calculate_scores(sim_mat=feat_sim_mat, indices=indices, ratings=test_ratings)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        train_and_save()
    elif mode == "load":
        load_from_saved()

