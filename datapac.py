import kagglehub
import pandas as pd
import numpy as np
from sklearn import preprocessing
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
import ast
import re
import json


def parse_production_companies(x):
    if type(x) != str:
        return ""
    pattern = r"[^a-zA-Z0-9\s]"
    try:
        g = ast.literal_eval(x)
        if len(g) <= 0:
            return "[]"

        out = ""
        li = []
        for i, x in enumerate(g):
            name = x['name'].lower()
            name = re.sub(pattern, "", name)
            out += f"{name}"
            li.append(f"{name}")
            if i < len(g) - 1:
                out += ","
        return out
    except:
        return ""


def genre_parser(data):
    g = data.replace("'", "\"")
    g = json.loads(g)
    li = []
    if len(g) > 0:
        out = ""
        for i, x in enumerate(g):
            out += f"{x['name'].lower()}"
            li.append(x['name'].lower())
            if i < len(g) - 1:
                out += ","
        return out
    else:
        return ""


def preprocess_production_companies(df: pd.DataFrame, col_name: str):
    df = df.drop(df.loc[df['production_companies'] == "[]"].index)
    df = df.drop(df.loc[df['production_companies'] == "False"].index)
    df = df.dropna(subset=['production_companies'])
    # df = df.loc[df['production_companies'].isnull() == "[]"]
    return df


def clean_movies_data(df: pd.DataFrame):
    df = df.drop(['production_countries', 'spoken_languages', 'belongs_to_collection', 'budget',
                 'poster_path', 'homepage', 'status', 'video', 'vote_count', 'vote_average'], axis=1)
    df = preprocess_production_companies(df, 'production_companies')
    df['production_companies'] = df['production_companies'].apply(
        parse_production_companies)
    df['genres'] = df['genres'].apply(genre_parser)
    df['overview'] = df['overview'].fillna('')
    df['title'] = df['title'].str.lower()
    df['id'] = df['id'].astype(int)
    df['movieId'] = df['id'].astype(int)
    return df


def load_movies():
    path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
    df = pd.read_csv(f"{path}/movies_metadata.csv", low_memory=False)
    df = clean_movies_data(df)
    return df


def load_ratings():
    path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
    ratings = pd.read_csv(f"{path}/ratings.csv")
    ratings = ratings.drop(labels=['timestamp'], axis=1)
    return ratings


def clean_ratings(ratings: pd.DataFrame):
    movies = load_movies()

    df = pd.merge(movies, ratings, left_on="id", right_on="movieId", how="right")
    df = df.dropna()
    df = df[['userId', 'id', 'rating']]
    df['movieId'] = df['id']
    df = df.drop(labels=['id'], axis=1)

    movies['id'][movies['id'].isin(ratings['movieId'])].value_counts()
    # if save:
    #     df.to_csv("data/item_ratings.csv")
    return df

def get_user_item_dataset(ratings: pd.DataFrame):
    ratings = normalize_ratings(ratings)
    ratings_dict = {
        "item": ratings['movieId'],
        "user": ratings['userId'],
        "rating": ratings['rating'],
    }
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    return data


def get_train_test_collab(ratings: pd.DataFrame, test_size=0.25):
    ratings_dict = {
        "item": ratings['movieId'],
        "user": ratings['userId'],
        "rating": ratings['rating'],
    }

    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=test_size)
    return trainset, testset


def normalize_ratings(ratings: pd.DataFrame):
    minmax = preprocessing.MinMaxScaler()
    ratings[['rating']] = minmax.fit_transform(ratings[['rating']])
    return ratings


# if __name__ == "__main__":
#     df = load_ratings()
#     train, test = get_train_test_set(df)
#     print(train)
