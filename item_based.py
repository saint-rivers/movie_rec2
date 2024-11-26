import surprise
from surprise import KNNWithMeans
from surprise.model_selection import cross_validate
from surprise import accuracy
import pickle
import numpy as np
import datapac
import json
import sys
import joblib


def calculate_metrics(model, testset: surprise.Trainset) :
    predictions = model.test(testset)
    print(testset[0])
    print(predictions[0])

    rmse = accuracy.rmse(predictions)  # Root Mean Squared Error
    mae = accuracy.mae(predictions)  # Mean Absolute Error

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    filename = "metrics/item_based.txt"
    with open(filename, "w+") as file:
        file.write(json.dumps({'mae': mae, 'rmse': rmse}))
    print(f"saved metrics to {filename}")


def knn_train(trainset):
    sim_options = {
        "name": "pearson_baseline",
        "user_based": False,
    }
    model = KNNWithMeans(sim_options=sim_options)
    # measures = cross_validate(model, trainset, measures=['RMSE', 'MAE'],
    #                           cv=2, verbose=True)
    model.fit(trainset)
    return model


def train_and_save():
    df = datapac.load_ratings()
    print("### Cleaning Ratings Data")
    df = datapac.clean_ratings(df)
    print(df)

    print("### Creating Similarity Matrix")
    # trainset = datapac.get_user_item_dataset(df)
    trainset, testset = datapac.get_train_test_collab(df)
    model = knn_train(trainset)

    # save model and similarity matrix
    with open('model/item_based_filtering.pkl', 'wb') as f:
        pickle.dump(model, f)
    np.save("matrices/item_sim_matrix.npy", model.sim)

    calculate_metrics(model, testset)

    print("### done training ###")


def load_from_saved():
    df = datapac.load_ratings()
    print("### Cleaning Ratings Data")
    df = datapac.clean_ratings(df)
    trainset, testset = datapac.get_train_test_collab(df)

    model = joblib.load('model/item_based_filtering.pkl')
    calculate_metrics(model, testset)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        train_and_save()
    elif mode == "load":
        load_from_saved()
