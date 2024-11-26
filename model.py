import joblib
import pandas as pd
import numpy as np

class IRec:
    def __init__(self):
        pass
    def get_recommendations(self, id):
        pass

class ItemBasedRecommender(IRec):
    model = None
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv("data/movies_processed_cleaned.csv")
        self.model = joblib.load('item_based_filtering/models/item_based_collab.pkl')

    def get_recommendations(self, movie_id):
        if self.model is None:
            print('no model')
        inner_id = self.model.trainset.to_inner_iid(int(movie_id))
        neighbor_ids = self.model.get_neighbors(inner_id, k =10)
        ids = [self.model.trainset.to_raw_iid(x) for x in neighbor_ids]
        indexes = self.df.index.isin(ids)
        return indexes

class ContentBasedRecommender(IRec):
    def __init__(self):
        super().__init__()
        self.cosine_sim = np.load("content_based_filtering/nparray/feature_sim_matrix.npy")
        self.df = pd.read_csv("data/movies_processed_cleaned.csv")
        self.indices = pd.Series(self.df.index, index=self.df['id']).drop_duplicates()

    def get_recommendations(self, id: int):
        idx = self.indices[id]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1] , reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        # return self.df[['id']].iloc[movie_indices]
        # !todo: wrong indices are being returned
        return movie_indices