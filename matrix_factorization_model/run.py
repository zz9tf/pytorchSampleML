import torch

from data import load_data, normalize_data
from matrix_factorization import mf

# Hyperparameters
config = {
    "select_data": False, # shrink the dataset size.
    "data_size": 1024,
    "sampling_rate": 0.7, # the rate of training set size of the whole data.
    "learning_rate": 1e-7,
    "batch": 1024,
    "train epochs": 10}

# load datasets
users, movies, ratings = load_data()
normalize_data(users, movies)

# model
model = mf(users, movies, ratings, config)
model.train(config["train epochs"])


# torch.save(self.w_user, "wUser.pt")
# torch.save(self.w_movies, "wMovies.pt")


