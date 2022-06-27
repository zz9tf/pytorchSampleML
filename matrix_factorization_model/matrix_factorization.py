import torch
from random import sample
import numpy as np
import matplotlib.pyplot as plt


class mf:

    def __init__(self, users, movies, ratings, config):
        """

        :param users:
        :param movies:
        :param ratings:
        :param config:
        """
        # Hyperparameters
        self.sampling_rate = config["sampling_rate"]
        self.learning_rate = config["learning_rate"]
        self.batch = config["batch"]
        self.sampling_num = int(len(ratings) * self.sampling_rate)

        # datasets
        if config["select_data"]:
            users = users.iloc[:config["data_size"], :]
            movies = movies.iloc[:config["data_size"], :]
            ratings = ratings.iloc[:config["data_size"], :]
        self.train_u, self.train_m, self.train_r, \
        self.test_u, self.test_m, self.test_r = self.split_datasets(users, movies, ratings)

        # Initialize fields in mf_model
        self.w_user = torch.rand((4, 4), requires_grad=True)
        self.w_movies = torch.rand((18, 4), requires_grad=True)
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []

        # Initialize layers
        self.layer = torch.nn.Linear
        # self.a = lambda x: torch.min(torch.max(x, torch.tensor(0)), torch.tensor(5))

        self.loss_function = torch.nn.MSELoss()

    def split_in_batch(self, users, movies, ratings):
        """
        Split data as lists of tensors in batch size.
        :param users: A dataframe
        :param movies: A dataframe
        :return: lists of tensors
        """
        users = torch.from_numpy(users.values).float()
        movies = torch.tensor(list(movies["genre"])).float()
        ratings = torch.tensor(ratings.values).float()
        u = []
        m = []
        r = []
        for i in range(0, users.shape[0], self.batch):
            u.append(users[i: i + self.batch])
            m.append(movies[i: i + self.batch])
            r.append(ratings[i: i + self.batch])
        return u, m, r

    def split_datasets(self, users, movies, ratings):
        """
        Split data into train sets and test sets
        :param users: A dataframe of users data
        :param movies: A dataframe of movies data
        :param ratings: A dataframe of ratings data
        :return: lists of train sets and test sets
        """
        train_id = sample(list(range(len(ratings))), int(self.sampling_rate * len(ratings)))
        train_r = ratings.iloc[train_id]
        train_u = users.iloc[list(train_r["userId"])]
        train_m = movies.iloc[list(train_r["itemId"])]
        train_r = train_r["rating"]
        train_u, train_m, train_r = self.split_in_batch(train_u, train_m, train_r)

        test_id = list(set(range(len(ratings))) - set(train_id))
        test_r = ratings.iloc[test_id]
        test_u = users.iloc[list(test_r["userId"])]
        test_m = movies.iloc[list(test_r["itemId"])]
        test_r = test_r["rating"]
        test_u, test_m, test_r = self.split_in_batch(test_u, test_m, test_r)

        return train_u, train_m, train_r, test_u, test_m, test_r

    def forward(self, u, m):
        """
        This method predicts a batch ratings based on the given u and m.
        :param u: A tensor represents a botch of users data
        :param m: A tensor represents a botch of movies data
        :return: A tensor represents a batch of predicted ratings.
        """
        latentUsers = torch.mm(u, self.w_user)
        latentMovies = torch.mm(m, self.w_movies)

        # latentUsers dot latentMovies
        r = [torch.dot(latentUsers[i], latentMovies[i]).reshape(1) for i in range(latentUsers.size()[0])]
        r = torch.cat(r, 0)
        # r = self.a(r)

        return r

    def backward(self, pred_r, real_r):
        """
        This method passes the loss gradient back
        :param pred_r: A tensor represents the predicted ratings
        :param real_r: A tensor represents the real ratings
        """
        loss = self.loss_function(pred_r, real_r)
        loss.backward()

    def train_an_epoch(self, train_users, train_movies, train_ratings):
        """
        This method train model for an epoch.
        :param train_users: A list of users tensors
        :param train_movies: A list of movies tensors
        :param train_ratings: A list of ratings tensors
        """
        loss = []
        # i = 1
        for u, m, r in zip(train_users, train_movies, train_ratings):
            # print("    data: " + str(i * self.batch) + "/" + str(self.sampling_num))
            predict_r = self.forward(u, m)
            loss.append(self.backward(predict_r, r))

            with torch.no_grad():
                self.w_user -= self.w_user.grad * self.learning_rate
                self.w_movies -= self.w_movies.grad * self.learning_rate
            # i += 1

    def preform(self, users, movies, ratings):
        """
        This method calculates the loss and accuracy of this model based on given data
        :return: A tuple represents the the loss and accuracy
        """
        acc = []
        loss = []
        for u, m, r in zip(users, movies, ratings):
            pred_r = torch.round(self.forward(u, m))
            loss.append(float(self.loss_function(pred_r, r)))
            acc.append(float(torch.eq(pred_r, r).sum()) / len(pred_r))
        return sum(loss)/len(loss), sum(acc) / len(acc)

    def train(self, num=50):
        """
        This method train the model with a specific number of epochs.
        :param num: A int represent the number of training epochs
        :return: None
        """
        for epoch in range(num):
            print("Epoch " + str(epoch))
            self.train_an_epoch(self.train_u, self.train_m, self.train_r)

            preform = self.preform(self.train_u, self.train_m, self.train_r)
            self.train_loss.append(preform[0])
            self.train_acc.append(preform[1])
            preform = self.preform(self.test_u, self.test_m, self.test_r)
            self.test_loss.append(preform[0])
            self.test_acc.append(preform[1])
            print("    train_loss: {:f}, train_acc: {:f}, test_loss: {:f}, test_acc: {:f}".format(
                self.train_loss[-1],
                self.train_acc[-1],
                self.test_loss[-1],
                self.test_acc[-1]))

        fig, axs = plt.subplots(2, 2, figsize=(9,6))
        axs[0, 0].plot(self.train_loss)
        axs[0, 0].set_ylabel("Train loss")
        axs[0, 0].set_xlabel("epoch")
        axs[0, 1].plot(self.train_acc)
        axs[0, 1].set_ylabel("Train acc")
        axs[0, 1].set_xlabel("epoch")
        axs[1, 0].plot(self.test_loss)
        axs[1, 0].set_ylabel("Test loss")
        axs[1, 0].set_xlabel("epoch")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_ylabel("Test acc")
        axs[1, 1].set_xlabel("epoch")
        plt.show()

    def load_w(self):
        self.w_user = torch.load("w_user.pt")
        self.w_movies = torch.load("w_movies.pt")
