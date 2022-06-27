import os
import random

import numpy as np
import pandas as pd
import torch


def load_data_from_csv():
    """
    This method loads stock data from thousands of stocks.
    :return: A dictionary contains all stocks' data.
    """
    data_set = {}
    validPoints = 0
    totalPoints = 0
    print("Detected stocks: " + str(len(os.listdir("./data/stocks"))))
    for fileName in os.listdir("./data/stocks"):
        data_set[fileName] = pd.read_csv("./data/stocks/" + fileName,
                                         sep=",",
                                         header=0,
                                         usecols=range(3, 22),
                                         encoding="gbk",
                                         dtype="float64"
                                         ).dropna()
        totalPoints += data_set[fileName].shape[0]
        if data_set[fileName].empty:
            # print("Warning: problematic stock dataset " + fileName)
            del data_set[fileName]
        else:
            validPoints += data_set[fileName].shape[0]
    print("Total data points: " + str(totalPoints))
    print("Valid data points: " + str(validPoints))
    print("Valid stocks: " + str(len(data_set.keys())))

    return data_set


def load_train_and_test():
    """
    This method merge all stocks data together, and count
    "Quote change" as y, and take the five-day averages of the other
    data which is from six days before to one day before the y as x
    :return: Tensors of x and y
    """

    data_set = load_data_from_csv()
    x = []
    y = []

    for stock in data_set.values():
        if stock.shape[0] <= 6:
            continue
        for rowId in range(6, stock.shape[0]):
            x.append(torch.tensor(stock.iloc[rowId - 6:rowId - 1, :].mean(axis=0).values))
            y.append(torch.tensor(stock.iloc[rowId, 1]))

    x = torch.stack(x).float()
    y = torch.stack(y).reshape(-1, 1).float()

    return x, y