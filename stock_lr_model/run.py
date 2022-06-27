import numpy as np
import torch, pandas
import os
from linear_regression import torch_lr, manual_lr
from load_data import load_train_and_test
import inf_function
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Hyperparameters
configs = {
    "sampling_rate": 0.7, # The percentage of samples are training sample.
    "learning_rate": 1e-2,
    "change_rate": None, # The minimal loss change rate after one epoch(None means not using it)
    "train_epoch": 10000,
    "show": False # show details in the model training
}

x, y = load_train_and_test()
x = x[:, 8:13]
y = y[:, :]



# Train model
torch.manual_seed(0)
theta = torch.rand(1, x.shape[1] + 1)
# torch.save(man_lr.theta.clone().detach(), "man_lr_theta.pt")

# Load parameters
man_lr = manual_lr(x, y, configs, theta)
man_lr.train()

dif = pandas.DataFrame(columns=["pred_diff", "real_diff"])

loss = man_lr.loss_func(man_lr.predict(man_lr.train_x), man_lr.train_y)
loss_test = man_lr.loss_func(man_lr.predict(man_lr.test_x[0, :]), man_lr.test_y[0, :])
for train_out_id in range(man_lr.train_x.shape[0]):
    print(train_out_id, "/", man_lr.train_x.shape[0])

    loss_out = man_lr.loss_func(man_lr.predict(man_lr.train_x[train_out_id, :]), man_lr.train_y[train_out_id, :])
    pred_test = float(-inf_function.inf_loss(man_lr.theta
                                                      , loss
                                                      , loss_out
                                                      , loss_test)/man_lr.train_x.shape[0])
    new_man_lr = manual_lr(x, y, configs, theta)
    new_man_lr.train_x = torch.cat(
        (man_lr.train_x[:train_out_id, :], man_lr.train_x[train_out_id+1:, :])
        , dim=0)
    new_man_lr.train_y = torch.cat(
        (man_lr.train_y[:train_out_id, :], man_lr.train_y[train_out_id+1:, :])
        , dim=0)
    new_man_lr.train()
    new_loss = new_man_lr.loss_func(
        new_man_lr.predict(new_man_lr.test_x[0, :])
        , new_man_lr.test_y[0, :])
    real_test = float(new_loss - loss_test)
    dif.loc[len(dif.index)] = [pred_test, real_test]

dif = dif[dif["real_diff"] < 0.0004]
helpful_dif = dif[dif["real_diff"] >= 0]
harmful_dif = dif[dif["real_diff"] < 0]
plt.scatter(helpful_dif["real_diff"], helpful_dif["pred_diff"], color="blue")
plt.scatter(harmful_dif["real_diff"], harmful_dif["pred_diff"], color="green")
min = min(min(dif["real_diff"]), min(dif["pred_diff"]))
max = max(max(dif["real_diff"]), max(dif["pred_diff"]))
plt.plot([min, max], [min, max], color="red")
print(pearsonr(dif["real_diff"], dif["pred_diff"]))
plt.show()










