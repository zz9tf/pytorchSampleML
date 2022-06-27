import torch
import matplotlib.pyplot as plt

class torch_lr(torch.nn.Module):
    def __init__(self, x, y, configs, weight=None, bias=None):
        """
        This class is a torch-based linear regression.
        :param x: A tensor represents the inputs
        :param y: A tensor represents the outputs
        :param configs: A dictionary of hyperparameters
        :param weight: The weights of linear model (default: None).
        :param bias: The bias  of linear model (default: None).
        """
        super(torch_lr, self).__init__()
        self.configs = configs
        x = x / x.max(dim=0).values
        self.train_x = x[:int(self.configs["sampling_rate"] * x.shape[0]), :]
        self.train_y = y[:int(self.configs["sampling_rate"] * y.shape[0]), :]
        self.test_x = x[int(self.configs["sampling_rate"] * x.shape[0]):, :]
        self.test_y = y[int(self.configs["sampling_rate"] * y.shape[0]):, :]

        self.linear = torch.nn.Linear(x.shape[1], 1)

        self.optimizer = torch.optim.SGD(self.parameters(),
                                         lr=self.configs["learning_rate"])
        if weight != None:
            with torch.no_grad():
                self.linear.weight.copy_(weight)

        if bias != None:
            with torch.no_grad():
                self.linear.bias.copy_(bias)

        self.criterion = torch.nn.MSELoss()
        self.train_loss = []
        self.test_loss = []

    def forward(self, x):
        """
        This method calculates y_hat based on provided x.
        :param x: A tensor represents the input
        :return: A tensor represents the output which is a float value.
        """
        return self.linear(x)

    def train_an_epoch(self):
        """
        This method updates the parameters for an epoch
        :return: None
        """
        self.optimizer.zero_grad()
        pred_y = self(self.train_x)
        loss = self.criterion(pred_y, self.train_y)
        loss.backward()
        self.optimizer.step()
        self.train_loss.append(float(loss))
        self.test_loss.append(float(self.criterion(self(self.test_x),self.test_y)))

    def train(self):
        """
        This method trains the model
        :param epoch_num: A integer represents the epoch to train.
        :return: None
        """
        for epoch in range(self.configs["train_epoch"]):
            self.train_an_epoch()
            print("Epoch {}, train loss {}, test loss {}".format(epoch, self.train_loss[-1], self.test_loss[-1]))
            if self.configs["change_rate"] is not None and len(self.train_loss) > 1:
                if abs(self.train_loss[-2]-self.train_loss[-1])/self.train_loss[-2] < self.configs["change_rate"]:
                    break
        plt.plot(self.train_loss, color="red")
        plt.plot(self.test_loss, color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

class manual_lr():

    def __init__(self, x, y, configs, theta=None):
        """
        This class is a manual linear regression.
        :param x: A tensor represents the inputs
        :param y: A tensor represents the outputs
        :param configs: A dictionary of hyperparameters
        """
        self.configs = configs
        x = x / x.max(dim=0).values
        self.train_x = x[:int(self.configs["sampling_rate"] * x.shape[0]), :]
        self.train_y = y[:int(self.configs["sampling_rate"] * y.shape[0]), :]
        self.test_x = x[int(self.configs["sampling_rate"] * x.shape[0]):, :]
        self.test_y = y[int(self.configs["sampling_rate"] * y.shape[0]):, :]
        self.theta = torch.rand(1, x.shape[1]+1)

        if theta != None:
            self.theta.copy_(theta)
        self.theta.requires_grad = True

        self.learning_rate = self.configs["learning_rate"]

        self.train_loss = []
        self.test_loss = []

    def loss_func(self, pred_y, real_y):
        """
        This method calculates the loss by MSE.
        :param pred_y: The y is predicted by model.
        :param real_y: Real y.
        :return: A tensor represents the MSE loss of this prediction.
        """
        return ((pred_y - real_y) ** 2).mean(dim=0)

    def predict(self, x):
        """
        This method calculates y_hat based on provided x.
        :param x: A tensor(a vector or a matrix) represents the input
        :return: A tensor represents the output which is a float value.
        """
        if len(x.shape) == 2:
            return (torch.mm(self.theta[:, :-1], x.T) + self.theta[:, -1]).T
        elif len(x.shape) == 1:
            return torch.matmul(self.theta[:, :-1], x.T) + self.theta[:, -1]
        raise ValueError("The tensor should be a vector or a matrix")

    def train_an_epoch(self):
        """
        This method updates the parameters for an epoch
        :return: None
        """
        loss = self.loss_func(self.predict(self.train_x), self.train_y)
        loss.backward()
        with torch.no_grad():
            self.theta.copy_(self.theta - self.learning_rate * self.theta.grad)
            self.theta.grad.zero_()

        self.train_loss.append(float(loss))
        self.test_loss.append(float(self.loss_func(self.predict(self.test_x), self.test_y)))

    def train(self):
        """
        This method trains the model
        :param epoch_num: A integer represents the epoch to train.
        :return: None
        """
        for epoch in range(self.configs["train_epoch"]):
            self.train_an_epoch()
            if self.configs["show"]:
                print("Epoch {}, train loss {}, test loss {}".format(epoch, self.train_loss[-1], self.test_loss[-1]))
                if self.configs["change_rate"] is not None and len(self.train_loss) > 1:
                    if abs(self.train_loss[-2]-self.train_loss[-1])/self.train_loss[-2] < self.configs["change_rate"]:
                        break
        if self.configs["show"]:
            plt.plot(self.train_loss, color="red")
            plt.plot(self.test_loss, color="blue")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()