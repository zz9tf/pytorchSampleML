import torch

def hessian_matrix(theta, y):
    """
    This method calculated the hessian_matrix of model.
    :param theta: A <1, n> tensor
    :param y: An scalar
    :return: A matrix represents the hessian matrix.
    """
    first_derivative = torch.autograd.grad(y**1, theta, create_graph=True)[0][0]
    second_derivative = torch.stack([torch.autograd.grad(element, theta, create_graph=True)[0][0]
                                     for element in first_derivative]
                                    , dim=0)
    return second_derivative.clone().detach()

def inf_params(theta, y, y_out):
    """
    This method calculates the changed of parameters in model when remove a special train sample.
    :param theta: A tensor represents the parameters in model.
    :param y: A tensor represents the loss value with all train data.
    :param y_out: A tensor represents the loss value with removed train data.
    :param n: A integer represents the number of train sample in the model.
    :return: A tensor represents the predicted change of parameters in model.
    """
    hessian = hessian_matrix(theta, y)
    first_derivative = torch.autograd.grad(y_out**1, theta, create_graph=True)[0]
    return -torch.mm(torch.linalg.inv(hessian), first_derivative.T)


def inf_loss(theta, y, y_out, y_test):
    """
    This method calculates the predicted loss change of a special test sample.
    :param theta: A tensor represents the parameters in model.
    :param y: A tensor represents the loss value with all train data.
    :param y_out: A tensor represents the loss value with removed train data.
    :param y_test: A tensor represents the loss value with a special test sample.
    :param n: A integer represents the number of train sample in the model.
    :return: A tensor represents the predicted change of parameters in model.
    """
    inf_param = inf_params(theta, y, y_out)
    test_fir_d = torch.autograd.grad(y_test**1, theta, create_graph=True)[0]
    return torch.mm(test_fir_d, inf_param).clone().detach()
