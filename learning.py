import torch
import math


def bce_loss(preds, targets, sum=True):
    losses = -((targets * torch.log(preds)) + ((1 - targets) * torch.log(1 - preds)))
    return losses if not sum else torch.nansum(losses)


def mse_loss(preds, targets, sum=True):
    losses = (preds - targets) ** 2
    return losses if not sum else torch.nansum(losses)


def sigmoid(X):
    return 1 / (1 + torch.exp(-X))


def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))


def predict(X, w, b, activity=None, activation=sigmoid):
    if activity is None:
        activity = 1
    preds = (w @ X.T) + b
    return (preds * (1 - activity)) + (activation(preds) * (activity))


def xavier_init(low, hi):
    limit = math.sqrt(6 / float(low + hi))
    return (-limit - limit) * torch.rand(size=(low, hi)) + limit
