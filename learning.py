import torch


def loss(preds, targets):
    return -((targets * torch.log(preds)) + ((1 - targets) * torch.log(1 - preds)))


def sigmoid(X):
    return 1 / (1 + torch.exp(-X))


def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))


def predict(X, w, b, planarity=0, activation=sigmoid):
    preds = (w @ X.T) + b
    return (preds * planarity) + (activation(preds) * (1 - planarity))
