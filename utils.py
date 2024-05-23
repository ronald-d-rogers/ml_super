import torch
import numpy as np

from themes import Theme


def loss(preds, targets):
    return -((targets * torch.log(preds)) + ((1 - targets) * torch.log(1 - preds)))


def predict(X, w, b, planarity):
    preds = (X @ w.T) + b
    return (preds * planarity) + (torch.sigmoid(preds) * (1 - planarity))


def ease_out(x):
    return np.sqrt(1 - pow(x - 1, 2))


def ease_in(x):
    return 1 - np.sqrt(1 - pow(x, 2))


def orbit(point, center, theta):
    p1 = torch.sub(point, center)
    p2 = torch.Tensor([np.cos(theta), np.sin(theta)])
    return torch.Tensor([p1[0] * p2[0] - p1[1] * p2[1], p1[0] * p2[1] + p1[1] * p2[0]]).add(center)


def get_domain(X):
    return torch.Tensor(
        [
            (torch.min(X[:, 0] - 2), torch.max(X[:, 0]) + 2),
            (torch.min(X[:, 1]) - 2, torch.max(X[:, 1]) + 2),
        ]
    )


def get_domain_vertices(domain, resolution=20):
    linspace = torch.stack(
        (
            torch.linspace(domain[0][0], domain[0][1], resolution),
            torch.linspace(domain[1][0], domain[1][1], resolution),
        ),
        dim=1,
    )

    vertices = torch.cartesian_prod(linspace[:, 0], linspace[:, 1])

    return linspace, vertices


def focusable_feature_colors(focused_feature, theme: Theme):
    if focused_feature is None:
        return theme.feature_colors

    if focused_feature == 0:
        return [theme.focused_feature_colors[0], theme.feature_colors[1]]
    else:
        return [theme.feature_colors[0], theme.focused_feature_colors[1]]
