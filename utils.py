import torch
import numpy as np


def ease_out(x):
    return np.sqrt(1 - pow(x - 1, 2))


def ease_in(x):
    return 1 - np.sqrt(1 - pow(x, 2))


def orbit(point, center, theta):
    p1 = torch.sub(point, center)
    p2 = torch.Tensor([[np.cos(theta), np.sin(theta)]])
    p1 = p1[0]
    p2 = p2[0]
    return torch.Tensor([p1[0] * p2[0] - p1[1] * p2[1], p1[0] * p2[1] + p1[1] * p2[0]]).add(center)


def lerp_rgb(rgb1: tuple, rgb2: tuple, frac):
    return tuple(x1 * (1 - frac) + x2 * frac for x1, x2 in zip(rgb1, rgb2))


def interp_rgb(color1, color2, n):
    if n == 1:
        return [lerp_rgb(color1, color2, 0.5)]
    return [lerp_rgb(color1, color2, i / (n - 1)) for i in range(n)]


def str_to_rgb(color):
    return tuple(map(int, color[4:-1].split(", ")))


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_str(color):
    return f"rgb{color}"
