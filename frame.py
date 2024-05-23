import torch
from dataclasses import dataclass
from typing import Union


@dataclass
class Frame:
    X: torch.Tensor
    preds: torch.Tensor
    targets: torch.Tensor
    w: torch.Tensor
    b: float
    domain: tuple
    epochs: int
    learning_rate: float
    surface_points: torch.Tensor
    surface_line: torch.Tensor
    inference: Union[None, torch.Tensor]
    planarity: float
    focused: Union[None, list]
    focused_feature: Union[None, int]
    focus_labels: Union[None, bool]
    focus_total_loss: Union[None, bool]
    focus_targets: bool
    show_preds: bool
    show_surface: bool
    show_decision_boundary: bool
    eye: tuple
    aspect_ratio: tuple
    zrange: tuple
    resolution: int
