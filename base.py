import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

from themes import Theme


def get_domain(X):
    return torch.Tensor(
        [
            (torch.min(X[:, 0]), torch.max(X[:, 0])),
            (torch.min(X[:, 1]), torch.max(X[:, 1])),
        ]
    )


def pad_domain(domain, padding):
    return torch.Tensor(
        [
            (domain[0][0] - padding, domain[0][1] + padding),
            (domain[1][0] - padding, domain[1][1] + padding),
        ]
    )


def get_domain_surface(domain, resolution=20):
    linspace = torch.stack(
        (
            torch.linspace(domain[0][0], domain[0][1], resolution),
            torch.linspace(domain[1][0], domain[1][1], resolution),
        ),
        dim=1,
    )

    vertices = torch.cartesian_prod(linspace[:, 0], linspace[:, 1])

    return linspace, vertices


def focusable_feature_colors(focused_feature: Union[None, bool], theme: Theme):
    if focused_feature is not None:
        return {
            0: theme.focused_feature_colors[0] if focused_feature == 0 else theme.feature_colors[0],
            1: theme.focused_feature_colors[1] if focused_feature == 1 else theme.feature_colors[1],
        }
    else:
        return theme.feature_colors


@dataclass
class Frame:
    X: torch.Tensor
    targets: torch.Tensor = None
    w: torch.Tensor = None
    b: float = None
    epochs: int = 30
    learning_rate: float = 0.5
    input_size: int = 2
    hidden_size: int = 2
    output_size: int = 1
    resolution: int = 30
    planarity: float = 0
    zrange: Optional[Tuple[float, float]] = (0, 1)
    bias_zrange: Optional[Tuple[float, float]] = (-1, 2)
    domain_padding: float = 2
    range_padding: float = 2.5
    zrange_padding: float = 0.05
    weight_range_padding: Tuple[float, float] = (2.5, 2.5)
    eye: Tuple[float, float, float] = (1, 1, 1)
    weight_eyes: Tuple[Tuple[float, float, float], ...] = ((0, 1, 0), (1, 0, 0))
    bias_eye: Tuple[float, float, float] = (1, 1, 0)
    aspect_ratio: Tuple[float, float, float] = (1, 1, 1)
    preds: Optional[List[torch.Tensor]] = None
    show_preds: Optional[bool] = True
    surface_color: Optional[str] = None
    show_decision_boundaries: Optional[bool] = False
    weights: Optional[Tuple[torch.Tensor, ...]] = None
    biases: Optional[Tuple[torch.Tensor, ...]] = None
    hidden_w: Optional[torch.Tensor] = None
    hidden_b: Optional[torch.Tensor] = None
    inference: Optional[torch.Tensor] = None
    # focused_node: Optional[Tuple[Tuple[Union[bool, int]], ...]] = None
    focused_feature: Optional[int] = None
    focused_inputs: Optional[list] = None
    focus_targets: Optional[bool] = False
    focus_labels: Optional[bool] = None
    focus_total_loss: Optional[bool] = None

    _X: torch.Tensor = field(init=False, repr=False)
    _domain: torch.Tensor = field(init=False, repr=False)
    _surface_points: torch.Tensor = field(init=False, repr=False)
    _surface_line: torch.Tensor = field(init=False, repr=False)

    @property
    def X(self) -> int:
        return self._X

    @property
    def domain(self):
        if self._domain is None:
            self._domain = get_domain(self.X)

        return self._domain

    @property
    def surface_points(self):
        if self._surface_points is None:
            self._surface_line, self._surface_points = get_domain_surface(self.get_domain(pad=True), self.resolution)
        return self._surface_points

    @property
    def surface_linspace(self):
        if self._surface_line is None:
            self._surface_line, self._surface_points = get_domain_surface(self.get_domain(pad=True), self.resolution)
        return self._surface_line

    @X.setter
    def X(self, X: torch.Tensor):
        self._domain = None
        self._surface_points, self._surface_line = None, None
        self._X = X

    def get_eye(self, as_dict=True):
        if not self.eye:
            return None

        if not as_dict:
            return self.eye

        return dict(x=self.eye[0], y=self.eye[1], z=self.eye[2])

    def get_aspect_ratio(self, as_dict=True):
        if not self.aspect_ratio:
            return None

        if not as_dict:
            return self.aspect_ratio

        return dict(x=self.aspect_ratio[0], y=self.aspect_ratio[1], z=self.aspect_ratio[2])

    def get_weight_eyes(self, as_dict=True):
        if not self.weight_eyes:
            return None

        if not as_dict:
            return self.weight_eyes

        return [dict(x=x, y=y, z=z) for x, y, z in self.weight_eyes]

    def get_bias_eye(self, as_dict=True):
        if not self.bias_eye:
            return None

        if not as_dict:
            return self.bias_eye

        return dict(x=self.bias_eye[0], y=self.bias_eye[1], z=self.bias_eye[2])

    def get_domain(self, pad=False, padding=None):
        if not pad:
            return self.domain

        if padding is None:
            padding = self.domain_padding

        return pad_domain(self.domain, padding)

    def get_zrange(self, pad=False, padding=None):
        if not pad:
            return self.zrange

        if padding is None:
            padding = self.zrange_padding

        return self.zrange[0] - padding, self.zrange[1] + padding

    def get_bias_zrange(self, pad=False, padding=None):
        if not pad:
            return self.bias_zrange

        if padding is None:
            padding = self.zrange_padding

        return self.bias_zrange[0] - padding, self.bias_zrange[1] + padding

    def get_range(self, dim=0, pad=False, padding=None):
        if not pad:
            return (self.domain[dim][0], self.domain[dim][1])

        if padding is None:
            padding = self.range_padding

        domain = self.get_domain(pad=True, padding=padding)

        return (domain[dim][0], domain[dim][1])

    def focusable_feature_colors(self, theme: Theme):
        return focusable_feature_colors(self.focused_feature, theme)


@dataclass
class Animation:
    frames: list[Frame]
    output_folder: str
    show_main: bool
    show_components: bool
    show_network: bool
    show_calculations: bool
    height: int
    width: int
    rows: dict
    marker_size: int
    line_width: int
    scale: int
    theme: Theme
