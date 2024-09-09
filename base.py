import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Union

from scenes.base import Scene
from themes import Theme, default_theme
from utils import clone, hex_to_rgb, interp_rgb, rgb_to_str


def parse_node_module(view_str: str):
    return view_str.split("_")[0]


def parse_node_index(view_str: str):
    return int(view_str.split("_")[1]) - 1


def get_domain(X):
    return torch.Tensor(
        [
            (torch.min(X[:, 0]), torch.max(X[:, 0])),
            (torch.min(X[:, 1]), torch.max(X[:, 1])),
        ]
    )


def get_padded_domain(domain, padding):
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


def focusable_feature_colors(focused_feature: Union[None, bool], colors: List[str], focused_colors: List[str]):
    # if focused_feature is not None:
    #     return {
    #         0: theme.focused_feature_colors[0] if focused_feature == 0 else theme.feature_colors[0],
    #         1: theme.focused_feature_colors[1] if focused_feature == 1 else theme.feature_colors[1],
    #     }
    # else:
    #     return theme.feature_colors
    if focused_feature is None:
        return colors
    else:
        return [focused_colors[focused_feature] if i == focused_feature else colors[i] for i in range(len(colors))]


def get_weight_eyes(weight_eyes, as_dict=False):
    if not weight_eyes:
        return None

    if not as_dict:
        return weight_eyes

    return [dict(x=x, y=y, z=z) for x, y, z in weight_eyes]


def interp_colors(color1, color2, n):
    colors = interp_rgb(hex_to_rgb(color1), hex_to_rgb(color2), n)
    return [rgb_to_str(c) for c in colors]


@dataclass
class NodeView:
    w: Dict[str, torch.Tensor]
    b: Dict[str, torch.Tensor]
    modules: List[str]
    targets: torch.Tensor
    preds: torch.Tensor = None
    derivatives: torch.Tensor = None
    errors: torch.Tensor = None
    losses: torch.Tensor = None
    costs: torch.Tensor = None
    focused_preds: List[int] = None
    focused_errors: List[int] = None
    activity: float = 1
    weight_eyes: Tuple[Tuple[float, float, float], ...] = None
    bias_eye: Tuple[float, float, float] = (1, 1, 0)

    def get_weight_eyes(self, as_dict=False):
        return get_weight_eyes(self.weight_eyes, as_dict=as_dict)


@dataclass
class Animation:
    frames: list["Frame"]
    model_node: str = "output_1"
    show_model: bool = True
    show_network: bool = False
    show_calculations: bool = False
    height: int = 1920
    width: int = 1080
    scale: int = 2
    show_bg: bool = True
    render_path: str = None
    show_parameters: bool = True
    show_gradients: bool = True
    show_weights_preds: bool = False
    show_label_names: bool = True
    theme: Theme = default_theme
    cells: dict = field(default_factory=dict)
    scene: List[Scene] = field(default_factory=list)

    _node_module = None
    _node_index = None

    @property
    def node_module(self):
        if self._node_module is None:
            self._node_module = parse_node_module(self.model_node)
        return self._node_module

    @property
    def node_index(self):
        if self._node_index is None:
            self._node_index = parse_node_index(self.model_node)
        return self._node_index

    def colors(self, size: int):
        values = self.theme.feature_colors
        return interp_colors(values[0], values[1], size)

    def focused_colors(self, size):
        values = self.theme.focused_feature_colors
        module_size = size
        return interp_colors(values[0], values[1], module_size)

    def focusable_colors(self, focused: int, size: int):
        return focusable_feature_colors(focused, self.colors(size), self.focused_colors(size))

    def node_view(
        self,
        frame: "Frame",
        component: Union[int, str] = None,
        weight_eyes: Tuple[Tuple[float, float, float], ...] = None,
        activity: float = None,
    ):
        module = self.node_module
        index = self.node_index

        def get_node_data(data: dict, module, index):
            if data and data.get(module) is not None and index < len(data[module]):
                return data[module][index]

        neuron_weight_eyes = ((0, -1, 0), (1, 0, 0))

        modules = frame.modules

        input = modules[0]
        hidden = modules[1]
        output = modules[-1]

        if len(modules) > 3:
            raise ValueError(
                f'Unsupported number of modules "{len(modules)}". The only currently supported number of modules is 3.'
            )

        frame_w: torch.Tensor = clone(frame.w)
        frame_b: torch.Tensor = clone(frame.b)

        w = {}
        b = {}
        if module == output:
            if output not in frame.w:
                w = {output: torch.zeros(1, frame.size[hidden])}
                b = {output: torch.zeros(1, 1)}
                modules = [input, output]
            else:
                w[output] = frame.w[output][index].unsqueeze(0)
                b[output] = frame.b[output][index].unsqueeze(0)
                weight_eyes = neuron_weight_eyes

                modules = [input, output]

                if not hidden == output and hidden in frame.w:
                    w[hidden] = frame_w[hidden]
                    b[hidden] = frame_b[hidden]
                    weight_eyes = frame.get_weight_eyes()

                    modules = frame.modules.copy()

        elif not hidden == output and module == hidden:
            if hidden not in frame.w:
                w = {output: torch.zeros(1, frame.size[input])}
                b = {output: torch.zeros(1, 1)}
            else:
                w = {output: frame_w[hidden][index].unsqueeze(0)}
                b = {output: frame_b[hidden][index].unsqueeze(0)}
            weight_eyes = neuron_weight_eyes

            modules = [input, output]

        if component is not None:
            if isinstance(component, int):
                component_index = component - 1

                # if we're viewing a node with hidden nodes
                if not hidden == output and hidden in w:
                    mask = [0 for _ in range(frame.size[hidden])]
                    mask[component_index] = 1
                    w[hidden] = torch.Tensor([mask]).T * w[hidden]
                    b[hidden] = torch.Tensor([mask]).T * b[hidden]
                    activity = activity if activity is not None else 1

                else:
                    mask = [0 for _ in range(frame.size[input])]
                    mask[component_index] = 1
                    w[output] = torch.Tensor([mask]) * w[output]
                    b[output] = torch.Tensor([[0]])
                    activity = activity if activity is not None else 0

            elif component == "b":
                w[output] = torch.Tensor([[0, 0]])
                activity = activity if activity is not None else 0

                if not hidden == output and hidden in w:
                    del w[hidden]

                modules = [input, output]

            else:
                raise ValueError(
                    f'Unsupported component "{component}". You may select a component from the following: "b", or an integer index of the compoennt.'
                )

        return NodeView(
            w=w,
            b=b,
            targets=frame.targets[0],
            modules=modules,
            preds=get_node_data(frame.preds, module, index),
            errors=get_node_data(frame.errors, module, index),
            losses=get_node_data(frame.losses, module, index),
            costs=get_node_data(frame.costs, module, index),
            focused_preds=get_node_data(frame.focused_preds, module, index),
            focused_errors=get_node_data(frame.focused_errors, module, index),
            weight_eyes=frame.weight_eyes if weight_eyes is None else weight_eyes,
            activity=frame.activity if activity is None else activity,
        )


default_eye = (1, 1, 1)


@dataclass
class Frame:
    X: torch.Tensor
    targets: torch.Tensor = None
    preds: Dict[str, torch.Tensor] = None
    derivatives: Dict[str, torch.Tensor] = None
    errors: Dict[str, torch.Tensor] = None
    losses: Dict[str, torch.Tensor] = None
    costs: Dict[str, torch.Tensor] = None
    loss: torch.Tensor = None
    w: Dict[str, torch.Tensor] = None
    b: Dict[str, torch.Tensor] = None
    loss_fn: callable = None
    activation_fns: Dict[str, callable] = None
    size: Dict[str, int] = field(default_factory=lambda: {"input": 2, "hidden": 2, "output": 1})
    modules: List[str] = field(default_factory=lambda: ["input", "hidden", "output"])
    epochs: int = 30
    learning_rate: float = 0.5
    resolution: int = 30
    activity: float = 1
    zrange: Optional[Tuple[float, float]] = (0, 1)
    bias_zrange: Optional[Tuple[float, float]] = (-1, 2)
    domain_padding: float = 2
    range_padding: float = 2.5
    zrange_padding: float = 0.05
    eye: Tuple[float, float, float] = default_eye
    weight_eyes: Tuple[Tuple[float, float, float], ...] = None
    bias_eye: Tuple[float, float, float] = (1, 1, 0)
    aspect_ratio: Tuple[float, float, float] = (1, 1, 1)
    show_preds: bool = True
    show_profile: bool = False
    show_decision_boundaries: bool = False
    inference: torch.Tensor = None
    active_preds: Dict[str, List[List[int]]] = field(default_factory=lambda: {"hidden": [], "output": []})
    active_errors: Dict[str, List[List[int]]] = field(default_factory=lambda: {"hidden": [], "output": []})
    focused_node: Dict[str, int] = field(default_factory=lambda: {"input": None, "hidden": None, "output": None})
    focused_connections: Dict[str, int] = field(default_factory=lambda: {"input": [], "hidden": [], "output": []})
    focused_feature: int = None
    focused_inputs: List[int] = field(default_factory=list)
    focused_targets: List[int] = field(default_factory=list)
    focused_preds: Dict[str, List[List[int]]] = field(default_factory=lambda: {"hidden": [], "output": []})
    focused_errors: Dict[str, List[List[int]]] = field(default_factory=lambda: {"hidden": [], "output": []})
    focused_losses: Dict[str, List[List[int]]] = field(default_factory=lambda: {"hidden": [], "output": []})
    focus_targets: bool = False
    focus_labels: bool = None
    focus_costs: bool = False

    _X: torch.Tensor = field(init=False, repr=False)
    _domain: torch.Tensor = field(init=False, repr=False)
    _surface_points: torch.Tensor = field(init=False, repr=False)
    _surface_line: torch.Tensor = field(init=False, repr=False)
    _size: Dict[str, int] = field(init=False, repr=False)
    _node_points: Dict[str, Tuple[Tuple[float], ...]] = field(init=False, repr=False)
    _weight_eyes: Tuple[Tuple[float, float, float], ...] = field(init=False, repr=False)

    @property
    def X(self) -> int:
        self._domain = None
        return self._X

    @property
    def size(self) -> Dict[str, int]:
        return self._size

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

    @property
    def node_points(self):
        if self._node_points is None:
            self._node_points = {}
            for i, module in enumerate(reversed(self.modules)):
                size = self.size[module]
                input_xs = np.linspace(-1, 1, size) if size > 1 else [0]
                input_ys = [-i] * size
                self._node_points[module] = list(zip(input_xs, input_ys))

        return self._node_points

    @property
    def weight_eyes(self):
        if self._weight_eyes is None:
            eye = self.eye if self.eye else default_eye
            eye = (eye[0], eye[1], 0)
            return [eye for _ in range(self.w[self.modules[-1]].size(1))]

        return self._weight_eyes

    @X.setter
    def X(self, X: torch.Tensor):
        self._domain = None
        self._surface_points, self._surface_line = None, None
        self._X = X

    @size.setter
    def size(self, size: Dict[str, int]):
        self._node_points = None
        self._size = size

    @weight_eyes.setter
    def weight_eyes(self, weight_eyes: Tuple[Tuple[float, float, float], ...]):
        self._weight_eyes = weight_eyes

    def get_eye(self, as_dict=False):
        if not self.eye:
            return None

        if not as_dict:
            return self.eye

        return dict(x=self.eye[0], y=self.eye[1], z=self.eye[2])

    def get_aspect_ratio(self, as_dict=False):
        if not self.aspect_ratio:
            return None

        if not as_dict:
            return self.aspect_ratio

        return dict(x=self.aspect_ratio[0], y=self.aspect_ratio[1], z=self.aspect_ratio[2])

    def get_weight_eyes(self, as_dict=False):
        return get_weight_eyes(self.weight_eyes, as_dict=as_dict)

    def get_bias_eye(self, as_dict=False):
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

        return get_padded_domain(self.domain, padding)

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
