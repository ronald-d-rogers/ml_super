from typing import Any, Optional, Union
from dataclasses import dataclass, field

import torch
import numpy as np


from themes import Theme, DEFAULT_THEME
from utils import clone, hex_to_rgb, interp_rgb, rgb_to_str


DEFAULT_CONTROLS_HEIGHT = 130


class PlotFrame:
    data: list[Any]
    annotations: list[Any]

    def __init__(self, data, annotations):
        self.data = data if data is not None else []
        self.annotations = annotations if annotations is not None else []

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def keys(self):
        return ["data", "annotations"]


class Plot2DFrame(PlotFrame):
    xaxis: Any
    yaxis: Any

    def __init__(self, data, annotations, xaxis, yaxis):
        super().__init__(data, annotations)
        self.xaxis = xaxis if xaxis is not None else None
        self.yaxis = yaxis if yaxis is not None else None

    def keys(self):
        return super().keys() + ["xaxis", "yaxis"]


class Plot3DFrame(PlotFrame):
    xaxis: Any
    yaxis: Any
    zaxis: Any
    camera: Any
    aspectratio: Any
    aspectmode: Any = "manual"

    def __init__(self, data, annotations, xaxis, yaxis, zaxis, camera, aspectratio, aspectmode="manual"):
        super().__init__(data, annotations)
        self.xaxis = xaxis if xaxis is not None else None
        self.yaxis = yaxis if yaxis is not None else None
        self.zaxis = zaxis if zaxis is not None else None
        self.camera = camera if camera is not None else None
        self.aspectratio = aspectratio if aspectratio is not None else None
        self.aspectmode = aspectmode if aspectmode is not None else "manual"

    def keys(self):
        return super().keys() + ["xaxis", "yaxis", "zaxis", "camera", "aspectratio", "aspectmode"]


@dataclass
class ParameterView:
    w: dict[str, torch.Tensor]
    b: dict[str, torch.Tensor]
    modules: list[str]
    targets: torch.Tensor
    preds: torch.Tensor = None
    derivatives: torch.Tensor = None
    errors: torch.Tensor = None
    losses: torch.Tensor = None
    costs: torch.Tensor = None
    focused_preds: list[int] = None
    focused_errors: list[int] = None
    activity: float = 1
    weight_eyes: tuple[tuple[float, float, float], ...] = None
    bias_eye: tuple[float, float, float] = (1, 1, 0)

    def get_weight_eyes(self, as_dict=False):
        return get_weight_eyes(self.weight_eyes, as_dict=as_dict)


@dataclass
class Animation:
    frames: list["AnimationFrame"]
    model_param: str = "output_1"
    show_model: bool = True
    show_network: bool = False
    show_calculations: bool = False
    height: int = 1920
    width: int = 1080
    scale: int = 2
    show_bg: bool = True
    render_path: str = None
    show_params: bool = True
    show_gradients: bool = True
    show_weights_preds: bool = False
    show_label_names: bool = True
    show_controls: bool = False
    theme: Theme = DEFAULT_THEME
    cells: dict = field(default_factory=dict)

    _param_module = None
    _param_index = None

    @property
    def param_module(self):
        if self._param_module is None:
            self._param_module = parse_param_module(self.model_param)
        return self._param_module

    @property
    def param_index(self):
        if self._param_index is None:
            self._param_index = parse_param_index(self.model_param)
        return self._param_index

    def colors(self, size: int):
        values = self.theme.feature_colors
        return interp_colors(values[0], values[1], size)

    def focused_colors(self, size):
        values = self.theme.focused_feature_colors
        module_size = size
        return interp_colors(values[0], values[1], module_size)

    def focusable_colors(self, focused: int, size: int):
        return focusable_feature_colors(focused, self.colors(size), self.focused_colors(size))

    def get_param_view(
        self,
        frame: "AnimationFrame",
        parameter: Union[int, str] = None,
        weight_eyes: tuple[tuple[float, float, float], ...] = None,
        activity: float = None,
    ):
        module = self.param_module
        index = self.param_index

        def get_param_data(data: dict, module, index):
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

        if parameter is not None:
            if isinstance(parameter, int):
                parameter_index = parameter - 1

                # if we're viewing a node with hidden nodes
                if not hidden == output and hidden in w:
                    mask = [0 for _ in range(frame.size[hidden])]
                    mask[parameter_index] = 1
                    w[hidden] = torch.Tensor([mask]).T * w[hidden]
                    b[hidden] = torch.Tensor([mask]).T * b[hidden]
                    activity = activity if activity is not None else 1

                else:
                    mask = [0 for _ in range(frame.size[input])]
                    mask[parameter_index] = 1
                    w[output] = torch.Tensor([mask]) * w[output]
                    b[output] = torch.Tensor([[0]])
                    activity = activity if activity is not None else 0

            elif parameter == "b":
                w[output] = torch.Tensor([[0, 0]])
                activity = activity if activity is not None else 0

                if not hidden == output and hidden in w:
                    del w[hidden]

                modules = [input, output]

            else:
                raise ValueError(
                    f'Unsupported parameter "{parameter}". You may select a parameter from the following: "b", or an integer index of the compoennt.'
                )

        return ParameterView(
            w=w,
            b=b,
            targets=frame.targets[0],
            modules=modules,
            preds=get_param_data(frame.preds, module, index),
            errors=get_param_data(frame.errors, module, index),
            losses=get_param_data(frame.losses, module, index),
            costs=get_param_data(frame.costs, module, index),
            focused_preds=get_param_data(frame.focused_preds, module, index),
            focused_errors=get_param_data(frame.focused_errors, module, index),
            weight_eyes=frame.weight_eyes if weight_eyes is None else weight_eyes,
            activity=frame.activity if activity is None else activity,
        )


default_eye = (1, 1, 1)


@dataclass
class AnimationFrame:
    X: torch.Tensor
    targets: torch.Tensor = None
    preds: dict[str, torch.Tensor] = None
    derivatives: dict[str, torch.Tensor] = None
    errors: dict[str, torch.Tensor] = None
    losses: dict[str, torch.Tensor] = None
    costs: dict[str, torch.Tensor] = None
    loss: torch.Tensor = None
    w: dict[str, torch.Tensor] = None
    b: dict[str, torch.Tensor] = None
    loss_fn: callable = None
    activation_fns: dict[str, callable] = None
    size: dict[str, int] = field(default_factory=lambda: {"input": 2, "hidden": 2, "output": 1})
    modules: list[str] = field(default_factory=lambda: ["input", "hidden", "output"])
    epochs: int = 30
    learning_rate: float = 0.5
    resolution: int = 30
    activity: float = 1
    zrange: Optional[tuple[float, float]] = (0, 1)
    bias_zrange: Optional[tuple[float, float]] = (-1, 2)
    domain_padding: float = 2
    range_padding: float = 2.5
    zrange_padding: float = 0.05
    eye: tuple[float, float, float] = default_eye
    weight_eyes: tuple[tuple[float, float, float], ...] = None
    bias_eye: tuple[float, float, float] = (1, 1, 0)
    aspect_ratio: tuple[float, float, float] = (1, 1, 1)
    show_preds: bool = True
    show_profile: bool = False
    show_decision_boundaries: bool = False
    inference: torch.Tensor = None
    active_preds: dict[str, list[list[int]]] = field(default_factory=lambda: {"hidden": [], "output": []})
    active_errors: dict[str, list[list[int]]] = field(default_factory=lambda: {"hidden": [], "output": []})
    focused_node: dict[str, int] = field(default_factory=lambda: {"input": None, "hidden": None, "output": None})
    focused_connections: dict[str, int] = field(default_factory=lambda: {"input": [], "hidden": [], "output": []})
    focused_feature: int = None
    focused_inputs: list[int] = field(default_factory=list)
    focused_targets: list[int] = field(default_factory=list)
    focused_preds: dict[str, list[list[int]]] = field(default_factory=lambda: {"hidden": [], "output": []})
    focused_errors: dict[str, list[list[int]]] = field(default_factory=lambda: {"hidden": [], "output": []})
    focused_losses: dict[str, list[list[int]]] = field(default_factory=lambda: {"hidden": [], "output": []})
    focus_targets: bool = False
    focus_labels: bool = None
    focus_costs: bool = False

    _X: torch.Tensor = field(init=False, repr=False)
    _domain: torch.Tensor = field(init=False, repr=False)
    _surface_points: torch.Tensor = field(init=False, repr=False)
    _surface_line: torch.Tensor = field(init=False, repr=False)
    _size: dict[str, int] = field(init=False, repr=False)
    _param_points: dict[str, tuple[tuple[float], ...]] = field(init=False, repr=False)
    _weight_eyes: tuple[tuple[float, float, float], ...] = field(init=False, repr=False)

    @property
    def X(self) -> int:
        self._domain = None
        return self._X

    @property
    def size(self) -> dict[str, int]:  # noqa: F811
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
    def param_points(self):
        if self._param_points is None:
            self._param_points = {}
            for i, module in enumerate(reversed(self.modules)):
                size = self.size[module]
                input_xs = np.linspace(-1, 1, size) if size > 1 else [0]
                input_ys = [-i] * size
                self._param_points[module] = list(zip(input_xs, input_ys))

        return self._param_points

    @property
    def weight_eyes(self):  # noqa: F811
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
    def size(self, size: dict[str, int]):
        self._param_points = None
        self._size = size

    @weight_eyes.setter
    def weight_eyes(self, weight_eyes: tuple[tuple[float, float, float], ...]):
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


class LayoutComponent:
    animation: Animation
    height: int
    names: list[str]
    types: list[str]

    @property
    def column_count(self):
        return len(self.types)

    def __init__(self, animation: Animation, plot_types: list[str]) -> None:
        self.animation = animation
        self.types = plot_types

    def create_component(self, frame: AnimationFrame, view: ParameterView) -> list[PlotFrame]:
        raise NotImplementedError

    def update_component(self, frame: AnimationFrame, view: ParameterView) -> list[PlotFrame]:
        raise NotImplementedError


class Layout:
    animation: Animation
    _comps: list[LayoutComponent] = []
    _specs: list[dict] = []
    _height: int = None
    _col_count: int = 0

    @property
    def height(self):
        if not self._height:
            self._height = sum([plot.height for plot in self._comps])
        return self._height

    @property
    def col_count(self):
        return self._col_count

    @property
    def row_count(self):
        return len(self._comps)

    @property
    def row_heights(self):
        return [comp.height for comp in self._comps]

    def __init__(self, animation: Animation):
        self.animation = animation

    def add_component(self, component: LayoutComponent):
        self._col_count = max(self._col_count, component.column_count)
        self._comps.append(component)

    def create_figure(self, first_frame: AnimationFrame):
        raise NotImplementedError

    def make_frames(self, frames: list[AnimationFrame]) -> list[Any]:
        raise NotImplementedError

    def make_frame(self, frame: AnimationFrame, name=0) -> Any:
        raise NotImplementedError


def parse_param_module(param_string: str):
    return param_string.split("_")[0]


def parse_param_index(param_string: str):
    return int(param_string.split("_")[1]) - 1


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


def focusable_feature_colors(focused_feature: Union[None, bool], colors: list[str], focused_colors: list[str]):
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
