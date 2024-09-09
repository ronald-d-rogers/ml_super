from typing import List, Tuple
import plotly.graph_objs as go
from plotly.basedatatypes import BaseTraceType
from animation import get_colors
from base import Animation, Frame, NodeView
from scenes.base import Scene, SceneUpdate
from scenes.model.annotations import (
    feature_annotations,
    inference_annotation,
    loss_annotation,
    prediction_annotations,
    weight_annotations,
)
from scenes.model.traces import model_traces, weights_traces
from themes import Theme
from utils import TRANSPARENT


def get_default_scene(frame: Frame, theme: Theme, show_bg: bool, output_colors: List[str]):
    return go.layout.Scene(
        camera=dict(eye=frame.get_eye(as_dict=True), projection=dict(type="orthographic")),
        aspectmode="manual",
        aspectratio=frame.get_aspect_ratio(as_dict=True),
        xaxis_title="",
        yaxis_title="",
        zaxis_title="",
        xaxis=dict(
            color=theme.feature_text_colors[0],
            gridcolor=theme.feature_grid_colors[0],
            backgroundcolor=output_colors[-1] if show_bg else TRANSPARENT,
            range=frame.get_range(dim=0, pad=True),
        ),
        yaxis=dict(
            color=theme.feature_text_colors[1],
            gridcolor=theme.feature_grid_colors[1],
            backgroundcolor=output_colors[0] if show_bg else TRANSPARENT,
            range=frame.get_range(dim=1, pad=True),
        ),
        zaxis=dict(
            color=theme.target_text_color,
            gridcolor=theme.target_grid_color,
            backgroundcolor=theme.target_color if show_bg else TRANSPARENT,
            range=frame.get_zrange(pad=True),
            tickvals=[0, 0.5, 1],
        ),
    )


class ModelScene(Scene):
    names = ["Model"]
    scene_types = ["scene"]
    height = 768

    def __init__(
        self, animation: Animation, specs: List[dict], row: int, col: int, name: str = None, height: int = None
    ):
        super().__init__(animation, specs, row, col)
        self.name = name or self.name
        self.height = height or self.height

    def create_scenes(self, view: NodeView, frame: Frame) -> List[SceneUpdate]:
        theme = self.animation.theme
        show_bg = self.animation.show_bg
        output_colors, _ = get_colors(frame, view, self.animation)
        scene = get_default_scene(frame, theme, show_bg, output_colors)
        return [SceneUpdate(scene, [], [])]

    def update_scenes(self, view: NodeView, frame: Frame) -> List[SceneUpdate]:
        view = self.animation.node_view(frame)

        show_label_names = self.animation.show_label_names
        _, feature_colors = get_colors(frame, view, self.animation)

        output_module = view.modules[-1]
        w = view.w[output_module][0]
        b = view.b[output_module][0]

        theme = self.animation.theme

        scene = go.layout.Scene(annotations=[])

        scene.annotations += inference_annotation(
            w=w,
            b=b,
            inference=frame.inference,
            theme=theme,
            show=self.animation.show_model,
        )

        scene.annotations += prediction_annotations(
            X=frame.X,
            targets=view.targets,
            preds=view.preds,
            focused_errors=view.focused_errors,
            theme=theme,
            show=self.animation.show_model and frame.focused_feature is None,
        )

        scene.annotations += feature_annotations(
            X=frame.X,
            targets=view.targets,
            preds=view.preds,
            focused_feature=frame.focused_feature,
            feature_colors=feature_colors,
            theme=theme,
            show=self.animation.show_model and frame.focused_feature is not None,
        )

        data = model_traces(view, frame, self.animation)

        annotations = []

        annotations += loss_annotation(
            loss=frame.loss,
            show_label_names=show_label_names,
            label_precision=theme.label_precision,
            label_yshift=theme.cost_label_yshift,
            label_xshift=theme.cost_label_xshift,
            label_xanchor=theme.cost_label_xanchor,
            label_font_size=theme.label_font_size,
            theme=self.animation.theme,
            show=self.animation.show_model,
        )

        annotations += weight_annotations(
            w=w,
            b=b,
            width=self.animation.width,
            feature_colors=feature_colors,
            show_label_names=show_label_names,
            label_precision=theme.label_precision,
            label_yshift=theme.label_yshift,
            label_font_size=theme.label_font_size,
            theme=self.animation.theme,
            show=self.animation.show_model,
        )

        return [SceneUpdate(scene, data, annotations)]


class WeightsScene(Scene):
    dim = None
    bias = True
    height = 182
    zoom = 1

    @property
    def names(self):
        return [f"Weight {i}" for i in range(self.animation.width)] + ["Bias"] if self.bias else []

    @property
    def scene_types(self):
        return ["scene"] * len(self.names)

    def __init__(self, animation: Animation, dim: int, bias: bool = True, height: int = 182, zoom=1):
        if not dim:
            raise ValueError("dim must be a positive integer")

        self.dim = dim
        self.bias = bias
        self.height = height or self.height
        super().__init__(animation, self.scene_types)

    def create_scenes(self, view: NodeView, frame: Frame) -> List[go.Scene]:
        theme = self.animation.theme
        show_bg = self.animation.show_bg
        output_colors, feature_colors = get_colors(frame, view, self.animation)
        weight_eyes = frame.get_weight_eyes()
        bias_eye = frame.get_bias_eye()
        zoom = self.zoom

        updates = []

        for i in range(self.dim):
            scene = get_default_scene(frame, theme, show_bg, output_colors)
            scene.update(
                aspectratio=dict(x=zoom, y=zoom, z=zoom / 2),
                camera=dict(eye=weight_eyes[i]),
                xaxis=dict(
                    title="",
                    showgrid=False,
                    showticklabels=False,
                    backgroundcolor=feature_colors[i],
                ),
                yaxis=dict(
                    title="",
                    showgrid=False,
                    showticklabels=False,
                    range=frame.get_range(dim=1, pad=True),
                    backgroundcolor=feature_colors[i],
                ),
                zaxis=dict(title="", showgrid=False, showticklabels=False),
            )
            updates.append(SceneUpdate(scene, [], []))

        if self.bias:
            scene = get_default_scene(frame, theme, show_bg, output_colors)
            scene.update(
                aspectratio=dict(x=zoom / 1.4, y=zoom / 1.4, z=zoom / 2),
                camera=dict(eye=bias_eye),
                xaxis=dict(
                    title="",
                    showgrid=False,
                    showticklabels=False,
                    backgroundcolor=feature_colors[-1],
                ),
                yaxis=dict(
                    title="",
                    showgrid=False,
                    showticklabels=False,
                    backgroundcolor=feature_colors[0],
                ),
                zaxis=dict(showticklabels=False, range=frame.get_bias_zrange(pad=True)),
            )
            updates.append(SceneUpdate(scene, [], []))

        return updates

    def update_scenes(self, view: NodeView, frame: Frame) -> List[Tuple[go.Scene, List[BaseTraceType]]]:
        view = self.animation.node_view(frame)
        _, feature_colors = get_colors(frame, view, self.animation)

        updates = []
        weight_eyes = frame.get_weight_eyes()
        for i in range(self.dim):
            scene = go.layout.Scene(
                camera=dict(eye=weight_eyes[i - 1]),
                xaxis=dict(backgroundcolor=feature_colors[i - 1]),
                yaxis=dict(backgroundcolor=feature_colors[i - 1], range=frame.get_range(dim=1, pad=True)),
            )
            data = weights_traces(frame, self.animation, component=i)
            updates.append(SceneUpdate(scene, data, []))

        if self.bias:
            bias_eye = frame.get_bias_eye()
            scene = go.layout.Scene(
                camera=dict(eye=bias_eye),
                xaxis=dict(backgroundcolor=feature_colors[-1]),
                yaxis=dict(backgroundcolor=feature_colors[0]),
            )
            data = weights_traces(frame, self.animation, component="b")
            updates.append(SceneUpdate(scene, data, []))

        return updates
