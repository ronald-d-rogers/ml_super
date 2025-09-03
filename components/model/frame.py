from typing import Union
import plotly.graph_objs as go

from animation_utils import get_colors
from base import Animation, AnimationFrame, LayoutComponent, NodeView
from base import ViewFrameUpdate
from components.model.annotations import (
    feature_annotations,
    inference_annotation,
    loss_annotation,
    prediction_annotations,
    weight_annotations,
)
from components.model.traces import model_traces, weights_traces
from themes import Theme
from utils import TRANSPARENT


def get_default_scene(frame: AnimationFrame, theme: Theme, show_bg: bool, output_colors: list[str]) -> go.layout.Scene:
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


class ModelComponent(LayoutComponent):
    view_names = ["Model"]
    view_types = ["scene"]
    height = 768

    def __init__(self, animation: Animation):
        super().__init__(animation, view_types=self.view_types)

    def create_component(self, view: NodeView, frame: AnimationFrame) -> list[ViewFrameUpdate]:
        theme = self.animation.theme
        show_bg = self.animation.show_bg
        output_colors, _ = get_colors(frame, view, self.animation)
        scene = get_default_scene(frame, theme, show_bg, output_colors)
        return [ViewFrameUpdate(scene, [], [])]

    def update_component(self, view: NodeView, frame: AnimationFrame) -> list[ViewFrameUpdate]:
        view = self.animation.node_view(frame)

        show_label_names = self.animation.show_label_names
        _, feature_colors = get_colors(frame, view, self.animation)

        output_module = view.modules[-1]
        w = view.w[output_module][0]
        b = view.b[output_module][0]

        theme = self.animation.theme

        annotations = []
        annotations += inference_annotation(
            w=w,
            b=b,
            inference=frame.inference,
            theme=theme,
            show=self.animation.show_model,
        )

        annotations += prediction_annotations(
            X=frame.X,
            targets=view.targets,
            preds=view.preds,
            focused_errors=view.focused_errors,
            theme=theme,
            show=self.animation.show_model and frame.focused_feature is None,
        )

        annotations += feature_annotations(
            X=frame.X,
            targets=view.targets,
            preds=view.preds,
            focused_feature=frame.focused_feature,
            feature_colors=feature_colors,
            theme=theme,
            show=self.animation.show_model and frame.focused_feature is not None,
        )

        scene = go.layout.Scene(annotations=annotations)

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

        return [ViewFrameUpdate(scene, data, annotations)]


class WeightsAndBiasesComponent(LayoutComponent):
    height = 250
    zoom = 3

    @property
    def view_names(self):
        return [f"Weight {i}" if i != "b" else "Bias" for i in self.parameters]

    @property
    def column_types(self):
        return ["scene"] * len(self.view_names)

    def __init__(self, animation: Animation, parameters: list[Union[int, str]], height: int = None, zoom: int = None):
        if not parameters:
            raise ValueError("Parameters must be a non-empty list of integers or strings")

        for i, parameter in enumerate(parameters):
            if not isinstance(parameter, (int, str)):
                raise ValueError(
                    f"Invalid parameter at index {i}: Parameters must be integers or strings but got `{parameter}`"
                )

        self.parameters = parameters
        self.height = height or self.height
        self.zoom = zoom or self.zoom
        super().__init__(animation, self.column_types)

    def create_component(self, view: NodeView, frame: AnimationFrame) -> list[ViewFrameUpdate]:
        theme = self.animation.theme
        show_bg = self.animation.show_bg
        output_colors, feature_colors = get_colors(frame, view, self.animation)
        weight_eyes = frame.get_weight_eyes(as_dict=True)
        bias_eye = frame.get_bias_eye(as_dict=True)
        zoom = self.zoom

        defs = []

        for parameter in self.parameters:
            if isinstance(parameter, int):
                i = parameter
                scene = get_default_scene(frame, theme, show_bg, output_colors)
                scene.update(
                    aspectratio=dict(x=zoom, y=zoom, z=zoom / 2),
                    camera=dict(eye=weight_eyes[i - 1]),
                    xaxis=dict(
                        title="",
                        showgrid=False,
                        showticklabels=False,
                        backgroundcolor=feature_colors[i - 1],
                    ),
                    yaxis=dict(
                        title="",
                        showgrid=False,
                        showticklabels=False,
                        range=frame.get_range(dim=1, pad=True),
                        backgroundcolor=feature_colors[i - 1],
                    ),
                    zaxis=dict(title="", showgrid=False, showticklabels=False),
                )
                defs.append(ViewFrameUpdate(scene, [], []))
            else:
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
                defs.append(ViewFrameUpdate(scene, [], []))

        return defs

    def update_component(self, view: NodeView, frame: AnimationFrame) -> list[ViewFrameUpdate]:
        _, feature_colors = get_colors(frame, view, self.animation)

        defs = []
        weight_eyes = frame.get_weight_eyes(as_dict=True)
        for parameter in self.parameters:
            i = parameter
            if isinstance(parameter, int):
                scene = go.layout.Scene(
                    camera=dict(eye=weight_eyes[i - 1]),
                    xaxis=dict(backgroundcolor=feature_colors[i - 1]),
                    yaxis=dict(backgroundcolor=feature_colors[i - 1], range=frame.get_range(dim=1, pad=True)),
                )
                data = weights_traces(frame, self.animation, parameter=i)
                defs.append(ViewFrameUpdate(scene, data, []))
            else:
                bias_eye = frame.get_bias_eye(as_dict=True)
                scene = go.layout.Scene(
                    camera=dict(eye=bias_eye),
                    xaxis=dict(backgroundcolor=feature_colors[-1]),
                    yaxis=dict(backgroundcolor=feature_colors[0]),
                )
                data = weights_traces(frame, self.animation, parameter="b")
                defs.append(ViewFrameUpdate(scene, data, []))

        return defs
