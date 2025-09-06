from typing import Union

from animation_utils import get_colors
from base import Animation, AnimationFrame, LayoutComponent, ParameterView, Plot3DFrame
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


def get_plot_defaults(frame: AnimationFrame, theme: Theme, show_bg: bool, output_colors: list[str]) -> dict:
    return dict(
        camera=dict(eye=frame.get_eye(as_dict=True), projection=dict(type="orthographic")),
        aspectmode="manual",
        aspectratio=frame.get_aspect_ratio(as_dict=True),
        # xaxis_title="",
        # yaxis_title="",
        # zaxis_title="",
        xaxis=dict(
            title="",
            color=theme.feature_text_colors[0],
            gridcolor=theme.feature_grid_colors[0],
            backgroundcolor=output_colors[-1] if show_bg else TRANSPARENT,
            range=frame.get_range(dim=0, pad=True),
        ),
        yaxis=dict(
            title="",
            color=theme.feature_text_colors[1],
            gridcolor=theme.feature_grid_colors[1],
            backgroundcolor=output_colors[0] if show_bg else TRANSPARENT,
            range=frame.get_range(dim=1, pad=True),
        ),
        zaxis=dict(
            title="",
            color=theme.target_text_color,
            gridcolor=theme.target_grid_color,
            backgroundcolor=theme.target_color if show_bg else TRANSPARENT,
            range=frame.get_zrange(pad=True),
            tickvals=[0, 0.5, 1],
        ),
    )


class ModelComponent(LayoutComponent):
    names = ["Model"]
    types = ["scene"]
    height = 768

    def __init__(self, animation: Animation):
        super().__init__(animation, view_types=self.types)

    def create_component(self, frame: AnimationFrame, view: ParameterView) -> list[Plot3DFrame]:
        theme = self.animation.theme
        show_bg = self.animation.show_bg
        output_colors, _ = get_colors(frame, view, self.animation)
        plot = get_plot_defaults(frame, theme, show_bg, output_colors)
        return [Plot3DFrame(**plot)]

    def update_component(self, frame: AnimationFrame, view: ParameterView) -> list[Plot3DFrame]:
        view = self.animation.get_param_view(frame)

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

        data = model_traces(view, frame, self.animation)

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

        return [Plot3DFrame(data=data, annotations=annotations)]


class WeightsAndBiasesComponent(LayoutComponent):
    height = 250
    zoom = 3

    @property
    def names(self):
        return [f"Weight {i}" if i != "b" else "Bias" for i in self.parameters]

    @property
    def types(self):
        return ["scene"] * len(self.names)

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
        super().__init__(animation, self.types)

    def create_component(self, frame: AnimationFrame, view: ParameterView) -> list[Plot3DFrame]:
        theme = self.animation.theme
        show_bg = self.animation.show_bg
        output_colors, feature_colors = get_colors(frame, view, self.animation)
        weight_eyes = frame.get_weight_eyes(as_dict=True)
        bias_eye = frame.get_bias_eye(as_dict=True)
        zoom = self.zoom

        frames = []

        for parameter in self.parameters:
            if isinstance(parameter, int):
                i = parameter
                plot = get_plot_defaults(frame, theme, show_bg, output_colors)
                plot.update(
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
                frames.append(Plot3DFrame(**plot))
            else:
                plot = get_plot_defaults(frame, theme, show_bg, output_colors)
                plot.update(
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
                frames.append(Plot3DFrame(**plot))

        return frames

    def update_component(self, frame: AnimationFrame, view: ParameterView) -> list[Plot3DFrame]:
        _, feature_colors = get_colors(frame, view, self.animation)

        frames = []
        weight_eyes = frame.get_weight_eyes(as_dict=True)
        for parameter in self.parameters:
            i = parameter
            if isinstance(parameter, int):
                plot = dict(
                    camera=dict(eye=weight_eyes[i - 1]),
                    xaxis=dict(backgroundcolor=feature_colors[i - 1]),
                    yaxis=dict(backgroundcolor=feature_colors[i - 1], range=frame.get_range(dim=1, pad=True)),
                )
                data = weights_traces(frame, self.animation, parameter=i)
                frames.append(Plot3DFrame(data=data, **plot))
            else:
                bias_eye = frame.get_bias_eye(as_dict=True)
                plot = dict(
                    camera=dict(eye=bias_eye),
                    xaxis=dict(backgroundcolor=feature_colors[-1]),
                    yaxis=dict(backgroundcolor=feature_colors[0]),
                )
                data = weights_traces(frame, self.animation, parameter="b")
                frames.append(Plot3DFrame(data=data, **plot))

        return frames
