import os
from typing import List
import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from components.base import SceneComponent
from components.gradients import GradientComponent, gradient_traces
from components.losses import CostComponent, LossesComponent, loss_traces
from components.nn.frame import NeuralNetworkScene
from components.nn.traces import nn_traces
from themes import Theme, default_theme, merge_themes, themes
from components.model.traces import model_surface, weights_traces
from components.nn.annotations import nn_annotations

from components.model.frame import ModelComponent, WeightsAndBiasesComponent

from base import AnimationFrame, Animation, NodeView
from utils import TRANSPARENT


def get_colors(frame: AnimationFrame, view: NodeView, animation: Animation):
    output = view.modules[-1]
    hidden_or_input = view.modules[-2]
    feature_colors = animation.focusable_colors(frame.focused_feature, frame.size[hidden_or_input])

    # if it has a hidden layer
    if len(view.modules) > 2:
        color = animation.colors(frame.size[output])[animation.node_index]
        output_colors = [color]
    else:
        output_colors = feature_colors

    return output_colors, feature_colors


def make_frame(frame: AnimationFrame, animation: Animation, name: str):
    if frame.eye:
        eye = dict(x=frame.eye[0], y=frame.eye[1], z=frame.eye[2])
    else:
        eye = None

    view = animation.node_view(frame)
    input_module = view.modules[-2]
    input_size = frame.size[input_module]
    output_colors, feature_colors = get_colors(frame, view, animation)
    weight_eyes = view.get_weight_eyes(as_dict=True)
    bias_eye = frame.get_bias_eye(as_dict=True)
    show_bg = animation.show_bg

    value = go.Frame(
        name=name,
        data=[
            *model_surface(frame, animation),
            *weights_traces(frame, animation),
            *loss_traces(view, frame, animation),
            *nn_traces(frame, animation),
            *gradient_traces(frame, animation),
        ],
        layout=dict(
            annotations=[
                *nn_annotations(frame, animation, show=animation.show_network),
            ],
            scene=dict(
                camera=dict(eye=eye),
                aspectratio=dict(
                    x=frame.aspect_ratio[0],
                    y=frame.aspect_ratio[1],
                    z=frame.aspect_ratio[2],
                ),
                zaxis_title="",
                xaxis=dict(
                    # color=animation.theme.feature_text_colors[0],
                    backgroundcolor=output_colors[0] if show_bg else TRANSPARENT,
                    range=frame.get_range(dim=0, pad=True),
                ),
                yaxis=dict(
                    # color=animation.theme.feature_text_colors[1],
                    backgroundcolor=output_colors[-1] if show_bg else TRANSPARENT,
                    range=frame.get_range(dim=1, pad=True),
                ),
                zaxis=dict(range=frame.get_zrange(pad=True)),
                annotations=[],
            ),
        ),
    )

    if animation.show_parameters:
        for i in animation.show_parameters:
            if isinstance(i, int):
                value.layout[f"scene{i + 1}"] = dict(
                    camera=dict(eye=weight_eyes[i - 1]),
                    xaxis=dict(backgroundcolor=feature_colors[i - 1]),
                    yaxis=dict(backgroundcolor=feature_colors[i - 1], range=frame.get_range(dim=1, pad=True)),
                )

            if i == "b":
                value.layout[f"scene{input_size + 2}"] = dict(
                    camera=dict(eye=bias_eye),
                    xaxis=dict(backgroundcolor=feature_colors[-1]),
                    yaxis=dict(backgroundcolor=feature_colors[0]),
                    zaxis=dict(range=frame.get_bias_zrange(pad=True)),
                )

    return value


def get_specs(scenes: List[SceneComponent], num_columns: int):
    return [
        dict(type=scene.scene_types[col]) if col < len(scene.scene_types) else None
        for col in range(num_columns)
        for scene in scenes
    ]


def get_scene_key(index: int):
    if index == 0:
        return "scene"
    return f"scene{index}"


class Layout:
    scenes: List[SceneComponent] = []

    _positions: dict = {}
    _specs = List[dict]

    _height: int = None
    _columns: int = None

    @property
    def height(self):
        if not self._height:
            self._height = sum([scene.height for scene in self._scenes])
        return self._height

    @property
    def columns(self):
        return self._columns

    @property
    def row_heights(self):
        return [scene.height for scene in self._scenes]

    @property
    def specs(self):
        if not self._specs:
            self._specs = get_specs(self._scenes, self._columns)
        return self._specs

    def __init__(self, layout: List[str], animation: Animation):
        self.animation = animation

    def add_scene(self, scene: SceneComponent):
        self._columns = max(self._columns, scene.columns)

        self.components.append(scene)

    def create_figure(self) -> go.Figure:
        fig = make_subplots(
            rows=len(self.scenes),
            cols=self._columns,
            specs=self.specs,
            row_heights=self.row_heights,
            horizontal_spacing=0,
            vertical_spacing=0,
        )

        fig.update_layout(
            autosize=False,
            width=self.animation.width,
            height=self.animation.height,
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False,
            transition=dict(duration=0, easing="linear", ordering="traces first"),
            font=dict(
                family=self.animation.theme.font_family,
                size=self.animation.theme.font_size,
                color=self.animation.theme.text_color,
            ),
            paper_bgcolor=self.animation.theme.background_color,
            plot_bgcolor="rgba(0, 0, 0, 0)",
            xaxis=dict(
                color=self.animation.theme.feature_text_colors[0],
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(color=self.animation.theme.feature_text_colors[1], showticklabels=False),
        )

        scene_counter = 0
        for scene in self.scenes:
            for scene_type in scene.scene_types:
                if scene_type == "scene":
                    scene_counter += 1
                    fig[get_scene_key(scene_counter)].update(scene.create_component())

        fig[get_scene_key(scene_counter)].update(self.scenes[scene_counter].create_component())

    def render_frames(self) -> List[go.Frame]:
        pass

    def render_frame(self, frame: AnimationFrame) -> go.Frame:
        pass


def animate(
    frames: list[AnimationFrame],
    model_node="output_1",
    show_model=True,
    show_parameters=False,
    show_gradients=False,
    show_tables=False,
    show_network=False,
    show_bg=True,
    show_weight_preds=False,
    scrollable=True,
    scale=1,
    render_path=None,
    theme=default_theme,
    overrides: Theme = None,
):
    cells = {}
    specs = []
    scenes = []

    if isinstance(theme, str):
        theme = themes[theme]

    if overrides:
        theme = merge_themes(theme, overrides)

    animation = Animation(
        frames=frames,
        model_node=model_node,
        render_path=render_path,
        show_model=show_model,
        show_parameters=show_parameters,
        show_gradients=show_gradients,
        show_network=show_network,
        show_calculations=show_tables,
        show_bg=show_bg,
        show_weights_preds=show_weight_preds,
        scale=scale,
        cells=cells,
        scenes=scenes,
        theme=theme,
    )

    frame = frames[0]
    view = animation.node_view(frame)

    input_module = view.modules[-2]
    input_size = frame.size[input_module]

    if show_parameters is True:
        show_parameters = list(range(1, input_size + 1)) + ["b"]

    animation.show_parameters = show_parameters

    height = 0
    width = 1080 * scale

    controls_height = 130

    # add one for bias
    num_columns = input_size + 1

    row_count = 0
    row_heights = []
    scenes = []

    if show_model:
        components.append(ModelComponent(animation))

    if show_parameters:
        components.append(WeightsAndBiasesComponent(animation, parameters=show_parameters))

    if show_gradients:
        components.append(GradientComponent(animation))

    if show_tables:
        components.append(LossesComponent(animation))
        components.append(CostComponent(animation))

    if show_network:
        components.append(NeuralNetworkScene(animation))

    if not render_path:
        height += controls_height

    _, feature_colors = get_colors(frame, view, animation)

    animation.width = width
    animation.height = height

    # model_node = animation.node_view(frame)

    fig = make_subplots(
        rows=row_count,
        cols=num_columns,
        specs=specs,
        row_heights=row_heights,
        horizontal_spacing=0,
        vertical_spacing=0,
    )

    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        transition=dict(duration=0, easing="linear", ordering="traces first"),
        font=dict(
            family=theme.font_family,
            size=theme.font_size,
            color=theme.text_color,
        ),
        paper_bgcolor=theme.background_color,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        xaxis=dict(
            color=theme.feature_text_colors[0],
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(color=theme.feature_text_colors[1], showticklabels=False),
    )

    fig.for_each_scene(lambda x: x.update(scene))

    if show_parameters:
        zoom = 3

        weight_eyes = view.get_weight_eyes(as_dict=True)
        bias_eye = frame.get_bias_eye(as_dict=True)

        for i in show_parameters:
            if isinstance(i, int):
                fig.layout[f"scene{i + 1}"].update(
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

            elif i == "b":
                fig.layout[f"scene{input_size + 2}"].update(
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

    def update_network():
        cell = cells["network"]
        fig.update_yaxes(range=[-2, 0], row=cell["row"], col=cell["col"])
        fig.update_xaxes(range=[-1.5, 1.5], row=cell["row"], col=cell["col"])

    if render_path:
        os.makedirs(render_path, exist_ok=True)
        frame = make_frame(frames[0], animation, name=0)

        for trace in frame.data:
            fig.add_trace(trace=trace, row=trace.meta["row"], col=trace.meta["col"])

        zlen = len(str(len(frames)))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        if show_network:
            update_network()

        for i, frame in enumerate(tqdm.tqdm(frames)):
            frame = make_frame(frame, animation, name=i)

            for trace1, trace2 in zip(fig.data, frame.data):
                trace1.update(trace2)

            fig.update_layout(frame.layout)
            num = str(i).zfill(zlen)
            fig.write_image(os.path.join(render_path, f"{num}.png"), format="png", scale=scale)

    else:
        frames = [make_frame(frame, animation, name=i) for i, frame in enumerate(frames)]

        for trace in frames[0].data:
            fig.add_trace(trace=trace, row=trace.meta["row"], col=trace.meta["col"])

        fig.update(frames=frames)

        if show_network:
            update_network()

        play_args = dict(
            frame=dict(duration=50, redraw=True),
            transition=dict(duration=0),
            mode="immediate",
        )

        pause_args = dict(frame=dict(duration=0, redraw=False), mode="immediate")

        sliders = [
            dict(
                font=dict(size=16),
                steps=[
                    dict(
                        label=i,
                        method="animate",
                        args=[[i, play_args], dict(mode="immediate")],
                    )
                    for i, _ in enumerate(fig.frames)
                ],
            )
        ]

        menu = dict(
            type="buttons",
            xanchor="center",
            direction="right",
            x=0.5,
            y=0,
            showactive=True,
            font=dict(size=16),
            buttons=[
                dict(label="Play", method="animate", args=[None, play_args]),
                dict(label="Pause", method="animate", args=[None, pause_args]),
            ],
        )

        fig.update_layout(updatemenus=[menu], sliders=sliders)

        fig.show(config=dict(scrollZoom=scrollable))
