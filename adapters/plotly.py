import os
import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from base import AnimationFrame, Layout, LayoutComponent, Plot2DFrame, Plot3DFrame, PlotFrame
from components.gradients import GradientComponent
from components.losses import CostComponent, LossesComponent
from components.nn.components import NeuralNetworkComponent
from components.model.components import ModelComponent, WeightsAndBiasesComponent
from themes import Theme, DEFAULT_THEME, merge_themes, themes

from base import Animation


class PlotlyLayout(Layout):
    def create_figure(self, first_frame: AnimationFrame) -> go.Figure:
        value = make_subplots(
            rows=len(self._comps),
            cols=self._col_count,
            specs=self.specs,
            row_heights=self.row_heights,
            horizontal_spacing=0,
            vertical_spacing=0,
        )

        value.update_layout(
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

        view = self.animation.get_param_view(first_frame)

        # Plotly refers to 3D plots as "scenes"
        scene_count = 0
        for component in self._comps:
            for update in component.create_component(view, first_frame):
                if isinstance(update, Plot3DFrame):
                    value.layout[get_scene_key(scene_count)].update(get_3d_plot(update))
                    scene_count += 1

                if update.annotations:
                    value.update_annotations(get_annotations(update))

        return value

    @property
    def specs(self):
        if not self._specs:
            self._specs = get_specs(self._comps, self._col_count)
        return self._specs

    def make_frames(self, frames: list[AnimationFrame]) -> list[go.Frame]:
        return [self.make_frame(frame, name=i) for i, frame in enumerate(frames)]

    def make_frame(self, frame: AnimationFrame, name=0) -> go.Frame:
        value = go.Frame(name=str(name), data=[], layout=go.Layout(annotations=[]))
        view = self.animation.get_param_view(frame)

        data = []

        row_count = 0
        scene_count = 0
        for component in self._comps:
            col_count = 0
            for update in component.update_component(view, frame):
                if isinstance(update, Plot3DFrame):
                    value.layout[get_scene_key(scene_count)] = get_3d_plot(update)
                    scene_count += 1

                if update.annotations:
                    value.update_annotations(get_annotations(update))

                if update.data:
                    for trace in update.data:
                        if trace.meta is None:
                            trace.meta = {}
                        trace.meta["row"] = row_count
                        trace.meta["col"] = col_count
                    data.extend(update.data)

                col_count += 1
            row_count += 1

        value.data = data

        return value


def get_annotations(frame: PlotFrame):
    if frame.annotations:
        return [
            go.layout.scene.Annotation(**{k: v for k, v in a.items() if k != "scope"})
            for a in frame.annotations
            if a.get("scope", "") == "plot"
        ]
    return []


def get_3d_plot(frame: Plot3DFrame):
    scene = go.layout.Scene()

    for key in frame.keys() - ["data", "annotations"]:
        setattr(scene, key, getattr(frame, key))

    for annotation in frame.annotations:
        if annotation.get("scope", "") == "plot":
            scene.annotations.append(
                go.layout.scene.Annotation(**{k: v for k, v in annotation.items() if k != "scope"})
            )

    return scene


def get_specs(components: list[LayoutComponent], max_cols: int):
    return [
        [
            dict(type=component.types[col], colspan=max_cols // len(component.types))
            if col < len(component.types)
            else None
            for col in range(max_cols)
        ]
        for component in components
    ]


def get_scene_key(index: int):
    if index == 0:
        return "scene"
    return f"scene{index + 1}"


def animate(
    frames: list[AnimationFrame],
    model_param="output_1",
    show_model=True,
    show_params=False,
    show_gradients=False,
    show_tables=False,
    show_network=False,
    show_bg=True,
    show_weight_preds=False,
    scrollable=True,
    scale=1,
    render_path=None,
    theme=DEFAULT_THEME,
    theme_args: Theme = None,
):
    if isinstance(theme, str):
        theme = themes[theme]

    if theme_args:
        if isinstance(theme_args, dict):
            theme_args = Theme(**theme_args)
        theme = merge_themes(theme, theme_args)

    animation = Animation(
        frames=frames,
        model_param=model_param,
        render_path=render_path,
        show_model=show_model,
        show_params=show_params,
        show_gradients=show_gradients,
        show_network=show_network,
        show_calculations=show_tables,
        show_bg=show_bg,
        show_weights_preds=show_weight_preds,
        scale=scale,
        theme=theme,
    )

    layout = PlotlyLayout(animation=animation)

    first_frame = frames[0]
    view = animation.get_param_view(first_frame)

    input_module = view.modules[-2]
    input_size = first_frame.size[input_module]

    if show_params is True:
        show_params = list(range(1, input_size + 1)) + ["b"]

    animation.show_params = show_params

    if show_model:
        layout.add_component(ModelComponent(animation))

    if show_params:
        layout.add_component(WeightsAndBiasesComponent(animation, parameters=show_params))

    if show_gradients:
        layout.add_component(GradientComponent(animation))

    if show_tables:
        layout.add_component(LossesComponent(animation))
        layout.add_component(CostComponent(animation))

    if show_network:
        layout.add_component(NeuralNetworkComponent(animation))

    fig = layout.create_figure(first_frame)

    if render_path:
        os.makedirs(render_path, exist_ok=True)
        first_frame = layout.make_frame(frames[0], name=0)

        for trace in first_frame.data:
            fig.add_trace(trace=trace, row=trace.meta["row"] + 1, col=trace.meta["col"] + 1)

        zlen = len(str(len(frames)))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        for i, frame in enumerate(tqdm.tqdm(frames)):
            first_frame = layout.make_frame(frame, name=i)

            for trace1, trace2 in zip(fig.data, frame.data):
                trace1.update(trace2)

            fig.update_layout(frame.layout)
            num = str(i).zfill(zlen)
            fig.write_image(os.path.join(render_path, f"{num}.png"), format="png", scale=scale)

    else:
        frames = layout.make_frames(frames)

        for trace in frames[0].data:
            fig.add_trace(trace=trace, row=trace.meta["row"] + 1, col=trace.meta["col"] + 1)

        fig.update(frames=frames)

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
