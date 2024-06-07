import os

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import tqdm


from annotations.neural_network import neural_network_annotations
from themes import default_theme
from annotations.scene import (
    error_annotations,
    feature_annotations,
    inference_annotation,
    weight_annotations,
)
from traces.scene import data_markers, model_surface

from traces.neural_network import neural_network
from traces.tables import cost_tables

from base import Frame, Animation, View


def scene_traces(
    view: View,
    frame: Frame,
    show_profile,
    focused_feature,
    marker_size,
    theme,
    meta=None,
):
    return [
        *data_markers(
            X=frame.X,
            w=view.w["output"][0],
            b=view.b["output"][0],
            preds=view.preds,
            targets=view.targets,
            inference=frame.inference,
            focused_feature=focused_feature,
            focused_errors=view.focused_errors,
            marker_size=marker_size,
            meta=meta,
            theme=theme,
        ),
        *model_surface(
            domain=frame.get_domain(pad=True),
            surface_points=frame.surface_points,
            surface_linspace=frame.surface_linspace,
            w=view.w,
            b=view.b,
            activity=view.activity,
            show_profile=show_profile,
            meta=meta,
        ),
    ]


def model_view(view: View, frame: Frame, animation: Animation):
    if not animation.show_model:
        return []

    meta = animation.meta["model"]

    return scene_traces(
        view,
        frame,
        show_profile=frame.show_profile,
        focused_feature=frame.focused_feature,
        marker_size=animation.marker_size,
        theme=animation.theme,
        meta=dict(row=meta["row"], col=meta["col"]),
    )


def components_view(frame: Frame, animation: Animation):
    show_components = animation.show_components

    if not show_components:
        return []

    meta1 = animation.meta["component1"]
    meta2 = animation.meta["component2"]
    meta3 = animation.meta["component3"]

    w1, w2, bias = [], [], []

    if isinstance(show_components, bool):
        show_w1, show_w2, show_bias = (True, True, True)
    else:
        show_w1, show_w2, show_bias = show_components

    scene_args = dict(
        marker_size=animation.marker_size * 0.666,
        show_profile=True,
        focused_feature=None,
        theme=animation.theme,
    )

    if show_w1:
        view = animation.node_view(frame, component="w1")

        w1 = scene_traces(
            view=view,
            frame=frame,
            meta=dict(row=meta1["row"], col=meta1["col"]),
            **scene_args,
        )

    if show_w2:
        view = animation.node_view(frame, component="w2")

        w2 = scene_traces(
            view=view,
            frame=frame,
            meta=dict(row=meta2["row"], col=meta2["col"]),
            **scene_args,
        )

    if show_bias:
        view = animation.node_view(frame, component="b")

        bias = scene_traces(
            view=view,
            frame=frame,
            meta=dict(row=meta3["row"], col=meta3["col"]),
            **scene_args,
        )

    return [*w1, *w2, *bias]


def tables_view(view: View, frame: Frame, animation: Animation):
    if not animation.show_calculations:
        return []

    losses_meta = animation.meta["losses"]
    cost_meta = animation.meta["cost"]

    return cost_tables(
        view,
        frame,
        animation,
        meta=(
            dict(row=losses_meta["row"], col=losses_meta["col"]),
            dict(row=cost_meta["row"], col=cost_meta["col"]),
        ),
    )


def network_view(frame: Frame, animation: Animation):
    if not animation.show_network:
        return []

    meta = animation.meta["network"]

    return [
        *neural_network(
            frame,
            animation,
            meta=dict(row=meta["row"], col=meta["col"]),
        )
    ]


def make_frame(frame: Frame, animation: Animation, name: str):
    feature_colors = animation.focusable_feature_colors(frame.focused_feature)

    if frame.eye:
        eye = dict(x=frame.eye[0], y=frame.eye[1], z=frame.eye[2])
    else:
        eye = None

    view = animation.node_view(frame)

    weight_eyes = view.get_weight_eyes()
    bias_eye = frame.get_bias_eye()

    w = view.w["output"][0]
    b = view.b["output"][0]

    return go.Frame(
        name=name,
        data=[
            *model_view(view, frame, animation),
            *components_view(frame, animation),
            *tables_view(view, frame, animation),
            *network_view(frame, animation),
        ],
        layout=dict(
            annotations=[
                *weight_annotations(
                    w=w,
                    b=b,
                    height=animation.height,
                    focused_feature=frame.focused_feature,
                    focus_labels=frame.focus_labels,
                    theme=animation.theme,
                    show=animation.show_model,
                ),
                *neural_network_annotations(frame, animation, show=animation.show_network),
            ],
            scene=dict(
                camera=dict(eye=eye),
                aspectratio=dict(x=frame.aspect_ratio[0], y=frame.aspect_ratio[1], z=frame.aspect_ratio[2]),
                xaxis_title="",
                yaxis_title="",
                zaxis_title="",
                xaxis=dict(backgroundcolor=feature_colors[1], range=frame.get_range(dim=0, pad=True)),
                yaxis=dict(backgroundcolor=feature_colors[0], range=frame.get_range(dim=1, pad=True)),
                zaxis=dict(range=frame.get_zrange(pad=True)),
                annotations=[
                    *inference_annotation(w=w, b=b, inference=frame.inference, show=animation.show_model),
                    *error_annotations(
                        X=frame.X,
                        targets=view.targets,
                        preds=view.preds,
                        focused_errors=view.focused_errors,
                        show=animation.show_model and frame.focused_feature is None,
                    ),
                    *feature_annotations(
                        X=frame.X,
                        targets=view.targets,
                        preds=view.preds,
                        focused_feature=frame.focused_feature,
                        feature_colors=feature_colors,
                        theme=animation.theme,
                        show=animation.show_model and frame.focused_feature is not None,
                    ),
                ],
            ),
            scene2=dict(
                camera=dict(eye=weight_eyes[0]),
                xaxis=dict(backgroundcolor=feature_colors[1]),
                yaxis=dict(backgroundcolor=feature_colors[0], range=frame.get_range(dim=1, pad=True)),
            ),
            scene3=dict(
                camera=dict(eye=weight_eyes[1]),
                xaxis=dict(backgroundcolor=feature_colors[1], range=frame.get_range(dim=0, pad=True)),
                yaxis=dict(backgroundcolor=feature_colors[0]),
            ),
            scene4=dict(
                camera=dict(eye=bias_eye),
                xaxis=dict(backgroundcolor=feature_colors[1]),
                yaxis=dict(backgroundcolor=feature_colors[0]),
                zaxis=dict(range=frame.get_bias_zrange(pad=True)),
            ),
        ),
    )


def animate(
    frames: list[Frame],
    model_node="output_1",
    show_model=True,
    show_components=True,
    show_tables=False,
    show_network=False,
    scale=2,
    render_path=None,
    theme=default_theme,
):
    views = {}
    specs = []

    height = 0
    width = 1080

    row_count = 0
    row_heights = []
    stage_heights = {}

    model_height = 768
    components_height = 250
    losses_height = 768
    costs_height = 128
    network_height = 1152

    if show_model:
        if "model" not in stage_heights:
            stage_heights["model"] = model_height + components_height
            row_heights.extend([model_height, components_height])
            specs.append([dict(type="scene", colspan=3), None, None])
            specs.append([dict(type="scene"), dict(type="scene"), dict(type="scene")])

        row_count += 1

        views["model"] = dict(
            name="model",
            stage="model",
            row=row_count,
            col=1,
            height=768,
        )

        if not show_components:
            row_count += 1

    if show_components:
        if "model" not in stage_heights:
            stage_heights["model"] = model_height + components_height
            row_heights.extend([model_height, components_height])
            specs.append([dict(type="scene", colspan=3), None, None])
            specs.append([dict(type="scene"), dict(type="scene"), dict(type="scene")])

        if not show_model:
            row_count += 1

        row_count += 1

        views["component1"] = dict(
            name="component1",
            stage="model",
            row=row_count,
            col=1,
            height=components_height,
        )

        views["component2"] = dict(
            name="component2",
            stage="model",
            row=row_count,
            col=2,
            height=components_height,
        )

        views["component3"] = dict(
            name="component3",
            stage="model",
            row=row_count,
            col=3,
            height=components_height,
        )

    if show_tables:
        stage_heights["tables"] = losses_height + costs_height
        row_heights.extend([losses_height, costs_height])

        row_count += 1

        views["losses"] = dict(
            name="losses",
            stage="tables",
            row=row_count,
            col=1,
            height=losses_height,
        )

        row_count += 1

        views["cost"] = dict(
            name="cost",
            stage="tables",
            row=row_count,
            col=1,
            height=costs_height,
        )

        height += losses_height + costs_height

        specs.append([dict(type="table", colspan=3), None, None])
        specs.append([dict(type="table", colspan=3), None, None])

    if show_network:
        stage_heights["network"] = network_height
        row_heights.append(network_height)

        row_count += 1

        views["network"] = dict(
            name="network",
            stage="network",
            row=row_count,
            col=1,
            height=network_height,
        )

        specs.append([dict(type="scatter", colspan=3), None, None])

    height = sum(stage_heights.values())

    controls_height = 130

    if not render_path:
        height += controls_height

    animation = Animation(
        frames=frames,
        model_node=model_node,
        render_path=render_path,
        show_model=show_model,
        show_network=show_network,
        show_components=show_components,
        show_calculations=show_tables,
        height=height,
        width=width,
        marker_size=30,
        line_width=10,
        scale=scale,
        theme=theme,
        meta=views,
    )

    frame = frames[0]
    view = animation.node_view(frame)

    # model_node = animation.node_view(frame)

    fig = make_subplots(
        rows=row_count,
        cols=3,
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
        font=dict(family="Comic Sans MS, Droid Sans, sans-serif", size=24),
        # annotations=[*weight_annotations(model_node, height, None, None, show=show_model, theme=theme)],
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showticklabels=False),
    )

    feature_colors = theme.feature_colors

    scene = dict(
        camera=dict(eye=frame.get_eye(), projection=dict(type="orthographic")),
        aspectmode="manual",
        aspectratio=frame.get_aspect_ratio(),
        xaxis_title="",
        yaxis_title="",
        zaxis_title="",
        xaxis=dict(backgroundcolor=feature_colors[1], range=frame.get_range(dim=0, pad=True)),
        yaxis=dict(backgroundcolor=feature_colors[0], range=frame.get_range(dim=1, pad=True)),
        zaxis=dict(
            backgroundcolor=theme.target_color,
            range=frame.get_zrange(pad=True),
            tickvals=[0, 0.5, 1],
        ),
    )

    fig.for_each_scene(lambda x: x.update(scene))

    if show_components:
        zoom = 3

        weight_eyes = view.get_weight_eyes()
        bias_eye = frame.get_bias_eye()

        fig.layout.scene2.update(
            dict(aspectratio=dict(x=zoom, y=zoom, z=zoom / 2)),
            camera=dict(eye=weight_eyes[0]),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(title="", showgrid=False, showticklabels=False, range=frame.get_range(dim=1, pad=True)),
            zaxis=dict(title="", showgrid=False, showticklabels=False),
        )

        fig.layout.scene3.update(
            dict(aspectratio=dict(x=zoom, y=zoom, z=zoom / 2)),
            camera=dict(eye=weight_eyes[1]),
            xaxis=dict(title="", showgrid=False, showticklabels=False, range=frame.get_range(dim=0, pad=True)),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(title="", showgrid=False, showticklabels=False),
        )

        fig.layout.scene4.update(
            dict(aspectratio=dict(x=zoom / 1.4, y=zoom / 1.4, z=zoom / 2)),
            camera=dict(eye=bias_eye),
            xaxis=dict(title="", showgrid=False, showticklabels=False),
            yaxis=dict(title="", showgrid=False, showticklabels=False),
            zaxis=dict(showticklabels=False, range=frame.get_bias_zrange(pad=True)),
        )

    def update_network():
        view = views["network"]
        fig.update_yaxes(range=[-2, 0], row=view["row"], col=view["col"])
        fig.update_xaxes(range=[-1.5, 1.5], row=view["row"], col=view["col"])

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

        control_args = dict(frame=dict(duration=50, redraw=True), transition=dict(duration=0), mode="immediate")

        sliders = [
            dict(
                steps=[
                    dict(label=i, method="animate", args=[[i, control_args], dict(mode="immediate")])
                    for i, _ in enumerate(fig.frames)
                ]
            )
        ]

        menu = dict(
            type="buttons",
            xanchor="center",
            x=0.5,
            y=0,
            buttons=[dict(label="Play", method="animate", args=[None, control_args])],
        )

        fig.update_layout(updatemenus=[menu], sliders=sliders)

        fig.show(config=dict(scrollZoom=False))
