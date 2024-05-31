import torch
import os

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dataclasses import replace

import tqdm


from themes import default_theme
from annotations import (
    loss_annotations,
    feature_annotations,
    inference_annotation,
    weight_annotations,
)
from traces import data_markers, model_surface, loss_tables, neural_network

from base import Frame, Animation


def scene_traces(
    frame: Frame,
    marker_size=30,
    line_width=10,
    meta=None,
    theme=default_theme,
):
    # hidden_w = frame.hidden_w
    # hidden_b = frame.hidden_b
    # w = frame.w
    # b = frame.b

    # if frame.show_nodes is not None:
    #     if hidden_w is not None and hidden_b is not None and frame.show_nodes[0] is not None:
    #         hidden_size = hidden_w.size(0)
    #         hidden_w = hidden_w * torch.Tensor([frame.show_nodes[0] for _ in range(hidden_size)]).T
    #         hidden_b = hidden_b * torch.Tensor(frame.show_nodes[0]).unsqueeze(1)

    #     if frame.show_nodes[1] is not None:
    #         output_size = w.size(0)
    #         w = w * torch.Tensor([frame.show_nodes[1] for _ in range(output_size)]).T
    #         b = b * torch.Tensor(frame.show_nodes[1])

    # if frame.inject_weights is not None:
    #     for i, weights in enumerate(frame.inject_weights[0]):
    #         if weights:
    #             hidden_w[i] = torch.Tensor(weights)

    #     if frame.inject_weights[1] is not None:
    #         w[0] = torch.Tensor(frame.inject_weights[1])

    # if frame.inject_biases is not None:
    #     for i, biases in enumerate(frame.inject_biases[0]):
    #         if biases:
    #             hidden_b[i] = biases

    #     if frame.inject_biases[1] is not None:
    #         b[0] = torch.Tensor(frame.inject_biases[1])

    return [
        *data_markers(
            frame,
            marker_size=marker_size,
            line_width=line_width,
            standoff=0.033,
            meta=meta,
            theme=theme,
        ),
        *model_surface(
            frame,
            # hidden_w=hidden_w,
            # hidden_b=hidden_b,
            # w=w,
            # b=b,
            meta=meta,
        ),
    ]


def main_traces(frame: Frame, animation: Animation):
    if not animation.show_main:
        return []

    marker_size = animation.marker_size
    line_width = animation.line_width
    theme = animation.theme

    row = animation.rows["main"]

    return scene_traces(
        frame,
        marker_size=marker_size,
        line_width=line_width,
        theme=theme,
        meta=dict(row=row, col=1),
    )


def component_traces(frame: Frame, animation: Animation):
    show_components = animation.show_components

    if not show_components:
        return []

    row = animation.rows["components"]

    w = frame.w
    hidden_w = frame.hidden_w
    hidden_b = frame.hidden_b

    w1, w2, bias = [], [], []

    if isinstance(show_components, bool):
        show_w1, show_w2, show_bias = (True, True, True)
    else:
        show_w1, show_w2, show_bias = show_components

    marker_size = animation.marker_size / 2
    line_width = animation.line_width / 2

    if show_w1:
        if hidden_w is not None:
            frame = replace(
                frame,
                hidden_w=torch.Tensor([[1, 0], [1, 0]]).T * hidden_w,
                hidden_b=torch.Tensor([[1], [0]]) * hidden_b,
                focused_feature=None,
                surface_color="black",
                planarity=0,
            )
        else:
            frame = replace(frame, w=torch.Tensor([1, 0]) * w, focused_feature=None, surface_color="black", planarity=1)

        w1 = scene_traces(
            frame,
            marker_size=marker_size,
            line_width=line_width,
            meta=dict(row=row, col=1),
        )

    if show_w2:
        if hidden_w is not None:
            frame = replace(
                frame,
                hidden_w=torch.Tensor([[0, 1], [0, 1]]).T * hidden_w,
                hidden_b=torch.Tensor([[0], [1]]) * hidden_b,
                focused_feature=None,
                surface_color="black",
                planarity=0,
            )
        else:
            frame = replace(frame, w=torch.Tensor([0, 1]) * w, focused_feature=None, surface_color="black", planarity=1)

        w2 = scene_traces(
            frame,
            marker_size=marker_size,
            line_width=line_width,
            meta=dict(row=row, col=2),
        )

    if show_bias:
        if hidden_w is not None:
            frame = replace(
                frame,
                w=torch.Tensor([[0, 0]]),
                hidden_w=None,
                hidden_b=None,
                focused_feature=None,
                surface_color="black",
                planarity=0,
            )
        else:
            frame = replace(frame, w=torch.Tensor([[0, 0]]), focused_feature=None, surface_color="black", planarity=1)

        bias = scene_traces(
            frame,
            marker_size=marker_size,
            line_width=line_width,
            meta=dict(row=row, col=3),
        )

    return [*w1, *w2, *bias]


def network_traces(frame: Frame, animation: Animation):
    if not animation.show_network:
        return []

    row = animation.rows["network"]

    return [
        *neural_network(
            frame,
            animation,
            meta=dict(row=row, col=1),
        )
    ]


def calculation_traces(frame: Frame, animation: Animation):
    if not animation.show_calculations:
        return []

    losses_row = animation.rows["losses"]
    cost_row = animation.rows["cost"]

    return loss_tables(
        frame,
        theme=animation.theme,
        meta=(dict(row=losses_row, col=1), dict(row=cost_row, col=1)),
    )


def make_frame(frame: Frame, animation: Animation, name: str):
    feature_colors = frame.focusable_feature_colors(theme=animation.theme)

    if frame.eye:
        eye = dict(x=frame.eye[0], y=frame.eye[1], z=frame.eye[2])
    else:
        eye = None

    weight_eyes = frame.get_weight_eyes()
    bias_eye = frame.get_bias_eye()

    return go.Frame(
        name=name,
        data=[
            *main_traces(frame, animation),
            *network_traces(frame, animation),
            *component_traces(frame, animation),
            *calculation_traces(frame, animation),
        ],
        layout=dict(
            annotations=[
                *weight_annotations(
                    frame.w,
                    frame.b,
                    animation.height,
                    frame.focused_feature,
                    frame.focus_labels,
                    visible=animation.show_main,
                    theme=animation.theme,
                )
            ],
            scene=dict(
                camera=dict(eye=eye),
                aspectratio=dict(x=frame.aspect_ratio[0], y=frame.aspect_ratio[1], z=frame.aspect_ratio[2]),
                xaxis_title="",
                yaxis_title="",
                zaxis_title="",
                xaxis=dict(backgroundcolor=feature_colors[1]),
                yaxis=dict(backgroundcolor=feature_colors[0]),
                zaxis=dict(range=frame.get_zrange(pad=True)),
                annotations=[
                    *inference_annotation(frame, visible=animation.show_main),
                    *loss_annotations(
                        frame,
                        visible=animation.show_main and frame.focused_feature is None,
                    ),
                    *feature_annotations(
                        frame,
                        visible=animation.show_main and frame.focused_feature is not None,
                        theme=animation.theme,
                    ),
                ],
            ),
            scene2=dict(
                camera=dict(eye=weight_eyes[0]),
                xaxis=dict(backgroundcolor=feature_colors[1]),
                yaxis=dict(backgroundcolor=feature_colors[0]),
            ),
            scene3=dict(
                camera=dict(eye=weight_eyes[1]),
                xaxis=dict(backgroundcolor=feature_colors[1]),
                yaxis=dict(backgroundcolor=feature_colors[0]),
            ),
            scene4=dict(
                camera=dict(eye=bias_eye),
                xaxis=dict(backgroundcolor=feature_colors[1]),
                yaxis=dict(backgroundcolor=feature_colors[0]),
            ),
        ),
    )


def animate(
    frames: list[Frame],
    output_folder=None,
    show_main=True,
    show_components=True,
    show_calculations=True,
    show_network=True,
    scale=2,
    theme=default_theme,
):
    rows = dict(main=1, components=2, losses=3, cost=4, network=5)

    main_height = 768
    components_height = 250
    losses_height = 768
    cost_height = 128
    network_height = 1080

    controls_height = 130

    row_heights = [main_height, components_height, losses_height, cost_height, network_height]

    width = 1080
    height = sum(row_heights)

    if not output_folder:
        height += controls_height

        if not show_main:
            height -= main_height
            row_heights[rows["main"] - 1] = 0

        if not show_components:
            height -= components_height
            row_heights[rows["components"] - 1] = 0

        if not show_calculations:
            height -= losses_height + cost_height
            row_heights[rows["losses"] - 1] = 0
            row_heights[rows["cost"] - 1] = 0

        if not show_network:
            height -= network_height
            row_heights[rows["network"] - 1] = 0

    animation = Animation(
        frames=frames,
        output_folder=output_folder,
        show_main=show_main,
        show_network=show_network,
        show_components=show_components,
        show_calculations=show_calculations,
        rows=rows,
        height=height,
        width=width,
        marker_size=30,
        line_width=10,
        scale=scale,
        theme=theme,
    )

    specs = [
        [dict(type="scene", colspan=3), None, None],
        [dict(type="scene"), dict(type="scene"), dict(type="scene")],
        [dict(type="table", colspan=3), None, None],
        [dict(type="table", colspan=3), None, None],
        [dict(type="scatter", colspan=3), None, None],
    ]

    frame = frames[0]

    fig = make_subplots(
        rows=len(row_heights),
        cols=max([len(row) for row in specs]),
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
        annotations=[*weight_annotations(frame.w, frame.b, height, None, None, visible=show_main, theme=theme)],
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

        weight_eyes = frame.get_weight_eyes()
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
        fig.update_yaxes(range=[-2, 0], row=rows["network"], col=1)
        fig.update_xaxes(range=[-1.5, 1.5], row=rows["network"], col=1)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

        frame = make_frame(frames[0], animation, name=0)

        for trace in frame.data:
            fig.add_trace(trace=trace, row=trace.meta["row"], col=trace.meta["col"])

        zlen = len(str(len(frames)))

        # fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        if show_network:
            update_network()

        for i, frame in enumerate(tqdm.tqdm(frames)):
            frame = make_frame(frame, animation, name=i)

            for trace1, trace2 in zip(fig.data, frame.data):
                trace1.update(trace2)

            fig.update_layout(frame.layout)

            num = str(i).zfill(zlen)

            fig.write_image(os.path.join(output_folder, f"{num}.png"), format="png", scale=scale)

    else:
        frames = [make_frame(frame, animation, name=i) for i, frame in enumerate(frames)]

        for trace in frames[0].data:
            fig.add_trace(trace=trace, row=trace.meta["row"], col=trace.meta["col"])

        fig.update(frames=frames)

        if show_network:
            update_network()

        sliders = [
            dict(
                steps=[
                    dict(label=i, method="animate", args=[[i], dict(mode="immediate")])
                    for i, _ in enumerate(fig.frames)
                ]
            )
        ]

        menu = dict(
            type="buttons",
            xanchor="center",
            x=0.5,
            y=0,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(frame=dict(duration=50, redraw=True), mode="immediate"),
                    ],
                )
            ],
        )

        fig.update_layout(updatemenus=[menu], sliders=sliders)

        fig.show(config=dict(scrollZoom=False))
