import torch
import os

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dataclasses import replace

import tqdm


from themes import default_theme
from annotations import (
    loss_annotations,
    focusable_feature_colors,
    feature_annotations,
    inference_annotation,
    weight_annotations,
)
from traces import data_markers, model_surface, loss_tables

from frame import Frame


def animate(
    frames,
    output_folder=None,
    show_stage=True,
    show_components=True,
    show_calculations=True,
    scale=2,
    theme=default_theme,
):
    height = 1920
    width = 1080

    frame = frames[0]

    def scene_traces(
        frame: Frame,
        marker_size=30,
        line_width=10,
        meta=None,
    ):
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
                meta=meta,
            ),
        ]

    def stage_traces(frame: Frame):
        if not show_stage:
            return []

        return scene_traces(frame, meta=dict(row=1, col=1))

    def component_traces(frame: Frame):
        if not show_components:
            return []

        w = frame.w

        w1, w2, bias = [], [], []

        if isinstance(show_components, bool):
            show_w1, show_w2, show_bias = (True, True, True)
        else:
            show_w1, show_w2, show_bias = show_components

        args = dict(marker_size=16, line_width=5)

        if show_w1:
            w1 = scene_traces(
                replace(frame, w=torch.Tensor([1, 0]) * w, focused_feature=None, show_surface=False, planarity=1),
                **args,
                meta=dict(row=2, col=1),
            )

        if show_w2:
            w2 = scene_traces(
                replace(frame, w=torch.Tensor([0, 1]) * w, focused_feature=None, show_surface=False, planarity=1),
                **args,
                meta=dict(row=2, col=2),
            )

        if show_bias:
            bias = scene_traces(
                replace(frame, w=torch.Tensor([0, 0]) * w, focused_feature=None, show_surface=False, planarity=1),
                **args,
                meta=dict(row=2, col=3),
            )

        return [*w1, *w2, *bias]

    def calculation_traces(frame: Frame):
        if not show_calculations:
            return []

        return loss_tables(
            frame,
            meta=(dict(row=3, col=1), dict(row=4, col=1)),
            theme=theme,
        )

    def make_frame(frame: Frame, name=None):
        feature_colors = focusable_feature_colors(frame.focused_feature, theme=theme)

        return go.Frame(
            name=name,
            data=[
                *stage_traces(frame),
                *component_traces(frame),
                *calculation_traces(frame),
            ],
            layout=dict(
                annotations=[
                    *weight_annotations(
                        frame.w, height, frame.focused_feature, frame.focus_labels, visible=show_stage, theme=theme
                    )
                ],
                scene=dict(
                    camera=dict(eye=dict(x=frame.eye[0], y=frame.eye[1], z=frame.eye[2])),
                    aspectratio=dict(x=frame.aspect_ratio[0], y=frame.aspect_ratio[1], z=frame.aspect_ratio[2]),
                    xaxis_title="",
                    yaxis_title="",
                    zaxis_title="",
                    xaxis=dict(backgroundcolor=feature_colors[1]),
                    yaxis=dict(backgroundcolor=feature_colors[0]),
                    zaxis=dict(range=frame.zrange),
                    annotations=[
                        *inference_annotation(frame, visible=show_stage),
                        *loss_annotations(
                            frame,
                            visible=show_stage and frame.focused_feature is None,
                        ),
                        *feature_annotations(
                            frame,
                            visible=show_stage and frame.focused_feature is not None,
                            theme=theme,
                        ),
                    ],
                ),
                scene2=dict(
                    xaxis=dict(backgroundcolor=feature_colors[1]),
                    yaxis=dict(backgroundcolor=feature_colors[0]),
                ),
                scene3=dict(
                    xaxis=dict(backgroundcolor=feature_colors[1]),
                    yaxis=dict(backgroundcolor=feature_colors[0]),
                ),
                scene4=dict(
                    xaxis=dict(backgroundcolor=feature_colors[1]),
                    yaxis=dict(backgroundcolor=feature_colors[0]),
                ),
            ),
        )

    specs = [
        [dict(type="scene", colspan=3), None, None],
        [dict(type="scene"), dict(type="scene"), dict(type="scene")],
        [dict(type="table", colspan=3), None, None],
        [dict(type="table", colspan=3), None, None],
    ]

    row_heights = [14, 4.5, 14, 2.333]

    if not output_folder:
        height += 130

        if not show_stage:
            height -= 1920 - (280 + 800)
            row_heights[0] = 0

        if not show_components:
            height -= 280
            row_heights[1] = 0

        if not show_calculations:
            height -= 800
            row_heights[2] = 0
            row_heights[3] = 0

    fig = make_subplots(
        rows=4,
        cols=3,
        specs=specs,
        row_heights=row_heights,
        horizontal_spacing=0,
        vertical_spacing=0,
    )

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        transition=dict(duration=0, easing="linear", ordering="traces first"),
        font=dict(family="Comic Sans MS, Droid Sans, sans-serif", size=24),
        annotations=[*weight_annotations(frame.w, height, None, None, visible=show_stage, theme=theme)],
    )

    feature_colors = theme.feature_colors

    scene = dict(
        camera=dict(eye=dict(x=frame.eye[0], y=frame.eye[1], z=frame.eye[2]), projection=dict(type="orthographic")),
        aspectmode="manual",
        aspectratio=dict(x=frame.aspect_ratio[0], y=frame.aspect_ratio[1], z=frame.aspect_ratio[2]),
        xaxis_title="",
        yaxis_title="",
        zaxis_title="",
        xaxis=dict(backgroundcolor=feature_colors[1], range=[frame.domain[0][0] - 0.2, frame.domain[0][1] + 0.2]),
        yaxis=dict(backgroundcolor=feature_colors[0], range=[frame.domain[1][0] - 0.2, frame.domain[1][1] + 0.2]),
        zaxis=dict(backgroundcolor=theme.target_color, range=frame.zrange, tickvals=[0, 0.5, 1]),
    )
    # xaxis=dict(backgroundcolor='rgb(218, 223, 229)', range=xrange or range),
    # yaxis=dict(backgroundcolor='rgb(242, 238, 232)', range=yrange or range),
    # zaxis=dict(backgroundcolor='rgb(240, 240 ,240)', range=zrange or range))

    fig.for_each_scene(lambda x: x.update(scene))

    zoom = 3
    if show_components:
        fig.layout.scene2.update(
            dict(aspectratio=dict(x=zoom, y=zoom, z=zoom / 2)),
            camera=dict(
                eye=dict(x=0, y=1, z=0),
            ),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(
                title="",
                showgrid=False,
                showticklabels=False,
                range=[frame.domain[1][0] - 0.05, frame.domain[1][1] + 0.05],
            ),
            zaxis=dict(title="", showgrid=False, showticklabels=False),
        )

        fig.layout.scene3.update(
            dict(aspectratio=dict(x=zoom, y=zoom, z=zoom / 2)),
            camera=dict(
                eye=dict(x=1, y=0, z=0),
            ),
            xaxis=dict(
                title="",
                showgrid=False,
                showticklabels=False,
                range=[frame.domain[0][0] - 0.05, frame.domain[0][1] + 0.05],
            ),
            yaxis=dict(showgrid=False, showticklabels=False),
            zaxis=dict(title="", showgrid=False, showticklabels=False),
        )

        fig.layout.scene4.update(
            dict(aspectratio=dict(x=zoom / 1.4, y=zoom / 1.4, z=zoom / 2)),
            camera=dict(
                eye=dict(x=1, y=1, z=0),
            ),
            xaxis=dict(title="", showgrid=False, showticklabels=False),
            yaxis=dict(title="", showgrid=False, showticklabels=False),
            zaxis=dict(showticklabels=False, range=[-5.1, 5.1]),
        )

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

        frame = make_frame(frames[0], 0)

        for trace in frame.data:
            fig.add_trace(trace=trace, row=trace.meta["row"], col=trace.meta["col"])

        zlen = len(str(len(frames)))

        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        for i, frame in enumerate(tqdm.tqdm(frames)):
            frame = make_frame(frame, i)

            for trace1, trace2 in zip(fig.data, frame.data):
                trace1.update(trace2)

            fig.update_layout(frame.layout)

            num = str(i).zfill(zlen)

            fig.write_image(os.path.join(output_folder, f"{num}.png"), format="png", scale=scale)

    else:
        frames = [make_frame(frame, i) for i, frame in enumerate(frames)]

        for trace in frames[0].data:
            fig.add_trace(trace=trace, row=trace.meta["row"], col=trace.meta["col"])

        fig.update(frames=frames)

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
