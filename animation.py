import os

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import tqdm


from annotations.neural_network import neural_network_annotations
from themes import default_theme, themes
from annotations.scene import (
    cost_annotation,
    prediction_annotations,
    feature_annotations,
    inference_annotation,
    weight_annotations,
)
from traces.scene import data_markers, gradient, model_surface

from traces.neural_network import activations, neural_network
from traces.tables import cost_tables

from base import Frame, Animation, View

transparent = "rgba(255,255,255,.1)"


def scene_traces(
    view: View,
    frame: Frame,
    show_profile,
    focused_feature,
    show_preds=True,
    show_targets=True,
    marker_size=30,
    profile_line_width=6,
    theme=default_theme,
    meta=None,
):
    modules = view.modules
    output = modules[-1]

    return [
        *data_markers(
            X=frame.X,
            w=view.w[output][0],
            b=view.b[output][0],
            preds=view.preds,
            targets=view.targets,
            inference=frame.inference,
            focused_feature=focused_feature,
            focused_errors=view.focused_errors,
            focus_targets=frame.focus_targets,
            show_preds=show_preds,
            show_targets=show_targets,
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
            modules=view.modules,
            activations=frame.activations,
            activity=view.activity,
            show_profile=show_profile,
            profile_line_width=profile_line_width,
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
    show_weights = animation.show_weights

    if not show_weights:
        return []

    scene_args = dict(
        marker_size=animation.marker_size * 0.666,
        show_profile=True,
        focused_feature=None,
        show_preds=animation.show_component_preds,
        show_targets=animation.show_component_preds,
        profile_line_width=animation.component_line_width,
        theme=animation.theme,
    )

    value = []

    for i in show_weights:
        if isinstance(i, int):
            name = f"weight{i}"
        else:
            name = "bias"

        meta = animation.meta[name]
        view = animation.node_view(frame, component=i)

        value += scene_traces(
            view=view,
            frame=frame,
            meta=dict(row=meta["row"], col=meta["col"]),
            **scene_args,
        )

    return value


def gradients_view(view: View, frame: Frame, animation: Animation):
    if not animation.show_gradients:
        return []

    meta = animation.meta["gradients"]

    scene = gradient(
        frame.X,
        frame.targets,
        frame.surface_points,
        frame.surface_linspace,
        frame.w,
        frame.b,
        frame.modules,
        frame.activations,
        param1=["output_1", 0],
        param2=["output_1", 1],
        show_profile=False,
        res=20,
        meta=meta,
    )

    return scene


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
        ),
        *activations(
            frame,
            animation,
            meta=dict(row=meta["row"], col=meta["col"]),
        ),
    ]


def get_colors(frame: Frame, view: View, animation: Animation):
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


def make_frame(frame: Frame, animation: Animation, name: str):
    if frame.eye:
        eye = dict(x=frame.eye[0], y=frame.eye[1], z=frame.eye[2])
    else:
        eye = None

    view = animation.node_view(frame)

    output_module = view.modules[-1]
    input_module = view.modules[-2]
    input_size = frame.size[input_module]

    show_label_names = animation.show_label_names
    show_bg = animation.show_bg

    output_colors, feature_colors = get_colors(frame, view, animation)

    weight_eyes = view.get_weight_eyes(as_dict=True)
    bias_eye = frame.get_bias_eye(as_dict=True)

    w = view.w[output_module][0]
    b = view.b[output_module][0]

    value = go.Frame(
        name=name,
        data=[
            *model_view(view, frame, animation),
            *components_view(frame, animation),
            *tables_view(view, frame, animation),
            *network_view(frame, animation),
        ],
        layout=dict(
            annotations=[
                *cost_annotation(
                    cost=frame.loss,
                    width=animation.width,
                    show_label_names=show_label_names,
                    label_precision=animation.label_precision,
                    label_yshift=animation.cost_label_yshift,
                    label_xshift=animation.cost_label_xshift,
                    label_xanchor=animation.cost_label_xanchor,
                    label_font_size=animation.label_font_size,
                    theme=animation.theme,
                    show=animation.show_model,
                ),
                *weight_annotations(
                    w=w,
                    b=b,
                    width=animation.width,
                    feature_colors=feature_colors,
                    show_label_names=show_label_names,
                    label_precision=animation.label_precision,
                    label_yshift=animation.label_yshift,
                    label_font_size=animation.label_font_size,
                    theme=animation.theme,
                    show=animation.show_model,
                ),
                *neural_network_annotations(frame, animation, show=animation.show_network),
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
                    backgroundcolor=output_colors[0] if show_bg else transparent,
                    range=frame.get_range(dim=0, pad=True),
                ),
                yaxis=dict(
                    # color=animation.theme.feature_text_colors[1],
                    backgroundcolor=output_colors[-1] if show_bg else transparent,
                    range=frame.get_range(dim=1, pad=True),
                ),
                zaxis=dict(range=frame.get_zrange(pad=True)),
                annotations=[
                    *inference_annotation(
                        w=w,
                        b=b,
                        inference=frame.inference,
                        theme=animation.theme,
                        show=animation.show_model,
                    ),
                    *prediction_annotations(
                        X=frame.X,
                        targets=view.targets,
                        preds=view.preds,
                        focused_errors=view.focused_errors,
                        theme=animation.theme,
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
        ),
    )

    if animation.show_weights:
        for i in animation.show_weights:
            if isinstance(i, int):
                value.layout[f"scene{i + 1}"] = dict(
                    camera=dict(eye=weight_eyes[i - 1]),
                    xaxis=dict(backgroundcolor=feature_colors[i - 1]),
                    yaxis=dict(
                        backgroundcolor=feature_colors[i - 1],
                        range=frame.get_range(dim=1, pad=True),
                    ),
                )

            if i == "b":
                value.layout[f"scene{input_size + 2}"] = dict(
                    camera=dict(eye=bias_eye),
                    xaxis=dict(backgroundcolor=feature_colors[-1]),
                    yaxis=dict(backgroundcolor=feature_colors[0]),
                    zaxis=dict(range=frame.get_bias_zrange(pad=True)),
                )

    return value


def animate(
    frames: list[Frame],
    model_node="output_1",
    show_model=True,
    show_weights=False,
    show_gradients=False,
    show_tables=False,
    show_network=False,
    show_bg=True,
    show_label_names=True,
    label_precision=3,
    label_yshift=0,
    cost_label_yshift=0,
    cost_label_xshift=0,
    cost_label_xanchor="left",
    label_font_size=40,
    component_line_width=6,
    show_component_preds=False,
    lock_scroll=True,
    scale=1,
    render_path=None,
    theme=default_theme,
):
    views = {}
    specs = []

    if isinstance(theme, str):
        theme = themes[theme]

    animation = Animation(
        frames=frames,
        model_node=model_node,
        render_path=render_path,
        show_model=show_model,
        show_weights=show_weights,
        show_gradients=show_gradients,
        show_network=show_network,
        show_calculations=show_tables,
        show_label_names=show_label_names,
        label_precision=label_precision,
        label_yshift=label_yshift,
        cost_label_xshift=cost_label_xshift,
        cost_label_yshift=cost_label_yshift,
        cost_label_xanchor=cost_label_xanchor,
        label_font_size=label_font_size,
        marker_size=30,
        line_width=10,
        component_line_width=component_line_width,
        show_bg=show_bg,
        show_component_preds=show_component_preds,
        scale=scale,
        theme=theme,
        meta=views,
    )

    frame = frames[0]
    view = animation.node_view(frame)

    input_module = view.modules[-2]
    input_size = frame.size[input_module]

    if show_weights is True:
        show_weights = list(range(1, input_size + 1)) + ["b"]

    animation.show_weights = show_weights

    height = 0
    width = 1080 * scale

    row_count = 0
    row_heights = []
    stage_heights = {}

    model_height = 768 * scale
    components_height = 250 * scale
    gradients_height = 768 * scale
    losses_height = 768 * scale
    costs_height = 128 * scale
    network_height = 1152 * scale

    # add one for bias
    components_size = input_size + 1

    if show_model:
        stage_heights["model"] = model_height
        row_heights.append(model_height)
        specs.append([dict(type="scene", colspan=components_size)] + [None] * (components_size - 1))

        row_count += 1

        views["model"] = dict(
            name="model",
            stage="model",
            row=row_count,
            col=1,
            height=768,
        )

    if show_weights:
        if "model" not in stage_heights:
            stage_heights["model"] = components_height
        else:
            stage_heights["model"] += components_height

        row_heights.append(components_height)

        row_count += 1

        for weight in show_weights:
            if isinstance(weight, int):
                name = f"weight{weight}"
            elif weight == "b":
                name = "bias"
            else:
                raise ValueError(f"Invalid weight: {weight}")

            col = weight if isinstance(weight, int) else input_size + 1

            views[name] = dict(
                name=name,
                stage="model",
                row=row_count,
                col=col,
                height=components_height,
            )

        specs.append([dict(type="scene")] * components_size)

    if show_gradients:
        stage_heights["gradients"] = gradients_height
        row_heights.append(gradients_height)

        row_count += 1

        views["gradients"] = dict(
            name="gradients",
            stage="gradients",
            row=row_count,
            col=1,
            height=gradients_height,
        )

        specs.append([dict(type="scene", colspan=components_size)] + [None] * (components_size - 1))

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

        specs.append([dict(type="table", colspan=components_size)] + [None] * (components_size - 1))
        specs.append([dict(type="table", colspan=components_size)] + [None] * (components_size - 1))

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

        specs.append([dict(type="scatter", colspan=components_size)] + [None] * (components_size - 1))

    height = sum(stage_heights.values())

    controls_height = 130

    if not render_path:
        height += controls_height

    output_colors, feature_colors = get_colors(frame, view, animation)

    animation.width = width
    animation.height = height
    animation.meta = views

    # model_node = animation.node_view(frame)

    fig = make_subplots(
        rows=row_count,
        cols=components_size,
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
            family="Comic Sans MS, Droid Sans, sans-serif",
            size=24,
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

    scene = dict(
        camera=dict(eye=frame.get_eye(as_dict=True), projection=dict(type="orthographic")),
        aspectmode="manual",
        aspectratio=frame.get_aspect_ratio(as_dict=True),
        xaxis_title="",
        yaxis_title="",
        zaxis_title="",
        xaxis=dict(
            color=theme.feature_text_colors[0],
            gridcolor=theme.feature_grid_colors[0],
            backgroundcolor=output_colors[-1] if show_bg else transparent,
            range=frame.get_range(dim=0, pad=True),
        ),
        yaxis=dict(
            color=theme.feature_text_colors[1],
            gridcolor=theme.feature_grid_colors[1],
            backgroundcolor=output_colors[0] if show_bg else transparent,
            range=frame.get_range(dim=1, pad=True),
        ),
        zaxis=dict(
            color=theme.target_text_color,
            gridcolor=theme.target_grid_color,
            backgroundcolor=theme.target_color if show_bg else transparent,
            range=frame.get_zrange(pad=True),
            tickvals=[0, 0.5, 1],
        ),
    )

    fig.for_each_scene(lambda x: x.update(scene))

    if show_weights:
        zoom = 3

        weight_eyes = view.get_weight_eyes(as_dict=True)
        bias_eye = frame.get_bias_eye(as_dict=True)

        for i in show_weights:
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

        fig.show(config=dict(scrollZoom=not lock_scroll))
