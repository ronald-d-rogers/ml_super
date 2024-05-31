import numpy as np
import torch
import plotly.graph_objects as go
from base import Frame, Animation
from learning import predict
from themes import default_theme
from utils import hex_to_rgb, interp_rgb, rgb_to_str


default_marker_size = 30


def loss_lines(X, preds, targs, width=10, opacity=1, line=None, visible=True, meta=None):
    if not line:
        line = dict(color="white", width=5, dash="dot")

    xs, ys, zs = [], [], []

    if X.shape[0]:
        for x, y, p, t in torch.stack((X[:, 0], X[:, 1], preds, targs), dim=1):
            xs.extend([x, x, None])
            ys.extend([y, y, None])
            zs.extend([t, p, None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        opacity=opacity,
        line=line,
        visible=visible,
        meta=meta,
    )


def feature_lines(
    X, preds, targs, focused_inputs=None, marker_size=default_marker_size, visible=True, meta=None, theme=default_theme
):
    m = X.shape[0]
    xs, ys, zs = [], [], []

    if m:
        for x, y, p, t in torch.stack((X[:, 0], X[:, 1], preds, targs), dim=1):
            t = int(t)
            xs.extend([x, x if focused_inputs == 1 else 0, None])
            ys.extend([y, y if focused_inputs == 0 else 0, None])
            zs.extend([p, p, None])

    color = theme.feature_colors[focused_inputs] if focused_inputs is not None else "black"

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+lines",
        marker=dict(size=marker_size / 2),
        marker_color=color,
        line=dict(color=color, width=10),
        visible=visible,
        meta=meta,
    )


def target_markers(
    X, preds, targs, indices, focus=None, size=default_marker_size, standoff=None, meta=None, theme=default_theme
):
    if focus:
        marker_color = [theme.class_colors[bool(t > 0.5)] for t in targs]
        textfont_color = "black"

    else:
        # if pred close enough to target, marker_color is transparent, else black
        marker_color = "rgba(0, 0, 0, 0)"
        textfont_color = [
            "white" if abs(t - p) > 0.25 else "rgba(0, 0, 0, 0)" for t, p in torch.stack((targs, preds), dim=1)
        ]

    return go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=targs + standoff,
        mode="markers+text",
        marker=dict(size=size, line=dict(color="white")),
        # opacity=0.4 if not focus else 1,
        # marker_color="rgba(0, 0, 0, 0)",
        text=[f"x<sub>{i+1}</sub>" for i in indices],
        marker_color=marker_color,
        textfont_size=size,
        textfont_color=textfont_color,
        textposition="middle center",
        meta=meta,
    )


def pred_markers(
    X,
    preds,
    targs,
    indices,
    size=default_marker_size,
    standoff=0.033,
    opacity=1,
    visible=True,
    meta=None,
    theme=default_theme,
):
    if not visible:
        preds = torch.Tensor([])

    return go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=(preds + standoff),
        mode="markers+text",
        marker=dict(size=size, line=dict(color="black", width=1)),
        opacity=opacity,
        marker_color=[theme.class_colors[bool(t > 0.5)] for t in targs],
        text=[f"x<sub>{i+1}</sub>" for i in indices],
        textfont_size=size,
        textposition="middle center",
        meta=meta,
    )


def inference_marker(inference, w, b, size=default_marker_size, theme=default_theme, meta=None):
    if inference is not None:
        x, y = [inference[0]], [inference[1]]
        z = [torch.sigmoid((inference @ w.T) + b).item()]
        marker_color = [theme.class_colors[bool(z[0] > 0.5)]]
    else:
        x, y, z = [], [], []
        marker_color = []

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers+text",
        marker=dict(size=size, line=dict(color="black", width=1)),
        marker_color=marker_color,
        meta=meta,
    )


def data_markers(
    frame: Frame,
    marker_size=30,
    line_width=10,
    standoff=0.033,
    meta=None,
    theme=default_theme,
):
    X = frame.X
    w = frame.w
    b = frame.b
    preds = frame.preds[0]
    targets = frame.targets[0]
    inference = frame.inference[0] if frame.inference is not None else None
    focused_feature = frame.focused_feature
    focus_targets = frame.focus_targets

    m = X.shape[0]

    focused_inputs = frame.focused_inputs

    if frame.focused_inputs is None:
        focused_inputs = []

    if isinstance(frame.focused_inputs, int):
        focused_inputs = [frame.focused_inputs]

    traces = [loss_lines(frame.X, frame.preds[0], frame.targets[0], width=line_width, meta=meta)]

    foc = torch.Tensor([i in focused_inputs for i in range(m)])

    X_foc, X = X[foc == True], X[foc == False]  # noqa: E712
    preds_foc, preds = preds[foc == True], preds[foc == False]  # noqa: E712
    targs_foc, targs = targets[foc == True], targets[foc == False]  # noqa: E712

    indices = [i for i in range(m) if i not in focused_inputs]
    indices_foc = [i for i in range(m) if i in focused_inputs]

    show_preds = frame.show_preds
    preds_opacity = 0.4 if focus_targets else 1
    show_feature_lines = focused_feature is not None

    traces += [
        target_markers(X, preds, targs, indices, focus_targets, size=marker_size, standoff=standoff, meta=meta),
        feature_lines(
            X,
            preds,
            targs,
            focused_feature,
            marker_size=marker_size,
            visible=show_feature_lines,
            meta=meta,
            theme=theme,
        ),
        pred_markers(
            X,
            preds,
            targs,
            indices,
            size=marker_size,
            standoff=standoff,
            opacity=preds_opacity,
            visible=show_preds,
            meta=meta,
        ),
        inference_marker(inference, w, b, size=marker_size, meta=meta),
    ]

    X, preds, targs, indices = X_foc, preds_foc, targs_foc, indices_foc
    line = dict(color="red", width=10)

    traces += [
        loss_lines(X, preds, targs, width=line_width, line=line, meta=meta),
        feature_lines(
            X,
            preds,
            targs,
            focused_feature,
            marker_size=marker_size,
            visible=show_feature_lines,
            meta=meta,
            theme=theme,
        ),
        target_markers(X, preds, targs, indices, focus_targets, size=marker_size, standoff=standoff, meta=meta),
        pred_markers(
            X,
            preds,
            targs,
            indices,
            size=marker_size,
            standoff=standoff,
            opacity=preds_opacity,
            visible=show_preds,
            meta=meta,
        ),
    ]

    return traces


def decision_boundary(w: torch.Tensor, b: torch.Tensor, domain, visible=True, meta=None):
    w = w.squeeze()

    m1 = w[0] / -w[1]
    m2 = -w[1] / w[0]

    x1 = domain[0][0]
    y1 = (domain[0][0] * m1) - (b / w[1])

    if abs(y1) > abs(domain[1][0]):
        x1 = (domain[1][0] * m2) - (b / w[0])
        y1 = domain[1][0]

    x2 = domain[0][1]
    y2 = (domain[0][1] * m1) - (b / w[1])

    if abs(y2) > abs(domain[1][1]):
        x2 = (domain[1][1] * m2) - (b / w[0])
        y2 = domain[1][1]

    return go.Scatter3d(
        x=[x1, x2],
        y=[y1, y2],
        z=[0.5, 0.5],
        mode="lines",
        line=dict(width=20, color="black", dash="dot"),
        visible=visible,
        meta=meta,
    )


def model_surface(frame: Frame, meta=None):
    surface_points = frame.surface_points
    surface_linspace = frame.surface_linspace
    hidden_w = frame.hidden_w
    hidden_b = frame.hidden_b
    w = frame.w
    b = frame.b
    domain = frame.get_domain(pad=True)
    planarity = frame.planarity
    surface_color = frame.surface_color
    show_decision_boundaries = frame.show_decision_boundaries
    res = frame.resolution

    sx = surface_linspace[:, 0]
    sy = surface_linspace[:, 1]

    if hidden_w is not None:
        preds = predict(surface_points, w=hidden_w, b=hidden_b).T
        preds = predict(preds, w, b)
    else:
        preds = predict(surface_points, w, b, planarity=planarity)

    preds = torch.reshape(preds, (res, res))
    preds = torch.rot90(preds, 3)
    preds = torch.flip(preds, dims=[1])

    surface = go.Surface(
        x=sx,
        y=sy,
        z=preds,
        showscale=False,
        colorscale="plasma" if not surface_color else [surface_color, surface_color],
        opacity=0.8 if not surface_color else 1,
        meta=meta,
    )

    left = torch.stack((domain[0][0].expand(res), torch.flip(sy, dims=[0])))
    top = torch.stack((sx, domain[1][0].expand(res)))
    right = torch.stack((domain[0][1].expand(res), sy))
    bottom = torch.stack((torch.flip(sx, dims=[0]), domain[1][1].expand(res)))

    lines = torch.cat((left.T, top.T, right.T, bottom.T))

    if hidden_w is not None:
        preds = predict(lines, hidden_w, hidden_b).T
        preds = predict(preds, w, b)
    else:
        preds = predict(lines, w, b, planarity=planarity)

    border = go.Scatter3d(
        x=lines[:, 0],
        y=lines[:, 1],
        z=preds[0],
        mode="lines",
        line=dict(color=surface_color or "black", width=6),
        meta=meta,
        visible=surface_color is not None,
    )

    boundary = decision_boundary(w, b, domain, visible=show_decision_boundaries, meta=meta)

    return [surface, boundary, border]


def neural_network(frame: Frame, animation: Animation, name="neural-network", meta=None):
    feature_colors = animation.theme.feature_colors

    # need to create a new trace that draws lines between neural network nodes
    # for each layer, draw lines between each node in the layer to each node in the next layer
    input_size = frame.input_size
    hidden_size = frame.hidden_size
    output_size = frame.output_size

    # space out node markers between -1, 1 centered on 0
    input_xs = np.linspace(-1, 1, input_size) if input_size > 1 else [0]
    hidden_xs = np.linspace(-1, 1, hidden_size) if hidden_size > 1 else [0]
    output_xs = np.linspace(-1, 1, output_size) if output_size > 1 else [0]

    input_ys = [-2] * input_size
    hidden_ys = [-1] * hidden_size
    output_ys = [0] * output_size

    input_points = list(zip(input_xs, input_ys))
    hidden_points = list(zip(hidden_xs, hidden_ys))
    output_points = list(zip(output_xs, output_ys))

    # create a scatter plot with the nodes
    nodes = go.Scatter(
        name=name,
        x=[x for x, _ in input_points + hidden_points + output_points],
        y=[y for _, y in input_points + hidden_points + output_points],
        mode="markers",
        marker=dict(size=80, color="black"),
        meta=meta,
    )

    feature_colors = [hex_to_rgb(c) for c in feature_colors.values()]
    input_colors = interp_rgb(feature_colors[0], feature_colors[1], input_size)
    hidden_colors = interp_rgb(feature_colors[0], feature_colors[1], hidden_size)
    input_colors = [rgb_to_str(c) for c in input_colors]
    hidden_colors = [rgb_to_str(c) for c in hidden_colors]

    lines = {}
    for i, p1 in enumerate(input_points):
        color = input_colors[i]
        for p2 in hidden_points:
            if color not in lines:
                lines[color] = []
            lines[color].extend([p1, p2, None])

    for i, p1 in enumerate(hidden_points):
        color = hidden_colors[i]
        for p2 in output_points:
            if color not in lines:
                lines[color] = []
            lines[color].extend([p1, p2, None])

    # create a scatter plot with the edges
    edges = []
    for color in lines:
        feature_lines = lines[color]
        edges.append(
            go.Scatter(
                name=name,
                x=[p[0] if p else None for p in feature_lines],
                y=[p[1] if p else None for p in feature_lines],
                mode="lines",
                line=dict(width=5, color=color),
                # marker_color=colors,
                meta=meta,
            )
        )

    return [nodes, *edges]


def loss_tables(
    frame: Frame,
    meta=None,
    theme=default_theme,
):
    X = frame.X
    preds = frame.preds
    targets = frame.targets
    focus_total_loss = frame.focus_total_loss
    lr = frame.learning_rate
    class_colors = theme.class_colors

    feature_colors = frame.focusable_feature_colors(theme)

    focused_inputs = frame.focused_inputs

    if frame.focused_inputs is None:
        focused_inputs = range(X.shape[0])

    if isinstance(focused_inputs, int):
        focused_inputs = list(range(focused_inputs + 1))

    X = X[focused_inputs]
    targets = targets[0][focused_inputs]
    preds = preds[0][focused_inputs]

    m = X.shape[0]

    errors = targets - preds
    loss = torch.stack((errors * X[:, 0], errors * X[:, 1]), dim=0)
    total_loss = torch.sum(loss, dim=1)

    font_size = 30

    header = dict(
        values=[
            "",
            "<b>error * x</b>",
            "<b>= loss<sup>1</sup><sub>i</sub></b>",
            "<b>error * y</b>",
            "<b>= loss<sup>2</sup><sub>i<sub></b>",
        ],
        line=dict(width=2),
        font=dict(size=font_size, color=["black", "black", "black", "black", "black"]),
        fill_color=["white", feature_colors[0], "white", feature_colors[1], "white"],
    )

    cols = [
        [f"<b>x<sub>{i}</sub><b>" for i in range(1, m + 1)] + ["<b>∑</b>"],
        [f"<b>{e:.2f} * {x:0.2f}</b>" for e, x in torch.stack((errors, X[:, 0]), dim=1)],
        [f"= <b>{l:.2f}</b>" for l in loss[0]] + [f"= <b>{total_loss[0]:.3f}</b>"],
        [f"<b>{e:.2f} * {y:0.2f}</b>" for e, y in torch.stack((errors, X[:, 1]), dim=1)],
        [f"= <b>{l:.2f}</b>" for l in loss[1]] + [f"= <b>{total_loss[1]:.3f}</b>"],
    ]

    cells = dict(
        values=cols,
        font=dict(
            size=font_size,
            color=[
                ["black"] * (len(cols[0]) - 1) + ["white"],
                ["black"] * len(cols[1]),
                ["black"] * (len(cols[2]) - 1) + ["white"],
                ["black"] * len(cols[3]),
                ["black"] * (len(cols[4]) - 1) + ["white"],
            ],
        ),
        line=dict(width=2),
        fill_color=[
            [class_colors[bool(t > 0.5)] for t in targets] + ["black"],
            [feature_colors[0]] * len(cols[1]) + ["black"],
            ["white"] * (len(cols[2]) - 1) + ["black"],
            [feature_colors[1]] * len(cols[3]) + ["black"],
            ["white"] * (len(cols[4]) - 1) + ["black"],
        ],
    )

    align = ["center", "center", "left", "center", "left"]

    loss_table = go.Table(
        header=dict(**header, align=align, height=50),
        cells=dict(**cells, align=align, height=50),
        columnorder=[0, 1, 2, 3, 4],
        columnwidth=[0.6, 2, 1.4, 2, 1.4],
        meta=meta[0],
    )

    header = dict(
        values=[
            "<b>1/m</b> * <b>lr</b> * <b>cost<sub>1</sub></b>",
            "= <b>-Δw<sub>1</sub></b>",
            "<b>1/m</b> * <b>lr</b> * <b>cost<sub>2</sub></b>",
            "= <b>-Δw<sub>2</sub></b>",
        ],
        font=dict(size=font_size, color=["black"] * 4),
        line=dict(width=2),
        fill_color=[feature_colors[0], "white", feature_colors[1], "white"],
    )

    cells = dict(
        values=[
            f"<b>1/{m} * {lr} * {total_loss[0]:.3f}</b>",
            f"= <b>{(1/m) * lr * total_loss[0]:.3f}</b>",
            f"<b>1/{m} * {lr} * {total_loss[1]:.3f}</b>",
            f"= <b>{(1/m) * lr * total_loss[1]:.3f}</b>",
        ],
        font=dict(size=font_size, color=["black", "white", "black", "white"]),
        line=dict(width=2),
        fill_color=[
            feature_colors[0],
            "grey" if focus_total_loss else "black",
            feature_colors[1],
            "grey" if focus_total_loss else "black",
        ],
    )

    align = ["center", "left", "center", "left"]

    total_loss_table = go.Table(
        header=dict(**header, align=align, height=44),
        cells=dict(**cells, align=align, height=44),
        columnorder=[0, 1, 2, 3],
        columnwidth=[3, 1.2, 3, 1.2],
        meta=meta[1],
    )

    return [loss_table, total_loss_table]
