import torch
import plotly.graph_objects as go
from frame import Frame
from themes import default_theme
from utils import predict, focusable_feature_colors


default_marker_size = 30


def loss_lines(X, preds, targs, width=10, opacity=1, line=None, visible=True, meta=None):
    if not line:
        line = dict(color="#2B3F60", width=width, dash="dot")

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
    X, preds, targs, focused=None, marker_size=default_marker_size, visible=True, meta=None, theme=default_theme
):
    m = X.shape[0]
    xs, ys, zs = [], [], []

    if m:
        for x, y, p, t in torch.stack((X[:, 0], X[:, 1], preds, targs), dim=1):
            t = int(t)
            xs.extend([x, x if focused == 1 else 0, None])
            ys.extend([y, y if focused == 0 else 0, None])
            zs.extend([p, p, None])

    color = theme.feature_colors[focused] if focused is not None else "black"

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
    X, targs, indices, focus=None, size=default_marker_size, standoff=None, meta=None, theme=default_theme
):
    marker_color = [theme.class_colors[bool(t > 0.5)] for t in targs]

    return go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=targs + standoff,
        mode="markers+text",
        marker=dict(size=size, line=dict(color="black", width=1)),
        opacity=0.4 if not focus else 1,
        marker_color=marker_color,
        text=[f"x<sub>{i+1}</sub>" for i in indices],
        textfont_size=size,
        textposition="middle center",
        meta=meta,
    )


def pred_markers(
    X,
    preds,
    targs,
    indices,
    size=default_marker_size,
    standoff=None,
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
    standoff=0,
    meta=None,
    theme=default_theme,
):
    X = frame.X
    w = frame.w
    b = frame.b
    preds = frame.preds
    targets = frame.targets
    inference = frame.inference
    focused_feature = frame.focused_feature
    focus_targets = frame.focus_targets

    m = X.shape[0]

    focused = frame.focused

    if frame.focused is None:
        focused = []

    if isinstance(frame.focused, int):
        focused = [frame.focused]

    traces = [loss_lines(frame.X, frame.preds, frame.targets, width=line_width, meta=meta)]

    foc = torch.Tensor([i in focused for i in range(m)])

    X_foc, X = X[foc == True], X[foc == False]  # noqa: E712
    preds_foc, preds = preds[foc == True], preds[foc == False]  # noqa: E712
    targs_foc, targs = targets[foc == True], targets[foc == False]  # noqa: E712

    indices = [i for i in range(m) if i not in focused]
    indices_foc = [i for i in range(m) if i in focused]

    show_preds = frame.show_preds
    preds_opacity = 0.4 if focus_targets else 1
    show_feature_lines = focused_feature is not None

    traces += [
        target_markers(X, targs, indices, focus_targets, size=marker_size, standoff=standoff, meta=meta),
        feature_lines(
            X, preds, targs, focused_feature, marker_size, visible=show_feature_lines, meta=meta, theme=theme
        ),
        pred_markers(
            X,
            preds,
            targs,
            indices,
            marker_size,
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
        loss_lines(X, preds, targs, line_width, line=line, meta=meta),
        feature_lines(
            X, preds, targs, focused_feature, marker_size, visible=show_feature_lines, meta=meta, theme=theme
        ),
        target_markers(X, targs, indices, focus_targets, marker_size, standoff=standoff, meta=meta),
        pred_markers(
            X,
            preds,
            targs,
            indices,
            marker_size,
            standoff=standoff,
            opacity=preds_opacity,
            visible=show_preds,
            meta=meta,
        ),
    ]

    return traces


def decision_boundary(w, b, domain, visible=True, meta=None):
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
    SX = frame.surface_points
    SL = frame.surface_line
    domain = frame.domain
    w = frame.w
    b = frame.b
    planarity = frame.planarity
    show_surface = frame.show_surface
    show_decision_boundary = frame.show_decision_boundary
    res = frame.resolution

    sx = SL[:, 0]
    sy = SL[:, 1]

    preds = predict(SX, w, b, planarity)
    preds = torch.reshape(preds, (res, res))
    preds = torch.rot90(preds, 3)
    preds = torch.flip(preds, dims=[1])

    surface = go.Surface(
        x=sx,
        y=sy,
        z=preds,
        showscale=False,
        colorscale="plasma" if show_surface else ["black", "black"],
        opacity=0.8 if show_surface else 1,
        meta=meta,
    )

    l1 = torch.stack((domain[0][0].expand(res), torch.flip(sy, dims=[0])))
    l2 = torch.stack((sx, domain[1][0].expand(res)))
    l3 = torch.stack((domain[0][1].expand(res), sy))
    l4 = torch.stack((torch.flip(sx, dims=[0]), domain[1][1].expand(res)))
    lines = torch.cat((l1.T, l2.T, l3.T, l4.T))

    preds = predict(lines, w, b, planarity)

    border = go.Scatter3d(
        x=lines[:, 0],
        y=lines[:, 1],
        z=preds,
        mode="lines",
        line=dict(color="black", width=6),
        meta=meta,
    )

    boundary = decision_boundary(w, b, domain, visible=show_decision_boundary, meta=meta)

    objs = [surface, boundary, border]

    return objs


def loss_tables(
    frame: Frame,
    meta=None,
    theme=default_theme,
):
    X = frame.X
    preds = frame.preds
    targets = frame.targets
    focused_feature = frame.focused_feature
    focus_total_loss = frame.focus_total_loss
    lr = frame.learning_rate
    class_colors = theme.class_colors

    feature_colors = focusable_feature_colors(focused_feature, theme)

    focused = frame.focused

    if frame.focused is None:
        focused = range(X.shape[0])

    if isinstance(focused, int):
        focused = list(range(focused + 1))

    X = X[focused]
    targets = targets[focused]
    preds = preds[focused]

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
