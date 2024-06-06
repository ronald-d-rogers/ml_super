from typing import List
import torch
import plotly.graph_objects as go
from learning import predict
from themes import Theme, default_theme


default_marker_size = 30


def loss_lines(X, preds, targs, opacity=1, line=None, visible=True, meta=None):
    if not line:
        line = dict(color="white", width=5, dash="dot")

    xs, ys, zs = [], [], []

    if X.size(0):
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
    m = X.size(0)
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
        text=[f"x<sub>{i+1}</sub>" for i in indices],
        marker_color=marker_color,
        textfont_size=size,
        textfont_color=textfont_color,
        textposition="middle center",
        meta=meta,
    )


def pred_markers(
    X: torch.Tensor,
    preds: torch.Tensor,
    targs: torch.Tensor,
    indices: List[int],
    size=default_marker_size,
    standoff=0.033,
    opacity=1,
    visible=True,
    meta: dict = None,
    theme: Theme = default_theme,
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
    X: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    inference: torch.Tensor = None,
    focused_feature: int = None,
    focused_errors: List[int] = None,
    focus_targets=False,
    show_preds=True,
    marker_size=30,
    meta: dict = None,
    theme: Theme = None,
):
    m = X.size(0)
    indices = [i for i in range(m)]
    preds_opacity = 0.4 if focus_targets else 1

    standoff = 0.033

    traces = [
        loss_lines(X, preds, targets, meta=meta),
        target_markers(X, preds, targets, indices, focus_targets, size=marker_size, standoff=standoff, meta=meta),
        inference_marker(inference, w, b, size=marker_size, meta=meta),
        feature_lines(
            X,
            preds,
            targets,
            focused_feature,
            marker_size=marker_size,
            visible=focused_feature is not None,
            meta=meta,
            theme=theme,
        ),
        pred_markers(
            X,
            preds,
            targets,
            indices,
            size=marker_size,
            standoff=standoff,
            opacity=preds_opacity,
            visible=show_preds,
            meta=meta,
        ),
    ]

    if focused_errors:
        mask = torch.Tensor([i in focused_errors for i in range(m)])

        X = X[mask == True]  # noqa: E712
        preds = preds[mask == True]  # noqa: E712
        targs = targets[mask == True]  # noqa: E712
    else:
        X = torch.Tensor([])
        preds = torch.Tensor([])
        targs = torch.Tensor([])

    traces += [
        loss_lines(X, preds, targs, line=dict(color="red", width=10), meta=meta),
    ]

    return traces


def decision_boundary(w: torch.Tensor, b: torch.Tensor, domain, visible=True, meta=None):
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


def model_surface(
    domain,
    surface_points,
    surface_linspace,
    w,
    b,
    activity=1,
    show_profile=False,
    show_decision_boundaries=False,
    res=20,
    meta=None,
):
    sx = surface_linspace[:, 0]
    sy = surface_linspace[:, 1]

    if "hidden" in w:
        preds = predict(surface_points, w["hidden"], b["hidden"]).T
        preds = predict(preds, w["output"], b["output"])
    else:
        preds = predict(surface_points, w["output"], b["output"], activity=activity)

    preds = torch.reshape(preds, (res, res))
    preds = torch.rot90(preds, 3)
    preds = torch.flip(preds, dims=[1])

    surface = go.Surface(
        x=sx,
        y=sy,
        z=preds,
        showscale=False,
        colorscale="plasma" if not show_profile else ["black", "black"],
        opacity=0.8 if not show_profile else 1,
        meta=meta,
    )

    left = torch.stack((domain[0][0].expand(res), torch.flip(sy, dims=[0])))
    top = torch.stack((sx, domain[1][0].expand(res)))
    right = torch.stack((domain[0][1].expand(res), sy))
    bottom = torch.stack((torch.flip(sx, dims=[0]), domain[1][1].expand(res)))

    lines = torch.cat((left.T, top.T, right.T, bottom.T))

    if "hidden" in w:
        preds = predict(lines, w["hidden"], b["hidden"]).T
        preds = predict(preds, w["output"], b["output"])
    else:
        preds = predict(lines, w["output"], b["output"], activity=activity)

    border = go.Scatter3d(
        x=lines[:, 0],
        y=lines[:, 1],
        z=preds[0],
        mode="lines",
        line=dict(color="black", width=6),
        meta=meta,
        visible=show_profile,
    )

    if "hidden" not in w:
        boundary = decision_boundary(
            w["output"][0], b["output"][0], domain, visible=show_decision_boundaries, meta=meta
        )
    else:
        boundary = go.Scatter3d(visible=False, meta=meta)

    return [surface, boundary, border]
