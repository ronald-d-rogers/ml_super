from typing import List
import torch
import plotly.graph_objects as go
from learning import log_loss, predict
from themes import Theme, default_theme


default_marker_size = 30


def loss_lines(X, preds, targs, opacity=1, line=None, visible=True, meta=None):
    if not line:
        line = dict(color="black", width=7, dash="dot")

    xs, ys, zs = [], [], []

    if X.size(0) and preds is not None and targs is not None:
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
    X,
    preds,
    targs,
    focused=None,
    marker_size=default_marker_size,
    visible=True,
    meta=None,
    theme=default_theme,
):
    m = X.size(0)
    xs, ys, zs = [], [], []

    if m and preds is not None and targs is not None:
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
    X,
    preds,
    targs,
    indices,
    focus=None,
    size=default_marker_size,
    standoff=None,
    meta=None,
    theme=default_theme,
    visible=True,
):
    m = X.size(0)
    xs, ys, zs = [], [], []
    marker_color = None
    textfont_color = None

    if m and preds is not None and targs is not None:
        xs, ys, zs = X[:, 0], X[:, 1], targs + standoff

        if focus:
            line = dict(color="black", width=1)
            marker_color = [theme.class_colors[bool(t > 0.5)] for t in targs]
            textfont_color = theme.focused_target_marker_text_color

        else:
            # if pred close enough to target, marker_color is transparent, else black
            line = None
            marker_color = "rgba(0, 0, 0, 0)"
            textfont_color = [
                theme.target_marker_text_color if abs(t - p) > 0.25 else "rgba(0, 0, 0, 0)"
                for t, p in torch.stack((targs, preds), dim=1)
            ]

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text",
        marker=dict(size=size, line=line),
        text=[f"x<sub>{i+1}</sub>" for i in indices],
        marker_color=marker_color,
        textfont_size=size,
        textfont_color=textfont_color,
        textposition="middle center",
        visible=visible,
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
    if not visible or preds is None or targs is None:
        preds = torch.Tensor([])
        targs = torch.Tensor([])
        indices = []

    return go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=(preds + standoff),
        mode="markers+text",
        marker=dict(size=size, line=dict(color="black", width=1)),
        marker_size=size,
        opacity=opacity,
        marker_color=[theme.class_colors[bool(t > 0.5)] for t in targs],
        text=[f"x<sub>{i+1}</sub>" for i in indices],
        textfont_size=size,
        textfont_color=theme.marker_text_color,
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
    show_targets=True,
    marker_size=30,
    meta: dict = None,
    theme: Theme = None,
):
    m = X.size(0)
    indices = [i for i in range(m)]
    preds_opacity = 0.1 if focus_targets else 1

    standoff = 0.033

    traces = [
        loss_lines(
            X,
            preds,
            targets,
            meta=meta,
            visible=show_targets,
        ),
        target_markers(
            X,
            preds,
            targets,
            indices,
            focus=focus_targets,
            size=marker_size,
            standoff=standoff,
            visible=show_targets,
            theme=theme,
            meta=meta,
        ),
        inference_marker(
            inference,
            w,
            b,
            size=marker_size,
            meta=meta,
        ),
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
            theme=theme,
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
        loss_lines(
            X,
            preds,
            targs,
            line=dict(color="red", width=10),
            meta=meta,
            visible=show_targets,
        ),
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
    modules,
    activations,
    activity=1,
    show_profile=False,
    show_decision_boundaries=False,
    profile_line_width=6,
    res=20,
    meta=None,
):
    sx = surface_linspace[:, 0]
    sy = surface_linspace[:, 1]

    preds = surface_points.T

    # skip input module
    modules = modules[1:]

    for i, module in enumerate(modules):
        activity = activity if i == len(modules) - 1 else None
        preds = predict(
            preds.T,
            w[module],
            b[module],
            activation=activations[module],
            activity=activity,
        )

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

    preds = lines.T
    for i, module in enumerate(modules):
        activity = activity if i == len(modules) - 1 else None
        preds = predict(
            preds.T,
            w[module],
            b[module],
            activation=activations[module],
            activity=activity,
        )

    border = go.Scatter3d(
        x=lines[:, 0],
        y=lines[:, 1],
        z=preds[0],
        mode="lines",
        line=dict(color="black", width=profile_line_width),
        meta=meta,
        visible=show_profile,
    )

    return [surface, border]


def gradient(
    X,
    targets,
    surface_points,
    surface_linspace,
    w,
    b,
    modules,
    activations,
    param1=["output_1", 0],
    param2=["output_1", 1],
    loss_function=log_loss,
    activity=1,
    show_profile=False,
    res=20,
    meta=None,
):
    sx = surface_linspace[:, 0]
    sy = surface_linspace[:, 1]

    # skip input module
    modules = modules[1:]

    losses = []

    for point in surface_points:
        for i, module in enumerate(modules):
            if module == param1[0]:
                module_w = w[param1[0]].clone()
                module_w[param1[1]] += point[0]
            if module == param2[0]:
                module_w = w[param2[0]].clone()
                module_w[param2[1]] += point[1]
            activity = activity if i == len(modules) - 1 else None

            preds = predict(
                X,
                module_w,
                b[module],
                activation=activations[module],
                activity=activity,
            )

            losses.append(torch.nansum(loss_function(preds, targets)))

    losses = torch.reshape(losses, (res, res))
    losses = torch.rot90(losses, 3)
    losses = torch.flip(losses, dims=[1])

    surface = go.Surface(
        x=sx,
        y=sy,
        z=losses,
        showscale=False,
        colorscale="plasma" if not show_profile else ["black", "black"],
        opacity=0.8 if not show_profile else 1,
        meta=meta,
    )

    # left = torch.stack((domain[0][0].expand(res), torch.flip(sy, dims=[0])))
    # top = torch.stack((sx, domain[1][0].expand(res)))
    # right = torch.stack((domain[0][1].expand(res), sy))
    # bottom = torch.stack((torch.flip(sx, dims=[0]), domain[1][1].expand(res)))

    # lines = torch.cat((left.T, top.T, right.T, bottom.T))

    # preds = lines.T
    # for i, module in enumerate(modules):
    #     activity = activity if i == len(modules) - 1 else None
    #     preds = predict(
    #         preds.T,
    #         w[module],
    #         b[module],
    #         activation=activations[module],
    #         activity=activity,
    #     )

    # border = go.Scatter3d(
    #     x=lines[:, 0],
    #     y=lines[:, 1],
    #     z=preds[0],
    #     mode="lines",
    #     line=dict(color="black", width=profile_line_width),
    #     meta=meta,
    #     visible=show_profile,
    # )

    return [surface]
