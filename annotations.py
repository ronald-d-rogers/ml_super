import torch

from typing import Union

from frame import Frame
from themes import Theme, default_theme


def loss_annotations(frame: Frame, visible=True):
    if not visible:
        return []

    X = frame.X
    targets = frame.targets
    preds = frame.preds

    m = X.shape[0]

    focused = frame.focused

    if not frame.focused:
        focused = range(m)

    if isinstance(frame.focused, int):
        focused = [frame.focused]

    # use pred for z if target is 0 else use target
    z = [pred if t == 0 else t for t, pred in zip(targets, preds)]

    test = [
        dict(
            x=X[:, 0][i],
            y=X[:, 1][i],
            z=z[i],
            yshift=40,
            bgcolor="rgba(255,255,255,.8)",
            bordercolor="rgba(0,0,0,.8)",
            text=f"<b>{float(targets[i] - pred):1.2f}</b>",
            align="center",
            font=dict(size=30),
            showarrow=False,
        )
        for i, pred in enumerate(preds)
        if i in focused
    ]

    return test


def focusable_feature_colors(focused_feature: Union[None, bool], theme: Theme):
    if focused_feature is not None:
        return {
            0: theme.focused_feature_colors[0] if focused_feature == 0 else theme.feature_colors[0],
            1: theme.focused_feature_colors[1] if focused_feature == 1 else theme.feature_colors[1],
        }
    else:
        return theme.feature_colors


def feature_annotations(
    frame: Frame,
    feature_colors=None,
    visible=False,
    theme=default_theme,
):
    if not visible:
        return []

    preds = frame.preds
    focused_feature = frame.focused_feature

    if feature_colors is None:
        feature_colors = theme.feature_colors

    X = frame.X
    targets = frame.targets

    m = X.shape[0]

    focused = frame.focused

    if not frame.focused:
        focused = range(m)

    annotations = []

    z = [pred if t == 0 else t for t, pred in zip(targets, preds)]

    def annotation(i, t, x, y, z):
        return dict(
            x=x,
            y=y,
            z=z,
            yshift=40,
            text=f"<b>{targets[i] - preds[i]:1.2f}({float(X[i, t]):1.2f})</b>",
            # startstandoff=40,
            bgcolor=feature_colors[t],
            bordercolor="rgba(0,0,0,.8)",
            align="center",
            font=dict(size=26),
            showarrow=False,
        )

    for i in focused:
        if focused_feature in (None, 0):
            note = annotation(i, 0, X[:, 0][i], X[:, 1][i], z[i])

        if focused_feature in (None, 1):
            note = annotation(i, 1, X[:, 0][i], X[:, 1][i], z[i])

        annotations.append(note)

    return annotations


def inference_annotation(frame: Frame, visible=True):
    if not visible:
        return []

    w, b = frame.w, frame.b
    inference = frame.inference

    if frame.inference is None:
        return []

    pred = torch.sigmoid((inference @ w.T) + b).item()
    text = f"<b>{pred:.2f} {'>' if pred > 0.5 else '<'} 0.5</b>"
    return [
        dict(
            x=inference[0],
            y=inference[1],
            z=pred,
            yshift=40,
            bgcolor="rgba(255,255,255,.8)",
            bordercolor="rgba(0,0,0,.8)",
            text=text,
            font=dict(size=30),
        )
    ]


def weight_annotations(
    w,
    height,
    focused_feature,
    focus_labels,
    visible=True,
    theme=default_theme,
):
    if not visible:
        return []

    if focus_labels:
        feature_colors = theme.focused_feature_colors
    else:
        feature_colors = focusable_feature_colors(focused_feature, theme)

    annotation = dict(
        yanchor="top",
        bordercolor="rgba(0,0,0,.8)",
        borderwidth=2,
        font=dict(size=40),
        showarrow=False,
    )

    return [
        dict(
            **annotation,
            x=0.05,
            xanchor="left",
            y=1 - (60 / height),
            bgcolor=feature_colors[0],
            text=f"Feature 1: <b>{w[0]:1.3f}</b>",
        ),
        dict(
            **annotation,
            x=1 - 0.05,
            xanchor="right",
            y=1 - (60 / height),
            bgcolor=feature_colors[1],
            text=f"Feature 2: <b>{w[1]:1.3f}</b>",
        ),
    ]
