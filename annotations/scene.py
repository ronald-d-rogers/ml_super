import numpy as np
import torch

from themes import default_theme


no_note = dict(visible=False, showarrow=False)


def error_annotations(X, targets, preds, focused_errors, show=True):
    m = X.size(0)

    if not show or not focused_errors:
        return [no_note for _ in range(m)]

    # use pred for z if target is 0 else use target
    z = [pred if t == 0 else t for t, pred in zip(targets, preds)]

    return [
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
        if i in focused_errors
    ]


def feature_annotations(
    X,
    preds,
    targets,
    focused_feature=None,
    feature_colors=None,
    show=False,
    theme=default_theme,
):
    if not show:
        return []

    if feature_colors is None:
        feature_colors = theme.feature_colors

    m = X.size(0)

    def annotate(i, t, x, y, z):
        return dict(
            x=x,
            y=y,
            z=z,
            yshift=40,
            text=f"<b>{targets[i] - preds[i]:1.2f}({float(X[i, t]):1.2f})</b>",
            bgcolor=feature_colors[t],
            bordercolor="rgba(0,0,0,.8)",
            align="center",
            font=dict(size=26),
            showarrow=False,
        )

    annotations = []
    z = [pred if t == 0 else t for t, pred in zip(targets, preds)]
    for i in range(m):
        if focused_feature in (None, 0):
            note = annotate(i, 0, X[:, 0][i], X[:, 1][i], z[i])

        elif focused_feature in (None, 1):
            note = annotate(i, 1, X[:, 0][i], X[:, 1][i], z[i])

        else:
            note = no_note

        annotations.append(note)

    return annotations


def inference_annotation(w, b, inference, show=True):
    if not show:
        return [no_note]

    if inference is None:
        return [no_note]

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
    b,
    width,
    feature_colors,
    show_label_names=True,
    label_precision=3,
    label_yshift=0,
    label_font_size=40,
    theme=default_theme,
    show=True,
):
    if not show:
        return []

    annotation = dict(
        yanchor="top",
        bordercolor="rgba(0,0,0,.8)",
        borderwidth=2,
        yshift=300 + label_yshift,
        font=dict(size=label_font_size),
        showarrow=False,
    )

    count = w.size(0) + 1

    start = width / (count * 2)

    xs = np.linspace(start, width - start, w.size(0) + 1)

    notes = []

    for i, x in enumerate(xs[:-1]):
        text = f"<b>{w[i]:.{label_precision}f}</b>"
        if show_label_names:
            text = f"w{i+1}: " + text

        notes.append(
            dict(
                **annotation,
                x=0,
                y=0,
                xshift=x,
                xanchor="center",
                bgcolor=feature_colors[i],
                text=text,
            )
        )

    text = f"<b>{b[0]:.{label_precision}f}</b>"
    if show_label_names:
        text = "bias: " + text

    notes.append(
        dict(
            **annotation,
            x=0,
            y=0,
            xshift=xs[-1],
            xanchor="center",
            bgcolor=theme.target_color,
            text=text,
            font_color=theme.target_text_color,
        )
    )

    return notes
