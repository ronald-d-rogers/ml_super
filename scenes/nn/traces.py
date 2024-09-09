import torch
import plotly.graph_objects as go

from base import Animation, Frame


def neural_network(frame: Frame, animation: Animation, name="neural-network", meta=None):
    size = frame.size
    node_points = frame.node_points
    focused_connections = frame.focused_connections

    modules = frame.modules

    w = frame.w

    colors = {}
    for module in modules:
        colors[module] = animation.colors(size[module])

    focused_colors = {}
    for module in modules:
        focused_colors[module] = animation.focused_colors(size[module])

    marker_color = []
    points = []
    for module in modules:
        marker_color.extend(colors[module])
        points.extend(node_points[module])

    bias_points = []
    for module in modules[1:]:
        bias_points.extend(node_points[module])

    nodes = go.Scatter(
        name=name,
        x=[x for x, _ in points],
        y=[y for _, y in points],
        mode="markers",
        marker=dict(size=80),
        marker_color=marker_color,
        meta=meta,
    )

    biases = go.Scatter(
        name=name,
        x=[x + 0.1 for x, _ in bias_points],
        y=[y + -0.033 for _, y in bias_points],
        mode="markers+text",
        marker=dict(size=30),
        marker_color="black",
        text="b",
        textposition="middle center",
        textfont=dict(color="white", size=10),
        meta=meta,
    )

    def edge(color, points, width=5):
        return go.Scatter(
            name=name,
            x=[p[0] if p else None for p in points],
            y=[p[1] if p else None for p in points],
            mode="lines",
            marker=dict(color=color),
            marker_color=color,
            line=dict(width=width, color=color),
            meta=meta,
        )

    edges = []

    for i, module in enumerate(modules[:-1]):
        for j, p1 in enumerate(node_points[module]):
            next_module = modules[i + 1]
            for k, p2 in enumerate(node_points[next_module]):
                color = (
                    focused_colors[module][j]
                    if j in focused_connections[module] and k in focused_connections[next_module]
                    else colors[module][j]
                )
                line = [p1, p2]
                width = max(2.5, abs(w[next_module][k][j].item())) * 2
                edges.append(edge(color, line, width))

    return [nodes, biases, *edges]


def nn_activations(frame: Frame, animation: Animation, name="activations", meta=None):
    node_points = frame.node_points

    modules = frame.modules
    activations = frame.activation_fns

    domain = 2
    res = 40

    y_ls = torch.linspace(-domain, domain, res)
    x_ls = torch.linspace(-1, 1, res)

    activation_lines = []

    for module in modules[1:]:
        activation = activations[module]
        for i, w in enumerate(frame.w[module]):
            ls = y_ls.expand(w.size(0), -1)

            preds = w @ ls
            preds = activation(preds + frame.b[module][i])

            node_point = node_points[module][i]
            line = []
            for j in range(preds.size(0)):
                x = node_point[0] + (x_ls[j] / 6.666)
                y = (node_point[1] + (preds[j].item() / 6.666)) - (0.5 / 6.666)
                line.append((x, y))
            activation_lines.append(line)

    xs = []
    ys = []
    for line in activation_lines:
        xs.extend([x for x, _ in line] + [None])
        ys.extend([y for _, y in line] + [None])

    return [
        go.Scatter(
            name=name,
            x=xs,
            y=ys,
            mode="lines",
            line=dict(width=4, color="black"),
            meta=meta,
        )
    ]


def nn_traces(frame: Frame, animation: Animation):
    if not animation.show_network:
        return []

    meta = animation.cells["network"]

    return [
        *neural_network(
            frame,
            animation,
            meta=dict(row=meta["row"], col=meta["col"]),
        ),
        *nn_activations(
            frame,
            animation,
            meta=dict(row=meta["row"], col=meta["col"]),
        ),
    ]
