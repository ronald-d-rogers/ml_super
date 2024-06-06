from base import Animation, Frame


import plotly.graph_objects as go


def neural_network(frame: Frame, animation: Animation, name="neural-network", meta=None):
    size = frame.size
    node_points = frame.node_points
    focused_connections = frame.focused_connections

    # create a scatter plot with the nodes
    nodes = go.Scatter(
        name=name,
        x=[x for x, _ in node_points["input"] + node_points["hidden"] + node_points["output"]],
        y=[y for _, y in node_points["input"] + node_points["hidden"] + node_points["output"]],
        mode="markers",
        marker=dict(size=80, color="black"),
        meta=meta,
    )

    input_colors = animation.input_colors(size["input"])
    hidden_colors = animation.hidden_colors(size["hidden"])

    def edge(color, points, width=5):
        return go.Scatter(
            name=name,
            x=[p[0] if p else None for p in points],
            y=[p[1] if p else None for p in points],
            mode="lines",
            line=dict(width=width, color=color),
            meta=meta,
        )

    edges = []
    for i, p1 in enumerate(node_points["input"]):
        for j, p2 in enumerate(node_points["hidden"]):
            color = input_colors[i]
            line = [p1, p2]
            width = 10 if i in focused_connections["input"] and j in focused_connections["hidden"] else 5
            edges.append(edge(color, line, width))

    for i, p1 in enumerate(node_points["hidden"]):
        for j, p2 in enumerate(node_points["output"]):
            color = hidden_colors[i]
            line = [p1, p2]
            width = 10 if i in focused_connections["hidden"] and j in focused_connections["output"] else 5
            edges.append(edge(color, line, width))

    return [nodes, *edges]
