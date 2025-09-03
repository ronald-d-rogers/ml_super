from base import Animation, AnimationFrame, LayoutComponent, NodeView

import plotly.graph_objects as go
import torch

from base import ViewFrameUpdate


class LossesComponent(LayoutComponent):
    view_names = ["Losses"]
    view_types = ["table"]
    height = 768

    def create_component(self, view: NodeView, frame: AnimationFrame):
        return []

    def update_component(self, view: NodeView, frame: AnimationFrame):
        return [ViewFrameUpdate(view=None, data=losses_table(view, frame, self.animation), annotations=[])]


class CostComponent(LayoutComponent):
    name = ["Cost"]
    view_types = ["table"]
    height = 128

    def create_component(self, view: NodeView, frame: AnimationFrame):
        return []

    def update_component(self, view: NodeView, frame: AnimationFrame):
        return [ViewFrameUpdate(view=None, data=cost_table(view, frame, self.animation), annotations=[])]


def losses_table(
    view: NodeView,
    frame: AnimationFrame,
    animation: Animation,
    meta=None,
):
    X = frame.X
    m = X.size(0)
    preds = view.preds
    targets = view.targets
    modules = frame.modules
    size = frame.size
    class_colors = animation.theme.class_colors

    focused_preds = view.focused_preds
    feature_colors = animation.focusable_colors(frame.focused_feature, size[modules[-2]])

    X = X if not focused_preds else X[focused_preds]

    targets = targets if not focused_preds else targets[focused_preds]
    preds = preds if not focused_preds else preds[focused_preds]

    m = X.size(0)

    errors = targets - preds
    losses = torch.stack((errors.T * X[:, 0], errors.T * X[:, 1]), dim=0)
    total_loss = torch.nansum(losses, dim=1)

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
        [f"= <b>{loss:.2f}</b>" for loss in losses[0]] + [f"= <b>{total_loss[0]:.3f}</b>"],
        [f"<b>{e:.2f} * {y:0.2f}</b>" for e, y in torch.stack((errors, X[:, 1]), dim=1)],
        [f"= <b>{loss:.2f}</b>" for loss in losses[1]] + [f"= <b>{total_loss[1]:.3f}</b>"],
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

    return go.Table(
        header=dict(**header, align=align, height=50),
        cells=dict(**cells, align=align, height=50),
        columnorder=[0, 1, 2, 3, 4],
        columnwidth=[0.6, 2, 1.4, 2, 1.4],
        meta=meta,
    )


def cost_table(view: NodeView, frame: AnimationFrame, animation: Animation, meta=None):
    X = frame.X
    m = X.size(0)
    preds = view.preds
    targets = view.targets
    focus_costs = frame.focus_costs
    modules = frame.modules
    size = frame.size
    lr = frame.learning_rate

    focused_preds = view.focused_preds
    feature_colors = animation.focusable_colors(frame.focused_feature, size[modules[-2]])

    X = X if not focused_preds else X[focused_preds]

    targets = targets if not focused_preds else targets[focused_preds]
    preds = preds if not focused_preds else preds[focused_preds]

    m = X.size(0)

    errors = targets - preds
    losses = torch.stack((errors.T * X[:, 0], errors.T * X[:, 1]), dim=0)
    total_loss = torch.nansum(losses, dim=1)

    font_size = 30

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
            f"= <b>{(1 / m) * lr * total_loss[0]:.3f}</b>",
            f"<b>1/{m} * {lr} * {total_loss[1]:.3f}</b>",
            f"= <b>{(1 / m) * lr * total_loss[1]:.3f}</b>",
        ],
        font=dict(size=font_size, color=["black", "white", "black", "white"]),
        line=dict(width=2),
        fill_color=[
            feature_colors[0],
            "grey" if focus_costs else "black",
            feature_colors[1],
            "grey" if focus_costs else "black",
        ],
    )

    align = ["center", "left", "center", "left"]

    return go.Table(
        header=dict(**header, align=align, height=44),
        cells=dict(**cells, align=align, height=44),
        columnorder=[0, 1, 2, 3],
        columnwidth=[3, 1.2, 3, 1.2],
        meta=meta,
    )
