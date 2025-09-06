import torch

import plotly.graph_objs as go

from base import (
    AnimationFrame,
    Plot3DFrame,
    LayoutComponent,
    ParameterView,
    get_domain_surface,
    parse_param_index,
    parse_param_module,
)
from learning import bce_loss, predict


class GradientComponent(LayoutComponent):
    names = ["Gradient"]
    types = ["scene"]
    height = 768

    def create_component(self, frame: AnimationFrame, view: ParameterView) -> list[Plot3DFrame]:
        return []

    def update_component(self, frame: AnimationFrame, view: ParameterView) -> list[Plot3DFrame]:
        return [
            Plot3DFrame(
                data=gradients(
                    frame.X,
                    frame.targets,
                    frame.surface_points,
                    frame.surface_linspace,
                    frame.w,
                    frame.b,
                    frame.modules,
                    frame.activation_fns,
                    param1=["output_1", 0],
                    param2=["output_1", 1],
                    show_profile=False,
                    res=20,
                )
            )
        ]


def gradients(
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
    loss_function=bce_loss,
    activity=1,
    show_profile=False,
    res=20,
    meta=None,
):
    domain = torch.Tensor([(-10, 10), (-10, 10)])
    surface_linspace, surface_points = get_domain_surface(domain, res)

    sx = surface_linspace[:, 0]
    sy = surface_linspace[:, 1]

    # skip input module
    modules = modules[1:]

    losses = []

    module_1 = parse_param_module(param1[0])
    module_2 = parse_param_module(param2[0])
    module_1_index = parse_param_index(param1[0])
    module_2_index = parse_param_index(param2[0])

    for point in surface_points:
        for i, module in enumerate(modules):
            module_w = w[module].clone()
            if module == module_1:
                module_w[module_1_index][param1[1]] = point[0]
            elif module == module_2:
                module_w[module_2_index][param2[1]] = point[1]
            activity = activity if i == len(modules) - 1 else None

            preds = predict(
                X,
                module_w,
                b[module],
                activation=activations[module],
                activity=activity,
            )

            losses.append(torch.nansum(loss_function(preds, targets)))

    losses = torch.stack(losses)
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
