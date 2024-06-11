from base import Animation, Frame


import torch


def neural_network_annotations(frame: Frame, animation: Animation, show=True):
    if not show:
        return []

    X = frame.X
    m = X.size(0)
    w = frame.w
    targets = frame.targets
    preds = frame.preds
    errors = frame.errors
    derivatives = frame.derivatives
    losses = frame.losses
    costs = frame.costs

    modules = frame.modules
    size = frame.size
    node_points = frame.node_points
    focused_node = frame.focused_node

    focused_inputs = frame.focused_inputs
    focused_targets = frame.focused_targets
    focused_preds = frame.focused_preds
    focused_errors = frame.focused_errors
    focused_losses = frame.focused_losses

    focus_costs = frame.focus_costs

    colors = {}
    for module in modules:
        colors[module] = animation.colors(size[module])

    target_color = animation.theme.target_color
    target_text_color = animation.theme.target_text_color

    note_height = 40

    max_notes = 0
    width = 70
    operator_width = 20
    border_width = 1
    x_yshift = 80

    error_color = "rgba(255,255,255,.5)"
    error_text_color = "red"
    operator_color = "rgba(255,255,255,.5)"
    operator_text_color = "black"

    def calc_yshift(index, size):
        return -(index * note_height)

    def annotate_value(
        node,
        value,
        index,
        focused_size,
        width,
        xshift,
        initial_yshift,
        feature_color,
        text_color=None,
        text_angle=0,
        border_color="rgba(0,0,0,.8)",
        border_width=0,
    ):
        yshift = calc_yshift(index, focused_size) + initial_yshift

        text = f"<b>{value:1.2f}</b>" if isinstance(value, (float, int, torch.Tensor)) else value

        note = dict(
            x=node[0],
            y=node[1],
            height=40,
            text=text,
            align="center",
            font_color=text_color,
            xanchor="left",
            yshift=yshift,
            xshift=xshift,
            width=width,
            textangle=text_angle,
            bgcolor=feature_color,
            borderwidth=border_width,
            bordercolor=border_color,
            showarrow=False,
            visible=True,
        )

        notes.append(note)

    def annotate_nodes(nodes, values, focused, width, xshift, initial_yshift, colors, text_color=None, text_angle=0):
        for i, indices in enumerate(focused):
            focused_size = len(indices)
            for j, index in enumerate(indices):
                if isinstance(values, str):
                    value = values
                else:
                    value = values[i][index]
                node = nodes[i]
                feature_color = colors[i]
                annotate_value(
                    node, value, j, focused_size, width, xshift, initial_yshift, feature_color, text_color, text_angle
                )

    notes = []

    max_input_notes = m * size["input"] * 1
    input_yshift = 0
    input_xshift = 0
    input_yshift -= x_yshift
    if any(x for x in focused_inputs):
        input_xshift -= (width + (border_width * 2)) / 2

        annotate_nodes(
            node_points["input"],
            X.T,
            focused_inputs,
            width,
            input_xshift,
            input_yshift,
            colors["input"],
            "black",
        )

        input_xshift += width + (border_width * 2)

    if any(x for x in focused_losses["input"]) and focused_node["hidden"]:
        input_xshift -= ((width + (border_width * 2)) * 3) / 2
        input_xshift -= ((operator_width + (border_width * 2)) * 2) / 2

        annotate_nodes(
            node_points["input"],
            errors["hidden"][focused_node["hidden"]].unsqueeze(0).expand((size["hidden"], m)),
            focused_losses["input"],
            width,
            input_xshift,
            input_yshift,
            [error_color] * size["input"],
            error_text_color,
        )

        input_xshift += width + (border_width * 2)

        annotate_nodes(
            node_points["input"],
            "*",
            focused_losses["input"],
            operator_width,
            input_xshift,
            input_yshift,
            [operator_color] * size["input"],
            operator_text_color,
        )

        input_xshift += operator_width + (border_width * 2)

        annotate_nodes(
            node_points["input"],
            X.T,
            focused_losses["input"],
            width,
            input_xshift,
            input_yshift,
            colors["input"],
            "black",
        )

        input_xshift += width + (border_width * 2)

        annotate_nodes(
            node_points["input"],
            "=",
            focused_losses["input"],
            operator_width,
            input_xshift,
            input_yshift,
            [operator_color] * size["input"],
            operator_text_color,
        )

        input_xshift += operator_width + (border_width * 2)

        annotate_nodes(
            node_points["input"],
            losses["hidden"],
            focused_losses["input"],
            width,
            input_xshift,
            input_yshift,
            colors["hidden"],
            "red",
        )

        input_xshift += width + (border_width * 2)

    max_hidden_notes = 0

    if len(modules) > 2:
        max_hidden_notes = (m * size["hidden"] * 5) + 2
        hidden_cost_xshift = 0
        hidden_cost_yshift = 0
        hidden_x_yshift = 0
        hidden_x_xshift = 0

        hidden_cost_yshift -= note_height + 40
        if focus_costs:
            for i in range(size["hidden"]):
                hidden_cost_xshift = -(
                    (((width + (border_width * 2)) * size["input"]) + ((size["input"] - 1) * 20)) / 2
                )
                for j in range(size["input"]):
                    hidden_cost = costs["hidden"][i][j]

                    annotate_value(
                        node_points["hidden"][i],
                        hidden_cost,
                        0,
                        1,
                        width,
                        hidden_cost_xshift,
                        hidden_cost_yshift,
                        colors["input"][j],
                        error_text_color,
                    )

                    hidden_cost_xshift += width + (border_width * 2) + 20

            hidden_x_yshift += hidden_cost_yshift + 20

        hidden_x_yshift -= x_yshift

        if any(x for x in focused_preds["hidden"]):
            hidden_x_xshift -= (width + (border_width * 2)) / 2

            annotate_nodes(
                node_points["hidden"],
                preds["hidden"],
                focused_preds["hidden"],
                width,
                hidden_x_xshift,
                hidden_x_yshift,
                colors["hidden"],
                "black",
            )

            hidden_x_xshift += width + (border_width * 2)

        if any(x for x in focused_losses["hidden"]) and focused_node["output"] is not None:
            hidden_x_xshift -= ((width + (border_width * 2)) * 3) / 2
            hidden_x_xshift -= ((operator_width + (border_width * 2)) * 2) / 2

            annotate_nodes(
                node_points["hidden"],
                errors["output"][focused_node["output"]].unsqueeze(0).expand((size["hidden"], m)),
                focused_losses["hidden"],
                width,
                hidden_x_xshift,
                hidden_x_yshift,
                [error_color] * size["hidden"],
                error_text_color,
            )

            hidden_x_xshift += width + (border_width * 2)

            annotate_nodes(
                node_points["hidden"],
                "*",
                focused_losses["hidden"],
                operator_width,
                hidden_x_xshift,
                hidden_x_yshift,
                [operator_color] * size["hidden"],
                operator_text_color,
            )

            hidden_x_xshift += operator_width + (border_width * 2)

            annotate_nodes(
                node_points["hidden"],
                preds["hidden"],
                focused_losses["hidden"],
                width,
                hidden_x_xshift,
                hidden_x_yshift,
                colors["hidden"],
                "black",
            )

            hidden_x_xshift += width + (border_width * 2)

            annotate_nodes(
                node_points["hidden"],
                "=",
                focused_losses["hidden"],
                operator_width,
                hidden_x_xshift,
                hidden_x_yshift,
                [operator_color] * size["hidden"],
                operator_text_color,
            )

            hidden_x_xshift += operator_width + (border_width * 2)

            annotate_nodes(
                node_points["hidden"],
                losses["output"][focused_node["output"]],
                focused_losses["hidden"],
                width,
                hidden_x_xshift,
                hidden_x_yshift,
                colors["hidden"],
                "red",
            )

        if any(x for x in focused_errors["hidden"]) and focused_node["output"] is not None:
            hidden_x_xshift -= ((width + (border_width * 2)) * 4) / 2
            hidden_x_xshift -= ((operator_width + (border_width * 2)) * 3) / 2

            annotate_nodes(
                node_points["hidden"],
                errors["output"][focused_node["output"]].unsqueeze(0).expand((size["hidden"], m)),
                focused_errors["hidden"],
                width,
                hidden_x_xshift,
                hidden_x_yshift,
                [error_color] * size["hidden"],
                error_text_color,
            )

            hidden_x_xshift += width + (border_width * 2)

            annotate_nodes(
                node_points["hidden"],
                "*",
                focused_errors["hidden"],
                operator_width,
                hidden_x_xshift,
                hidden_x_yshift,
                [operator_color] * size["hidden"],
                operator_text_color,
            )

            hidden_x_xshift += operator_width + (border_width * 2)

            for i, indices in enumerate(focused_errors["hidden"]):
                if len(indices):
                    annotate_value(
                        node_points["hidden"][i],
                        w["output"][focused_node["output"]][i],
                        0,
                        1,
                        width,
                        hidden_x_xshift,
                        hidden_x_yshift - (((len(indices) * note_height) / 2) - (note_height / 2)),
                        colors["input"][i],
                        "black",
                        border_color="black",
                        border_width=1,
                    )

            hidden_x_xshift += width + (border_width * 2)

            annotate_nodes(
                node_points["hidden"],
                "*",
                focused_errors["hidden"],
                operator_width,
                hidden_x_xshift,
                hidden_x_yshift,
                [operator_color] * size["hidden"],
                operator_text_color,
            )

            hidden_x_xshift += operator_width + (border_width * 2)

            annotate_nodes(
                node_points["hidden"],
                derivatives["hidden"],
                focused_errors["hidden"],
                width,
                hidden_x_xshift,
                hidden_x_yshift,
                colors["input"],
                "black",
                text_angle=-15,
            )

            hidden_x_xshift += width + (border_width * 2)

            annotate_nodes(
                node_points["hidden"],
                "=",
                focused_errors["hidden"],
                operator_width,
                hidden_x_xshift,
                hidden_x_yshift,
                [operator_color] * size["hidden"],
                operator_text_color,
            )

            hidden_x_xshift += operator_width + (border_width * 2)

            annotate_nodes(
                node_points["hidden"],
                errors["hidden"],
                focused_errors["hidden"],
                width,
                hidden_x_xshift,
                hidden_x_yshift,
                colors["input"],
                error_text_color,
            )

            hidden_x_xshift += width + (border_width * 2)

    max_output_notes = (m * size["output"] * 5) + 2
    output_cost_xshift = 0
    output_cost_yshift = 0
    output_x_xshift = 0
    output_x_yshift = 0

    hidden = "hidden" if "hidden" in modules else "input"

    if focus_costs:
        output_cost_yshift -= note_height + 40
        for i in range(size["output"]):
            output_cost_xshift = -((((width + (border_width * 2)) * size[hidden]) + ((size[hidden] - 1) * 20)) / 2)
            for j in range(size[hidden]):
                cost = costs["output"][i][j]

                annotate_value(
                    node_points["output"][i],
                    cost,
                    0,
                    1,
                    width,
                    output_cost_xshift,
                    output_cost_yshift,
                    colors[hidden][j],
                    error_text_color,
                )

                output_cost_xshift += width + (border_width * 2) + 20

        output_x_yshift += output_cost_yshift + 20

    output_x_yshift -= x_yshift

    if any(focused_targets):
        output_x_xshift -= (width + (border_width * 2)) / 2

        annotate_nodes(
            node_points["output"],
            targets,
            focused_targets,
            width,
            output_x_xshift,
            output_x_yshift,
            [target_color] * size["output"],
            target_text_color,
        )

    if any(x for x in focused_preds["output"]):
        output_x_xshift -= (width + (border_width * 2)) / 2

        annotate_nodes(
            node_points["output"],
            preds["output"],
            focused_preds["output"],
            width,
            output_x_xshift,
            output_x_yshift,
            colors["output"],
            "black",
        )

    if any(x for x in focused_errors["output"]) and focused_node[hidden] is not None:
        output_x_xshift -= ((width + (border_width * 2)) * 3) / 2
        output_x_xshift -= ((operator_width + (border_width * 2)) * 2) / 2

        annotate_nodes(
            node_points["output"],
            preds["output"],
            focused_errors["output"],
            width,
            output_x_xshift,
            output_x_yshift,
            colors["output"],
            "black",
        )

        output_x_xshift += width + (border_width * 2)

        annotate_nodes(
            node_points["output"],
            "-",
            focused_errors["output"],
            operator_width,
            output_x_xshift,
            output_x_yshift,
            [operator_color] * size[hidden],
            operator_text_color,
        )

        output_x_xshift += operator_width + (border_width * 2)

        annotate_nodes(
            node_points["output"],
            targets,
            focused_errors["output"],
            width,
            output_x_xshift,
            output_x_yshift,
            [target_color] * size["output"],
            target_text_color,
        )

        output_x_xshift += width + (border_width * 2)

        annotate_nodes(
            node_points["output"],
            "=",
            focused_errors["output"],
            operator_width,
            output_x_xshift,
            output_x_yshift,
            [operator_color] * size["output"],
            operator_text_color,
        )

        output_x_xshift += operator_width + (border_width * 2)

        annotate_nodes(
            node_points["output"],
            errors["output"][focused_node["output"]].unsqueeze(0).expand((size["output"], m)),
            focused_errors["output"],
            width,
            output_x_xshift,
            output_x_yshift,
            [error_color] * size["output"],
            error_text_color,
        )

        output_x_xshift += width + (border_width * 2)

    # we need to pad the notes with empty ones to our total block size otherwise previous notes will still render even
    # though they aren't returned here
    no_note = dict(text="", x=0, y=0, yshift=note_height + 20, bgcolor="rgba(0,0,0,0)", showarrow=False, visible=False)
    max_notes = max_input_notes + max_hidden_notes + max_output_notes
    if len(notes) < max_notes:
        notes.extend([no_note] * (max_notes - len(notes)))

    return notes
