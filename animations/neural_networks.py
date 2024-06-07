import torch

from base import Frame
from learning import predict
from numpy import linspace as ls

from utils import clone

all_chapters = ["logistic", "xor", "neural", "fit", "weights"]


def get_animation(
    chapters=None,
    resolution=20,
):
    if chapters is None:
        chapters = all_chapters

    def capture(count=1):
        frame = Frame(
            X=X,
            targets=targets,
            preds=clone(preds),
            derivatives=clone(derivatives),
            errors=clone(errors),
            losses=clone(losses),
            costs=clone(costs),
            w=clone(w),
            b=clone(b),
            epochs=epochs,
            learning_rate=learning_rate,
            size=size,
            bias_zrange=bias_zrange,
            domain_padding=0.4,
            range_padding=0.5,
            eye=eye,
            weight_eyes=weight_eyes,
            bias_eye=bias_eye,
            inference=inference.clone() if inference is not None else None,
            activity=activity,
            focused_node=focused_node,
            focused_connections=clone(focused_connections),
            focused_feature=focused_feature,
            focused_inputs=clone(focused_inputs),
            focused_targets=clone(focused_targets),
            focused_preds=clone(focused_preds),
            focused_errors=clone(focused_errors),
            focused_losses=clone(focused_losses),
            focus_labels=focus_labels,
            focus_targets=focus_targets,
            show_preds=show_preds,
            show_profile=show_profile,
            show_decision_boundaries=show_decision_boundaries,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )

        frames.extend([frame] * count)

    frames = []

    torch.manual_seed(42)

    size = {"input": 2, "hidden": 2, "output": 1}

    xor_X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_targets = torch.Tensor([[0, 1, 1, 0]])
    inference = None

    intial_eye = (1.2, -0.8, 1)
    initial_weight_eyes = ((1, -1, 0), (1, -1, 0))

    initial_w = {
        "hidden": torch.randn(size["hidden"], size["input"]),
        "output": torch.randn(size["output"], size["hidden"]),
    }
    initial_b = {"hidden": torch.zeros((size["hidden"], 1)), "output": torch.zeros((size["output"], 1))}

    final_w = {
        "output": torch.Tensor([[-14.2149982452, 12.6667270660]]),
        "hidden": torch.Tensor([[4.4657111168, 4.4657068253], [10.1644496918, 10.1644334793]]),
    }
    final_b = {"output": torch.Tensor([[-5.3260169029]]), "hidden": torch.Tensor([[-6.7141456604], [-6.7141456604]])}

    eye = intial_eye
    hidden_weight_eyes = ((0, 1, 0), (1, 0, 0))

    bias_eye = (1, -1, 0)
    bias_zrange = (-5.5, 6.6)

    activity = 1
    epochs = 30
    learning_rate = 1

    aspect_ratio = (1.8, 1.8, 0.6)
    # final_eye = (2, 0.64, 0)
    # final_aspect_ratio = tuple(x + 0.2 for x in aspect_ratio)

    preds = {"output": None, "hidden": None}
    derivatives = {"output": None, "hidden": None}
    errors = {"output": None, "hidden": None}
    losses = {"output": [[] for _ in range(size["output"])], "hidden": [[] for _ in range(size["hidden"])]}
    costs = {
        "output": torch.zeros(size["output"], size["hidden"]),
        "hidden": torch.zeros(size["hidden"], size["input"]),
    }

    focused_node = {"input": None, "hidden": None, "output": None}
    focused_connections = {"input": [], "hidden": [], "output": []}

    focused_inputs = [[] for _ in range(size["input"])]
    focused_targets = [[] for _ in range(size["output"])]
    focused_preds = {"output": [[] for _ in range(size["output"])], "hidden": [[] for _ in range(size["hidden"])]}
    focused_errors = {"output": [[] for _ in range(size["output"])], "hidden": [[] for _ in range(size["hidden"])]}
    focused_losses = {"hidden": [[] for _ in range(size["hidden"])], "input": [[] for _ in range(size["input"])]}

    activity = 1
    focused_feature = None
    focus_labels = None
    focus_targets = False
    show_preds = True
    show_profile = False
    show_decision_boundaries = False

    if "logistic" in chapters:
        X = torch.Tensor([[1, 0], [0, 1]])
        targets = torch.Tensor([[0, 1]])
        w = {"output": torch.Tensor([[0, 0]])}
        b = {"output": torch.Tensor([[0.5]])}
        preds["output"] = predict(X, w["output"], b["output"])

        weight_eyes = hidden_weight_eyes

        capture()

        m = X.size(0)
        for _ in range(epochs):
            w["output"] -= learning_rate * ((1 / m) * ((preds["output"] - targets) @ X))
            b["output"] -= learning_rate * ((1 / m) * torch.sum(preds["output"] - targets))
            preds["output"] = predict(X, w["output"], b["output"])
            capture()

        focused_errors["output"] = [list(range(m)) for _ in range(size["output"])]

        capture(10)

        focused_errors["output"] = [[] for _ in range(size["output"])]

    if "xor" in chapters:
        X = xor_X
        targets = xor_targets
        w = {"output": torch.Tensor([[0, 0]])}
        b = {"output": torch.Tensor([[0.5]])}
        preds["output"] = predict(X, w["output"], b["output"])

        weight_eyes = hidden_weight_eyes

        capture()

        m = X.size(0)
        for _ in range(epochs):
            w["output"] -= learning_rate * ((1 / m) * ((preds["output"] - targets) @ X))
            b["output"] -= learning_rate * ((1 / m) * torch.sum(preds["output"] - targets))
            preds["output"] = predict(X, w["output"], b["output"])
            capture()

        capture(10)

    if "neural" in chapters:
        X = xor_X
        m = X.size(0)
        targets = xor_targets
        w = clone(initial_w)
        b = clone(initial_b)

        preds["hidden"] = predict(X, w["hidden"], b["hidden"])
        preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])

        derivatives["output"] = preds["output"] * (1 - preds["output"])
        derivatives["hidden"] = preds["hidden"] * (1 - preds["hidden"])

        errors["output"] = preds["output"] - targets
        errors["hidden"] = (w["output"].T @ errors["output"]) * (preds["hidden"] * (1 - preds["hidden"]))

        weight_eyes = initial_weight_eyes

        capture()

        # show forward propagation

        # focus inputs and targets one by one
        for i in range(m):
            for j in range(size["input"]):
                focused_inputs[j].append(i)
            for j in range(size["output"]):
                focused_targets[j].append(i)
            capture()

        focused_targets = [[] for _ in range(size["output"])]

        focused_connections["input"] = list(range(size["input"]))

        # focus hidden predictions one by one
        for i in range(m):
            for j in range(size["input"]):
                focused_inputs[j].remove(i)

            # focus input to hidden connections before showing each hidden prediction
            for j in range(size["hidden"]):
                focused_connections["hidden"].append(j)
                capture()
                focused_connections["hidden"].remove(j)

            for j in range(size["hidden"]):
                focused_preds["hidden"][j].append(i)
            capture()

        focused_connections["input"] = []

        focused_connections["hidden"] = list(range(size["hidden"]))

        # focus output predictions one by one
        for i in range(m):
            for j in range(size["hidden"]):
                focused_preds["hidden"][j].remove(i)

            # focus hidden to output connections before showing each output prediction
            for j in range(size["output"]):
                focused_connections["output"].append(j)
                capture()
                focused_connections["output"].remove(j)

            for j in range(size["output"]):
                focused_preds["output"][j].append(i)
            capture()

        focused_connections["hidden"] = []

        focused_preds["output"] = [[] for _ in range(size["output"])]

        # show backpropagation

        # focus output errors one by one
        for i in range(m):
            for j in range(size["output"]):
                focused_errors["output"][j].append(i)
            capture()

        # for each output node, show it's contribution to the loss
        for i in range(size["output"]):
            focused_node["output"] = i

            losses["output"][focused_node["output"]] = errors["output"][focused_node["output"]] * preds["hidden"]

            focused_connections["output"] = list(range(size["output"]))

            # focus hidden errors one by one
            for j in range(m):
                focused_errors["output"][i].remove(j)
                for k in range(size["hidden"]):
                    # focus output to hidden connections before showing each hidden error
                    for ii in range(size["hidden"]):
                        focused_connections["hidden"].append(ii)
                        capture()
                        focused_connections["hidden"].remove(ii)
                    focused_losses["hidden"][k].append(j)
                capture()

            focused_connections["output"] = []

            focused_connections["hidden"] = list(range(size["hidden"]))

            # update current output node's cost(s) one by one
            for j in range(m):
                for k in range(size["hidden"]):
                    # focus hidden to output connections before updating each data point's contribution
                    for ii in range(size["output"]):
                        focused_connections["output"].append(ii)
                        capture()
                        focused_connections["output"].remove(ii)
                    focused_losses["hidden"][k].remove(j)
                # update the output cost
                costs["output"][focused_node["output"]] += losses["output"][focused_node["output"]].T[j]
                capture()

            focused_connections["hidden"] = []

            # focus hidden errors one by one again
            for j in range(m):
                for k in range(size["hidden"]):
                    capture()
                    focused_errors["hidden"][k].append(j)
                capture()

            # for each hidden node, show it's contribution to the loss
            for j in range(size["hidden"]):
                focused_node["hidden"] = j

                losses["hidden"] = errors["hidden"][focused_node["hidden"]] * X.T

                focused_connections["hidden"] = [focused_node["hidden"]]

                # focus input losses one by one
                for k in range(m):
                    focused_errors["hidden"][j].remove(k)
                    for ii in range(size["input"]):
                        # focus hidden to input connections before showing each input loss
                        for ij in range(size["input"]):
                            focused_connections["input"].append(ij)
                            capture()
                            focused_connections["input"].remove(ij)
                        focused_losses["input"][ii].append(k)
                    capture()

                focused_connections["hidden"] = []

                focused_connections["input"] = list(range(size["input"]))

                # update current hidden node's cost(s) one by one
                for k in range(m):
                    for ii in range(size["input"]):
                        # focus input to hidden connections before updating each data point's contribution
                        focused_connections["hidden"].append(focused_node["hidden"])
                        capture()
                        focused_connections["hidden"].remove(focused_node["hidden"])
                        focused_losses["input"][ii].remove(k)
                    costs["hidden"][focused_node["hidden"]] += losses["hidden"].T[k]
                    capture()

                focused_connections["input"] = []

    if "fit" in chapters:
        X = xor_X
        targets = xor_targets

        m = X.size(0)

        w = clone(initial_w)
        b = clone(initial_b)

        preds["hidden"] = predict(X, w["hidden"], b["hidden"])
        preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])

        weight_eyes = initial_weight_eyes

        capture()

        learning_rate = 2.5
        epochs = 1000
        for i in range(epochs):
            errors["output"] = preds["output"] - targets
            dw = errors["output"] @ preds["hidden"].T
            db = torch.sum(errors["output"])
            errors["hidden"] = (w["output"].T @ errors["output"]) * (preds["hidden"] * (1 - preds["hidden"]))
            dw1 = errors["hidden"] @ X
            db1 = torch.sum(errors["hidden"])

            w["output"] -= (1 / m) * learning_rate * dw
            b["output"] -= (1 / m) * learning_rate * db
            w["hidden"] -= (1 / m) * learning_rate * dw1
            b["hidden"] -= (1 / m) * learning_rate * db1

            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])

            # get 30 frames given any number of epochs
            count = epochs // 30
            remainder = epochs % count
            if i % count == remainder:
                capture()

    # show the effect of changing weights and biases in a trained neural network
    if "weights" in chapters:
        X = xor_X
        targets = xor_targets
        w = clone(final_w)
        b = clone(final_b)

        preds["hidden"] = predict(X, w["hidden"], b["hidden"])
        preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])

        weight_eyes = initial_weight_eyes

        capture()

        eye = None

        # set all output weights to 0
        for i, j, k in zip(
            ls(final_w["output"][0][0], 0, 10), ls(final_w["output"][0][1], 0, 10), ls(final_b["output"][0][0], 0, 10)
        ):
            w["output"][0][0] = i
            w["output"][0][1] = j
            b["output"][0][0] = k
            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])
            capture()

        # lift weight 1 up to 10
        for i in ls(0, 10, 10):
            w["output"][0][0] = i
            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])
            capture()

        # bring weight 1 down to -10
        for i in ls(10, -10, 20):
            w["output"][0][0] = i
            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])
            capture()

        capture(10)

        # reset weight 1 to 0
        for i in ls(-10, 0, 10):
            w["output"][0][0] = i
            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])
            capture()

        # lift weight 1 up to 10
        for i in ls(0, 10, 10):
            w["output"][0][1] = i
            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])
            capture()

        # bring weight 1 down to -10
        for i in ls(10, -10, 20):
            w["output"][0][1] = i
            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])
            capture()

        capture(10)

        # reset all output weights to 0
        for i, j, k in zip(ls(w["output"][0][0], 0, 10), ls(w["output"][0][1], 0, 10), ls(b["output"][0][0], 0, 10)):
            w["output"][0][0] = i
            w["output"][0][1] = j
            b["output"][0][0] = k
            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])
            capture()

        # bring down weight 0 to -10
        for i in ls(0, -10, 10):
            w["output"][0][0] = i
            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])
            capture()

        # lift up weight 1 to 10
        for i in ls(0, 10, 10):
            w["output"][0][1] = i
            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])
            capture()

        capture(5)

        # bring down bias to -5 to show a fit model
        for i in ls(0, -5, 10):
            b["output"][0][0] = i
            preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])
            capture()

        capture(10)

    return frames
