import random
import torch

from base import AnimationFrame
from learning import predict, sigmoid, sigmoid_derivative
from numpy import linspace as ls
import numpy as np

from utils import clone

from sklearn.datasets import make_circles

all_chapters = ["fail-gaussian"]


def get_animation(
    chapters=None,
    resolution=20,
):
    if chapters is None:
        chapters = all_chapters

    def capture(count=1):
        frame = AnimationFrame(
            X=X,
            targets=targets,
            preds=clone(preds),
            derivatives=clone(derivatives),
            errors=clone(errors),
            losses=clone(losses),
            costs=clone(costs),
            w=clone(w),
            b=clone(b),
            activation_fns=clone(activations),
            epochs=epochs,
            learning_rate=learning_rate,
            modules=clone(modules),
            size=size,
            bias_zrange=bias_zrange,
            domain_padding=0.4,
            range_padding=0.5,
            eye=eye,
            weight_eyes=weight_eyes,
            bias_eye=bias_eye,
            inference=inference.clone() if inference is not None else None,
            activity=activity,
            active_preds=clone(active_preds),
            active_errors=clone(active_errors),
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
            focus_costs=focus_costs,
            show_preds=show_preds,
            show_profile=show_profile,
            show_decision_boundaries=show_decision_boundaries,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )

        frames.extend([frame] * count)

    frames = []

    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(0)

    activations = {"hidden": sigmoid, "output": sigmoid}

    modules = ["input", "hidden", "output"]
    size = {"input": 2, "hidden": 2, "output": 1}

    xor_X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_targets = torch.Tensor([[0, 1, 1, 0]])
    inference = None

    initial_eye = (1.2, -0.8, 1)

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

    eye = initial_eye
    weight_eyes = None
    neuron_weight_eyes = ((0, 1, 0), (1, 0, 0))
    xor_weight_eyes = ((1, -1, 0), (1, -1, 0))

    bias_eye = (1, -1, 0)
    bias_zrange = (-5.5, 6.6)

    activity = 1
    epochs = 30
    learning_rate = 1

    aspect_ratio = (1.8, 1.8, 0.6)

    preds = {"output": None, "hidden": None}
    derivatives = {"output": None, "hidden": None}
    errors = {"output": None, "hidden": None}
    losses = {"output": [[] for _ in range(size["output"])], "hidden": [[] for _ in range(size["hidden"])]}
    costs = {
        "output": torch.zeros(size["output"], size["hidden"]),
        "hidden": torch.zeros(size["hidden"], size["input"]),
    }

    initial_costs = clone(costs)

    active_preds = {"output": [[] for _ in range(size["output"])], "hidden": [[] for _ in range(size["hidden"])]}
    active_errors = {"output": [[] for _ in range(size["output"])], "hidden": [[] for _ in range(size["hidden"])]}

    focused_node = {"input": None, "hidden": None, "output": None}
    focused_connections = {"input": [], "hidden": [], "output": []}

    focused_inputs = [[] for _ in range(size["input"])]
    focused_targets = [[] for _ in range(size["output"])]
    focused_preds = {"output": [[] for _ in range(size["output"])], "hidden": [[] for _ in range(size["hidden"])]}
    focused_errors = {"output": [[] for _ in range(size["output"])], "hidden": [[] for _ in range(size["hidden"])]}
    focused_losses = {"hidden": [[] for _ in range(size["hidden"])], "input": [[] for _ in range(size["input"])]}

    activity = 1
    focused_feature = None
    focus_costs = False
    focus_labels = None
    focus_targets = False
    show_preds = True
    show_profile = False
    show_decision_boundaries = False

    if "fail-gaussian" in chapters:
        torch.manual_seed(20)

        eye = initial_eye

        X, targets = make_circles(30, factor=0.1, noise=0.1)

        X = torch.from_numpy(X).float()
        targets = torch.from_numpy(targets).float().unsqueeze(0)
        m = X.size(0)

        size = {"input": 2, "hidden": 3, "output": 1}

        w = {
            "hidden": torch.randn(size["hidden"], size["input"]),
            "output": torch.randn(size["output"], size["hidden"]),
        }

        b = {
            "hidden": torch.zeros((size["hidden"], 1)),
            "output": torch.zeros((size["output"], 1)),
        }

        activations = {"hidden": sigmoid, "output": sigmoid}

        preds["hidden"] = predict(X, w["hidden"], b["hidden"])
        preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])

        learning_rate = 2.5

        costs = {
            "output": torch.zeros(size["output"], size["hidden"]),
            "hidden": torch.zeros(size["hidden"], size["input"]),
        }

        weight_eyes = None

        capture()

        learning_rate = 2.5
        epochs = 1000
        for i in range(epochs):
            errors["output"] = preds["output"] - targets

            dw = errors["output"] @ preds["hidden"].T
            db = torch.sum(errors["output"])
            costs["output"] = (1 / m) * learning_rate * dw

            errors["hidden"] = (w["output"].T @ errors["output"]) * (preds["hidden"] * (1 - preds["hidden"]))

            dw1 = errors["hidden"] @ X
            db1 = torch.sum(errors["hidden"])
            costs["hidden"] = (1 / m) * learning_rate * dw1

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

        # show profile of the model
        for z in ls(eye[2], 3, 10):
            eye = (eye[0], eye[1], z)
            capture()

        torch.manual_seed(42)

    if "gaussian" in chapters:
        torch.manual_seed(40)

        eye = initial_eye

        X, targets = make_circles(30, factor=0.1, noise=0.1)

        X = torch.from_numpy(X).float()
        targets = torch.from_numpy(targets).float().unsqueeze(0)
        m = X.size(0)

        size = {"input": 2, "hidden": 3, "output": 1}

        w = {
            "hidden": torch.randn(size["hidden"], size["input"]),
            "output": torch.randn(size["output"], size["hidden"]),
        }

        b = {
            "hidden": torch.zeros((size["hidden"], 1)),
            "output": torch.zeros((size["output"], 1)),
        }

        activations = {"hidden": sigmoid, "output": sigmoid}

        preds["hidden"] = predict(X, w["hidden"], b["hidden"])
        preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])

        learning_rate = 2.5

        costs = {
            "output": torch.zeros(size["output"], size["hidden"]),
            "hidden": torch.zeros(size["hidden"], size["input"]),
        }

        weight_eyes = None

        capture()

        learning_rate = 2.5
        epochs = 1000
        for i in range(epochs):
            errors["output"] = preds["output"] - targets

            dw = errors["output"] @ preds["hidden"].T
            db = torch.sum(errors["output"])
            costs["output"] = (1 / m) * learning_rate * dw

            errors["hidden"] = (w["output"].T @ errors["output"]) * (preds["hidden"] * (1 - preds["hidden"]))

            dw1 = errors["hidden"] @ X
            db1 = torch.sum(errors["hidden"])
            costs["hidden"] = (1 / m) * learning_rate * dw1

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

        # show profile of the model
        for z in ls(eye[2], 3, 10):
            eye = (eye[0], eye[1], z)
            capture()

        torch.manual_seed(42)

    if "sgd" in chapters:
        torch.manual_seed(20)

        eye = initial_eye

        X, targets = make_circles(30, factor=0.1, noise=0.1)

        X = torch.from_numpy(X).float()
        targets = torch.from_numpy(targets).float().unsqueeze(0)
        m = X.size(0)

        size = {"input": 2, "hidden": 3, "output": 1}

        w = {
            "hidden": torch.randn(size["hidden"], size["input"]),
            "output": torch.randn(size["output"], size["hidden"]),
        }

        b = {
            "hidden": torch.zeros((size["hidden"], 1)),
            "output": torch.zeros((size["output"], 1)),
        }

        activations = {"hidden": sigmoid, "output": sigmoid}

        preds["hidden"] = predict(X, w["hidden"], b["hidden"])
        preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])

        learning_rate = 2.5

        costs = {
            "output": torch.zeros(size["output"], size["hidden"]),
            "hidden": torch.zeros(size["hidden"], size["input"]),
        }

        weight_eyes = None

        capture()

        learning_rate = 2.5
        epochs = 1000
        batch_size = 5
        for i in range(epochs):
            for j in range(0, m, batch_size):
                X_batch = X[j : j + batch_size]
                targets_batch = targets[:, j : j + batch_size]

                preds["hidden"] = predict(X_batch, w["hidden"], b["hidden"])
                preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])

                errors["output"] = preds["output"] - targets_batch

                dw = errors["output"] @ preds["hidden"].T
                db = torch.sum(errors["output"])
                costs["output"] = (1 / batch_size) * learning_rate * dw

                errors["hidden"] = (w["output"].T @ errors["output"]) * (preds["hidden"] * (1 - preds["hidden"]))

                dw1 = errors["hidden"] @ X_batch
                db1 = torch.sum(errors["hidden"])
                costs["hidden"] = (1 / batch_size) * learning_rate * dw1

                w["output"] -= (1 / batch_size) * learning_rate * dw
                b["output"] -= (1 / batch_size) * learning_rate * db
                w["hidden"] -= (1 / batch_size) * learning_rate * dw1
                b["hidden"] -= (1 / batch_size) * learning_rate * db1

                preds["hidden"] = predict(X, w["hidden"], b["hidden"])
                preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])

            # errors["output"] = preds["output"] - targets

            # dw = errors["output"] @ preds["hidden"].T
            # db = torch.sum(errors["output"])
            # costs["output"] = (1 / m) * learning_rate * dw

            # errors["hidden"] = (w["output"].T @ errors["output"]) * (preds["hidden"] * (1 - preds["hidden"]))

            # dw1 = errors["hidden"] @ X
            # db1 = torch.sum(errors["hidden"])
            # costs["hidden"] = (1 / m) * learning_rate * dw1

            # w["output"] -= (1 / m) * learning_rate * dw
            # b["output"] -= (1 / m) * learning_rate * db
            # w["hidden"] -= (1 / m) * learning_rate * dw1
            # b["hidden"] -= (1 / m) * learning_rate * db1

            # preds["hidden"] = predict(X, w["hidden"], b["hidden"])
            # preds["output"] = predict(preds["hidden"].T, w["output"], b["output"])

            # get 30 frames given any number of epochs
            count = epochs // 30
            remainder = epochs % count
            if i % count == remainder:
                capture()

        # show profile of the model
        for z in ls(eye[2], 3, 10):
            eye = (eye[0], eye[1], z)
            capture()

        torch.manual_seed(42)

    return frames
