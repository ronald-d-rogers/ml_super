import random
import torch

from base import AnimationFrame
from learning import bce_loss, predict, sigmoid
from numpy import linspace as ls
import numpy as np

from utils import clone, orbit

from sklearn.datasets import make_blobs

all_chapters = ["1", "2", "3"]


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
            loss_fn=bce_loss,
            loss=loss.clone(),
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

    inference = None

    activations = {
        "hidden": sigmoid,
        "output": sigmoid,
    }

    eye = None
    weight_eyes = None
    neuron_weight_eyes = ((0, 1, 0), (1, 0, 0))

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

    if "1" in chapters:
        w = {"output": torch.Tensor([[0, 0]])}
        b = {"output": torch.Tensor([[0.5]])}

        modules = ["input", "output"]

        # create a randomly distributed of points with a z of either 0 and 1
        X = torch.rand(20, 2)
        targets = torch.randint(0, 2, (1, 20)).bool().float()

        preds = {"output": predict(X, w["output"], b["output"])}

        m = X.size(0)

        focused_errors["output"] = [[] for _ in range(size["output"])]

        focus_targets = True

        capture()

        focus_targets = False

        capture()

        learning_rate = 0.5

        for _ in range(epochs):
            loss = bce_loss(preds["output"], targets)
            w["output"] -= learning_rate * ((1 / m) * ((preds["output"] - targets) @ X))
            b["output"] -= learning_rate * ((1 / m) * torch.sum(preds["output"] - targets))
            preds["output"] = predict(X, w["output"], b["output"])
            capture()

        # Now let's make a prediction. the points on the upper half are classified as 1 and the below half, 0.
        # And the dotted line that separates the halves is the decision boundary.

        focus_targets = True

        capture()

    if "2" in chapters:
        w = {"output": torch.Tensor([[0, 0]])}
        b = {"output": torch.Tensor([[0.5]])}

        modules = ["input", "output"]

        X, targets = make_blobs(
            n_samples=20,
            centers=[(-1, 1), (1, -1)],
            shuffle=False,
            cluster_std=2,
            random_state=42,
        )

        X = torch.from_numpy(X).float()
        targets = torch.from_numpy(targets).float().unsqueeze(0)
        preds = {"output": predict(X, w["output"], b["output"])}
        loss = bce_loss(preds["output"], targets)

        m = X.size(0)

        focused_errors["output"] = [[] for _ in range(size["output"])]

        focus_targets = True

        capture()

        focus_targets = False

        capture()

        learning_rate = 0.5

        for _ in range(epochs):
            w["output"] -= learning_rate * ((1 / m) * ((preds["output"] - targets) @ X))
            b["output"] -= learning_rate * ((1 / m) * torch.sum(preds["output"] - targets))
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        # Now let's make a prediction. the points on the upper half are classified as 1 and the below half, 0.
        # And the dotted line that separates the halves is the decision boundary.

        capture()

        inference = torch.Tensor([-0.7, 0.7])
        center = torch.Tensor([-0.3, -0.1])

        for _ in range(45):
            inference = orbit(inference, center, np.pi / 15)
            capture()

        focus_targets = True

        capture()

        for _ in range(45):
            inference = orbit(inference, center, np.pi / 15)
            capture()

    # wiggle weight one up and down
    if "3" in chapters:
        X, targets = make_blobs(
            n_samples=10,
            centers=[(-3, 3), (3, -3)],
            shuffle=False,
            cluster_std=2,
            random_state=42,
        )

        X = torch.from_numpy(X).float()
        targets = torch.from_numpy(targets).float().unsqueeze(0)

        m = X.size(0)

        modules = ["input", "output"]
        size = {"input": 2, "output": 1}

        w = {"output": torch.Tensor([[0.0, 0.0]])}
        b = {"output": torch.Tensor([[0.5]])}
        preds["output"] = predict(X, w["output"], b["output"])
        loss = bce_loss(preds["output"], targets)

        focus_targets = True

        capture()

        focus_targets = False

        for i in ls(w["output"][0][0], 0, 10):
            w["output"][0][0] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        for i in ls(w["output"][0][0], 1, 10):
            w["output"][0][0] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        for i in ls(w["output"][0][0], 0, 10):
            w["output"][0][0] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        for i in ls(w["output"][0][0], -1, 10):
            w["output"][0][0] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        for i in ls(w["output"][0][0], 0, 10):
            w["output"][0][0] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        # wiggle weight two up and down
        for i in ls(w["output"][0][1], 0, 10):
            w["output"][0][1] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        for i in ls(w["output"][0][1], 1, 10):
            w["output"][0][1] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        for i in ls(w["output"][0][1], 0, 10):
            w["output"][0][1] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        for i in ls(w["output"][0][1], -1, 10):
            w["output"][0][1] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        for i in ls(w["output"][0][1], 0, 10):
            w["output"][0][1] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        # first set the weights to bad values

        for i, j in zip(ls(w["output"][0][0], 0, 10), ls(w["output"][0][1], -1, 10)):
            w["output"][0][0] = i
            w["output"][0][1] = j
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        # wiggle bias up and down
        for i in ls(b["output"][0][0], -5, 10):
            b["output"][0][0] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        for i in ls(b["output"][0][0], 5, 10):
            b["output"][0][0] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        for i in ls(b["output"][0][0], 0.5, 10):
            b["output"][0][0] = i
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        # set the weights to decent values
        for i, j in zip(ls(w["output"][0][0], 1, 10), ls(w["output"][0][1], -1, 10)):
            w["output"][0][0] = i
            w["output"][0][1] = j
            preds["output"] = predict(X, w["output"], b["output"])
            loss = bce_loss(preds["output"], targets)
            capture()

        preds["output"] = predict(X, w["output"], b["output"])

        weight_eyes = neuron_weight_eyes

        focused_errors["output"] = [list(range(m)) for _ in range(size["output"])]

        capture()

    return frames
