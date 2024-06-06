import torch
from sklearn.datasets import make_blobs
import numpy as np

from base import Frame
from learning import predict
from utils import ease_in, ease_out, orbit
from numpy import linspace as ls


def get_animation(
    chapters=None,
    resolution=20,
):
    if not chapters:
        chapters = ["targets", "logistic", "weights", "bias", "fit", "inference"]

    frames = []

    X, targets = make_blobs(
        n_samples=10,
        centers=[(-3, 3), (3, -3)],
        shuffle=False,
        cluster_std=2,
        random_state=42,
    )

    X = torch.Tensor(X)
    targets = torch.Tensor(targets)

    m = X.size(0)
    w = torch.Tensor([0, 0])
    b = 0.5
    preds = predict(X, w, b, activity=0)

    epochs = 30
    learning_rate = 0.1

    final_w = torch.Tensor([0.4068973660, -0.8636139035])
    final_b = torch.Tensor([0.4366261363])

    eye = (1, 1, 1)
    weight_eyes = ((0, 1, 0), (1, 0, 0))
    bias_eye = (1, -1, 0)

    aspect_ratio = (1.8, 1.8, 0.6)
    final_eye = (2, 0.654, 0)
    final_aspect_ratio = tuple(x + 0.2 for x in aspect_ratio)

    inference = None
    activity = 0
    focused_preds = None
    focused_errors = None
    focused_feature = None
    focus_labels = None
    focus_costs = None
    focus_targets = False
    show_preds = True
    show_profile = False
    show_decision_boundaries = False

    initial_targets = targets.clone()
    initial_eye = eye
    initial_aspect_ratio = aspect_ratio

    def capture(count=1):
        frame = Frame(
            X=X,
            preds={"output": preds.unsqueeze(0)},
            targets=targets.unsqueeze(0),
            w={"output": w.clone().unsqueeze(0)},
            b={"output": torch.Tensor([[b]])},
            epochs=epochs,
            learning_rate=learning_rate,
            size={"input": 2, "output": 1},
            inference=inference.clone() if inference is not None else None,
            activity=activity,
            focused_preds={"output": [focused_preds.copy()]} if focused_preds is not None else None,
            focused_errors={"output": [focused_errors.copy()]} if focused_errors is not None else None,
            focused_feature=focused_feature,
            focus_labels=focus_labels,
            focus_costs=focus_costs,
            focus_targets=focus_targets,
            show_preds=show_preds,
            show_profile=show_profile,
            show_decision_boundaries=show_decision_boundaries,
            eye=eye,
            bias_eye=bias_eye,
            weight_eyes=weight_eyes,
            bias_zrange=(-5, 6),
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )

        frames.extend([frame] * count)

    # These points are where we want to be, these are where we are, and this is how far we have to get there, or the error.
    if "targets" in chapters:
        capture()

        focus_targets = True

        capture(10)

        focus_targets = False
        focused_preds = None

        capture(10)

        focus_targets = False

        capture(10)

        focused_preds = list(range(m))
        focused_errors = list(range(m))

        capture(20)

        focus_targets = False
        focused_preds = None

        capture(10)

    # Fitting a plane to the data works well linear data but when the targets just 1s and 0s, the error is too high,
    # so we tilt the plane up and squash with the logistic function.
    if "logistic" in chapters:
        for eye_x, eye_y, eye_z, ax, ay, az in zip(
            ls(eye[0], 1, 10),
            ls(eye[1], 0.682, 10),
            ls(eye[2], 0, 10),
            ls(aspect_ratio[0], final_aspect_ratio[0], 10),
            ls(aspect_ratio[1], final_aspect_ratio[1], 10),
            ls(aspect_ratio[2], final_aspect_ratio[2], 10),
        ):
            eye = (eye_x, eye_y, eye_z)
            aspect_ratio = (ax, ay, az)
            capture()

        linear_targets = predict(X, torch.Tensor([[0.05, -0.05]]), 0.5, 1)

        for i in ls(0, 1, 10):
            targets = initial_targets + ((linear_targets - initial_targets) * i)
            capture()

        for i, j, k in zip(ls(w[0][0], 0.05, 10), ls(w[0][1], -0.05, 10), ls(b[0][0], 0.5, 10)):
            w[0] = i
            w[1] = j
            b = k
            preds = predict(X, w, b, activity)
            capture()

        for i in ease_out(ls(1, 0, 10)):
            targets = initial_targets + ((linear_targets - initial_targets) * i)
            capture()

        capture(10)

        show_profile = True
        focused_preds = list(range(m))
        focused_errors = list(range(m))

        capture(20)

        focused_preds = None

        for s, i, j, k in zip(
            ease_in(ls(0, 1, 10)),
            ls(w[0], 0.5, 10),
            ls(w[1], -0.5, 10),
            ls(b[0], 0.5, 10),
        ):
            activity = s
            w[0] = i
            w[1] = j
            b[0] = k
            preds = predict(X, w, b, activity)
            capture()

        focused_preds = list(range(m))
        focused_errors = list(range(m))

        capture(15)

        focused_preds = None
        show_profile = False

        capture(10)

        for eye_x, eye_y, eye_z, ax, ay, az, i, j, k in zip(
            ls(eye[0], initial_eye[0], 5),
            ls(eye[1], initial_eye[1], 5),
            ls(eye[2], initial_eye[2], 5),
            ls(aspect_ratio[0], initial_aspect_ratio[0], 5),
            ls(aspect_ratio[1], initial_aspect_ratio[1], 5),
            ls(aspect_ratio[2], initial_aspect_ratio[2], 5),
            ls(w[0], 0, 5),
            ls(w[1], 0, 5),
            ls(b[0], 0.5, 5),
        ):
            eye = (eye_x, eye_y, eye_z)
            aspect_ratio = (ax, ay, az)
            w[0] = i
            w[1] = j
            b[0] = k
            preds = predict(X, w, b, activity)
            capture()

        activity = 0

        capture(5)

    # When we amplify just weight 2 the plane tilts up just in feature 2 so that it can be squashed to fit its binary targets.
    # When both weights have a value the plane cuts deeper into the feature with the highest weight.
    # The more the 0s and 1s are separated in a single feature the more the plane will be fit to it minimizing error.
    if "weights" in chapters:
        for i in ls(w[1], -0.25, 10):
            w[0][1] = i
            preds = predict(X, w, b, activity)
            capture()

        capture(5)

        for i in ls(w[1], -2, 10):
            w[0][1] = i
            preds = predict(X, w, b, activity)
            capture()

        capture(10)

        for i in ease_in(ls(0, 1, 10)):
            activity = i
            preds = predict(X, w, b, activity)
            capture()

        capture(30)

        for i, j, k in zip(ls(w[0], 1.5, 15), ls(w[1], -0.5, 15), list(ease_in(ls(1, 0, 15)))):
            w[0] = i
            w[1] = j
            activity = k
            preds = predict(X, w, b, activity)
            capture()

        for i, j in zip(ls(w[0], 0.5, 15), ls(w[1], -1.5, 15)):
            w[0] = i
            w[1] = j
            preds = predict(X, w, b, activity)
            capture()

        for i in ease_in(ls(0, 1, 5)):
            activity = i
            preds = predict(X, w, b, activity)
            capture()

        for i, j in zip(ls(w[0], 1.5, 15), ls(w[1], -0.5, 15)):
            w[0] = i
            w[1] = j
            preds = predict(X, w, b, activity)
            capture()

        for i, j in zip(ls(w[0], 0.5, 15), ls(w[0][1], -1.5, 15)):
            w[0] = i
            w[1] = j
            preds = predict(X, w, b, activity)
            capture()

        for eye_x, eye_y, eye_z, ax, ay, az in zip(
            ls(eye[0], 1, 10),
            ls(eye[1], 0, 10),
            ls(eye[2], 1, 10),
            ls(aspect_ratio[0], aspect_ratio[0] + 0.2, 10),
            ls(aspect_ratio[1], aspect_ratio[1] + 0.2, 10),
            ls(aspect_ratio[2], aspect_ratio[2] + 0.2, 10),
        ):
            eye = (eye_x, eye_y, eye_z)
            capture()

        focused_preds = list(range(m))
        focused_errors = list(range(m))
        focus_costs = True
        show_decision_boundaries = True

        capture(30)

        focused_preds = None
        focus_costs = False
        show_decision_boundaries = False

        for eye_x, eye_y, eye_z, ax, ay, az in zip(
            ls(eye[0], initial_eye[0], 10),
            ls(eye[1], initial_eye[1], 10),
            ls(eye[2], initial_eye[2], 10),
            ls(aspect_ratio[0], initial_aspect_ratio[0], 10),
            ls(aspect_ratio[1], initial_aspect_ratio[1], 10),
            ls(aspect_ratio[2], initial_aspect_ratio[2], 10),
        ):
            eye = (eye_x, eye_y, eye_z)
            capture()

        for i, j, k in zip(ls(w[0], 0, 10), ls(w[1], 0, 10), ls(b, 0.5, 10)):
            w[0] = i
            w[1] = j
            b[0] = k
            preds = predict(X, w, b, activity)
            capture()

        activity = 0

    # Amplifying the bias when there is no tilt shifts the plane up and down. when the plane has a tilt the bias shifts the plane in the direction of the tilt.
    # So the weights angle and tilt, and the bias centers it all.
    if "bias" in chapters:
        capture(15)

        for i in ls(b, 0.9, 10):
            b = i
            preds = predict(X, w, b, activity)
            capture()

        for i in ls(b, 0.1, 10):
            b = i
            preds = predict(X, w, b, activity)
            capture()

        for i, j in zip(ls(w[0], 0.5, 10), ls(w[1], -0.5, 10)):
            w[0] = i
            w[1] = j
            preds = predict(X, w, b, activity)
            capture()

        capture(5)

        for i in ls(b, 5, 10):
            b = i
            preds = predict(X, w, b, activity)
            capture()

        for i in ls(b, -5, 10):
            b = i
            preds = predict(X, w, b, activity)
            capture()

        capture(10)

        activity = 1

        b = 0.5

        # so the weights angle
        for i, j in zip(ls(w[0], 1, 7), ls(w[1], -2, 7)):
            w[0] = i
            w[1] = j
            preds = predict(X, w, b, activity)
            capture()

        for i, j in zip(ls(w[0], 2, 7), ls(w[1], -1, 7)):
            w[0] = i
            w[1] = j
            preds = predict(X, w, b, activity)
            capture()

        capture(5)

        w = torch.Tensor([2, -2])

        for i in ls(b, 5, 7):
            b = i
            preds = predict(X, w, b, activity)
            capture()

        for i in ls(b, -5, 7):
            b = i
            preds = predict(X, w, b, activity)
            capture()

        capture(5)

        for i, j, k in zip(ls(w[0], 0, 7), ls(w[1], 0, 7), ls(b, 0.5, 7)):
            w[0] = i
            w[1] = j
            b = k
            preds = predict(X, w, b, activity)
            capture()

        activity = 0

        capture(5)

    # To fit the model we take the error and multiply x and y values of each point to get the loss then sum up by feature, and tilt the plane up by
    # these new amounts, averaged and reduced by the learning rate. We rerun this process until we've minimized the loss and the model has "converged".
    if "fit" in chapters:
        capture(10)

        focused_preds = list(range(m))
        focused_errors = list(range(m))

        capture(20)

        focused_feature = 0

        capture(4)

        focused_feature = 1

        capture(7)

        focused_feature = None
        focused_preds = None

        capture()

        for eye_x, eye_y, eye_z, ax, ay, az in zip(
            ls(eye[0], final_eye[0], 10),
            ls(eye[1], final_eye[1], 10),
            ls(eye[2], final_eye[2], 10),
            ls(aspect_ratio[0], final_aspect_ratio[0], 10),
            ls(aspect_ratio[1], final_aspect_ratio[1], 10),
            ls(aspect_ratio[2], final_aspect_ratio[2], 10),
        ):
            eye = (eye_x, eye_y, eye_z)
            aspect_ratio = (ax, ay, az)
            capture()

        show_profile = True
        show_epochs = 3

        for i in range(show_epochs):
            focus_costs = False

            focused_preds = []
            for i in range(m):
                focused_preds.append(i)
                focused_errors = [i]
                capture()

            for i in ease_out(ls(1, 0, 3)):
                activity = i
                capture()

            focused_preds = None
            focused_errors = None
            focus_costs = True

            capture(13)

            focus_costs = False
            focus_labels = True
            new_w = w - learning_rate * ((1 / m) * ((preds - targets) @ X))
            new_b = b - learning_rate * ((1 / m) * torch.sum(preds - targets))
            for i, j, k in zip(ls(w[0], new_w[0], 3), ls(w[1], new_w[1], 3), ls(b, new_b, 3)):
                w = torch.Tensor([i, j])
                b = k
                capture()

            capture(5)

            for i in ease_in(ls(0, 1, 3)):
                activity = i
                capture()

            preds = predict(X, w, b, activity)

            focus_labels = False

            capture(5)

        for _ in range(epochs - show_epochs):
            w -= learning_rate * ((1 / m) * ((preds - targets) @ X))
            b -= learning_rate * ((1 / m) * torch.sum(preds - targets))
            preds = predict(X, w, b, activity)
            capture()

    # Now let's make a prediction. the points on the upper half are classified as 1 and the below half, 0.
    # And the dotted line that separates the halves is the decision boundary.
    if "inference" in chapters:
        w = final_w.clone()
        b = final_b.clone()
        activity = 1
        preds = predict(X, w, b, activity)

        eye = final_eye
        aspect_ratio = final_aspect_ratio

        show_profile = True
        show_decision_boundaries = True

        capture(5)

        inference = torch.Tensor([1, 1])
        center = torch.Tensor([-b, b])
        for _ in range(30):
            inference = orbit(inference, center, np.pi / 15)
            capture()

        show_profile = False

        for eye_x, eye_y, eye_z in zip(
            ls(final_eye[0], initial_eye[0], 10),
            ls(final_eye[1], initial_eye[1], 10),
            ls(0, initial_eye[2], 10),
        ):
            eye = (eye_x, eye_y, eye_z)
            inference = orbit(inference, center, np.pi / 15)
            capture()

        for _ in range(60):
            inference = orbit(inference, center, np.pi / 15)
            capture()

    return frames
