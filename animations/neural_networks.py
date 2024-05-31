import torch

from base import Frame
from learning import predict
from numpy import linspace as ls


def get_animation(
    chapters=None,
    resolution=20,
    verbose=False,
):
    if chapters is None:
        chapters = ["logistic", "xor", "neural", "weights", "fit"]

    torch.manual_seed(42)

    final_w = torch.Tensor([[-14.2149982452, 12.6667270660]])
    final_b = torch.Tensor([[-5.3260169029]])
    final_hidden_w = torch.Tensor([[4.4657111168, 4.4657068253], [10.1644496918, 10.1644334793]])
    final_hidden_b = torch.Tensor([[-6.7141456604], [-6.7141456604]])

    hidden_w = None
    hidden_b = None

    intial_eye = (1.2, -0.8, 1)
    initial_weight_eyes = ((0, 1, 0), (1, 0, 0))
    initial_hidden_weight_eyes = ((1, -1, 0), (1, -1, 0))

    eye = intial_eye
    bias_eye = (1, -1, 0)

    frames = []

    planarity = 0
    epochs = 30
    learning_rate = 1

    input_size = 2
    hidden_size = 2
    output_size = 1

    aspect_ratio = (1.8, 1.8, 0.6)
    # final_eye = (2, 0.64, 0)
    # final_aspect_ratio = tuple(x + 0.2 for x in aspect_ratio)

    inference = None
    planarity = 0
    focused_inputs = None
    focused_feature = None
    focus_labels = None
    focus_total_loss = None
    focus_targets = False
    show_preds = True
    surface_color = None
    show_decision_boundaries = False

    def capture(count=1):
        frame = Frame(
            X=X,
            preds=preds,
            targets=targets,
            hidden_w=hidden_w.clone() if hidden_w is not None else None,
            hidden_b=hidden_b.clone() if hidden_b is not None else None,
            w=w.clone(),
            b=b.clone(),
            epochs=epochs,
            domain_padding=0.4,
            range_padding=0.5,
            learning_rate=learning_rate,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            inference=inference.clone() if inference is not None else None,
            planarity=planarity,
            eye=eye,
            weight_eyes=weight_eyes,
            bias_eye=bias_eye,
            focused_inputs=focused_inputs,
            focused_feature=focused_feature,
            focus_labels=focus_labels,
            focus_total_loss=focus_total_loss,
            focus_targets=focus_targets,
            show_preds=show_preds,
            surface_color=surface_color,
            show_decision_boundaries=show_decision_boundaries,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )

        frames.extend([frame] * count)

    if "logistic" in chapters:
        X = torch.Tensor([[1, 0], [0, 1]])
        targets = torch.Tensor([[0, 1]])
        w = torch.Tensor([[0, 0]])
        b = torch.Tensor([[0.5]])
        preds = predict(X, w, b)

        weight_eyes = initial_weight_eyes

        capture()

        m = X.shape[0]
        for _ in range(epochs):
            w -= learning_rate * ((1 / m) * ((preds - targets) @ X))
            b -= learning_rate * ((1 / m) * torch.sum(preds - targets))
            preds = predict(X, w, b)
            capture()

        capture(10)

    if "xor" in chapters:
        X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        targets = torch.Tensor([[0, 1, 1, 0]])
        w = torch.Tensor([[0, 0]])
        b = torch.Tensor([[0.5]])
        preds = predict(X, w, b)

        weight_eyes = initial_weight_eyes

        capture()

        m = X.shape[0]
        for _ in range(epochs):
            w -= learning_rate * ((1 / m) * ((preds - targets) @ X))
            b -= learning_rate * ((1 / m) * torch.sum(preds - targets))
            preds = predict(X, w, b)
            capture()

        capture(10)

    if "neural" in chapters:
        X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        targets = torch.Tensor([[0, 1, 1, 0]])

        m = X.shape[0]

        input_size = 2
        hidden_size = 2
        output_size = 1

        hidden_w = torch.randn(hidden_size, input_size)
        hidden_b = torch.zeros((hidden_size, 1))
        w = torch.randn(output_size, hidden_size)
        b = torch.zeros((output_size, 1))

        hidden_preds = predict(X, hidden_w, hidden_b)
        preds = predict(hidden_preds.T, w, b)

        weight_eyes = initial_hidden_weight_eyes

        capture()

        learning_rate = 2
        epochs = 2000
        for i in range(epochs):
            error = preds - targets
            # print(error, w.T, preds1.T, (1 - preds1).T, sep="\n", end="\n\n")
            dw = (1 / m) * (error @ hidden_preds.T)
            db = (1 / m) * torch.sum(error)
            # print((w.T @ error).T, (preds1 * (1 - preds1)).T, sep="\n", end="\n\n\n")
            hidden_error = (w.T @ error) * (hidden_preds * (1 - hidden_preds))
            dw1 = (1 / m) * (hidden_error @ X)
            db1 = (1 / m) * torch.sum(hidden_error)

            w -= learning_rate * dw
            b -= learning_rate * db
            hidden_w -= learning_rate * dw1
            hidden_b -= learning_rate * db1

            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)

            # get 30 frames given any number of epochs
            count = epochs / 30
            if i % (epochs // count) == 0:
                # print(b)
                capture()

    if "weights" in chapters:
        X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        targets = torch.Tensor([[0, 1, 1, 0]])
        hidden_w = final_hidden_w.clone()
        hidden_b = final_hidden_b.clone()
        w = final_w.clone()
        b = final_b.clone()

        hidden_preds = predict(X, hidden_w, hidden_b)
        preds = predict(hidden_preds.T, w, b)

        weight_eyes = initial_hidden_weight_eyes

        capture()

        eye = None

        for i, j, k in zip(ls(final_w[0][0], 0, 10), ls(final_w[0][1], 0, 10), ls(final_b[0][0], 0, 10)):
            w[0][0] = i
            w[0][1] = j
            b[0][0] = k
            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)
            capture()

        for i in ls(0, 10, 10):
            w[0][0] = i
            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)
            capture()

        for i in ls(10, -10, 20):
            w[0][0] = i
            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)
            capture()

        capture(10)

        for i in ls(-10, 0, 10):
            w[0][0] = i
            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)
            capture()

        for i in ls(0, 10, 10):
            w[0][1] = i
            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)
            capture()

        for i in ls(10, -10, 20):
            w[0][1] = i
            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)
            capture()

        capture(10)

        # everything back to 0 from their current values
        for i, j, k in zip(ls(w[0][0], 0, 10), ls(w[0][1], 0, 10), ls(b[0][0], 0, 10)):
            w[0][0] = i
            w[0][1] = j
            b[0][0] = k
            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)
            capture()

        # lift down weight 0 to -10
        for i in ls(0, -10, 10):
            w[0][0] = i
            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)
            capture()

        capture(5)

        # lift up weight 1 to 10
        for i in ls(0, 10, 10):
            w[0][1] = i
            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)
            capture()

        capture(5)

        for i in ls(0, -5, 10):
            b[0][0] = i
            hidden_preds = predict(X, hidden_w, hidden_b)
            preds = predict(hidden_preds.T, w, b)
            capture()

        capture(10)

    if "fit" in chapters:
        pass

    return frames
