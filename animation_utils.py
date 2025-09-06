from base import Animation, AnimationFrame, ParameterView


def get_colors(frame: AnimationFrame, view: ParameterView, animation: Animation):
    output = view.modules[-1]
    hidden_or_input = view.modules[-2]
    feature_colors = animation.focusable_colors(frame.focused_feature, frame.size[hidden_or_input])

    # if it has a hidden layer
    if len(view.modules) > 2:
        color = animation.colors(frame.size[output])[animation.param_index]
        output_colors = [color]
    else:
        output_colors = feature_colors

    return output_colors, feature_colors
