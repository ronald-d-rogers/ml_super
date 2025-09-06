# def make_frame(frame: AnimationFrame, animation: Animation, name: str):
#     if frame.eye:
#         eye = dict(x=frame.eye[0], y=frame.eye[1], z=frame.eye[2])
#     else:
#         eye = None

#     view = animation.node_view(frame)
#     input_module = view.modules[-2]
#     input_size = frame.size[input_module]
#     output_colors, feature_colors = get_colors(frame, view, animation)
#     weight_eyes = view.get_weight_eyes(as_dict=True)
#     bias_eye = frame.get_bias_eye(as_dict=True)
#     show_bg = animation.show_bg

#     value = go.Frame(
#         name=name,
#         data=[
#             *model_surface(frame, animation),
#             *weights_traces(frame, animation),
#             # *loss_traces(view, frame, animation),
#             *nn_traces(frame, animation),
#             # *gradient_traces(frame, animation),
#         ],
#         layout=dict(
#             annotations=[
#                 *nn_annotations(frame, animation, show=animation.show_network),
#             ],
#             scene=dict(
#                 camera=dict(eye=eye),
#                 aspectratio=dict(
#                     x=frame.aspect_ratio[0],
#                     y=frame.aspect_ratio[1],
#                     z=frame.aspect_ratio[2],
#                 ),
#                 zaxis_title="",
#                 xaxis=dict(
#                     # color=animation.theme.feature_text_colors[0],
#                     backgroundcolor=output_colors[0] if show_bg else TRANSPARENT,
#                     range=frame.get_range(dim=0, pad=True),
#                 ),
#                 yaxis=dict(
#                     # color=animation.theme.feature_text_colors[1],
#                     backgroundcolor=output_colors[-1] if show_bg else TRANSPARENT,
#                     range=frame.get_range(dim=1, pad=True),
#                 ),
#                 zaxis=dict(range=frame.get_zrange(pad=True)),
#                 annotations=[],
#             ),
#         ),
#     )

#     if animation.show_parameters:
#         for i in animation.show_parameters:
#             if isinstance(i, int):
#                 value.layout[f"scene{i + 1}"] = dict(
#                     camera=dict(eye=weight_eyes[i - 1]),
#                     xaxis=dict(backgroundcolor=feature_colors[i - 1]),
#                     yaxis=dict(backgroundcolor=feature_colors[i - 1], range=frame.get_range(dim=1, pad=True)),
#                 )

#             if i == "b":
#                 value.layout[f"scene{input_size + 2}"] = dict(
#                     camera=dict(eye=bias_eye),
#                     xaxis=dict(backgroundcolor=feature_colors[-1]),
#                     yaxis=dict(backgroundcolor=feature_colors[0]),
#                     zaxis=dict(range=frame.get_bias_zrange(pad=True)),
#                 )

#     return value
