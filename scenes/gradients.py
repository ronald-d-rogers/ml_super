from typing import List
from base import Frame, NodeView
from scenes.base import Scene, SceneUpdate
from scenes.model.traces import gradients

import plotly.graph_objects as go


class GradientScene(Scene):
    names = ["Gradient"]
    scene_types = ["scene"]
    height = 768

    def create_scenes(self, view: NodeView, frame: Frame) -> List[SceneUpdate]:
        return []

    def update_scenes(self, view: NodeView, frame: Frame) -> List[SceneUpdate]:
        return [
            SceneUpdate(
                scene=go.layout.Scene(),
                traces=gradients(
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
                ),
                annotations=[],
            )
        ]
