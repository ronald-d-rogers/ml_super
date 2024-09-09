from typing import List

from base import Frame, NodeView
from scenes.base import Scene, SceneUpdate
from scenes.nn.annotations import nn_annotations
from scenes.nn.traces import nn_traces


class NeuralNetworkScene(Scene):
    name = ["neural_network"]
    scene_types = ["scatter"]
    height = 768

    def create_scenes(self, view: NodeView, frame: Frame) -> List[SceneUpdate]:
        return []

    def update_scenes(self, view: NodeView, frame: Frame) -> List[SceneUpdate]:
        return [
            SceneUpdate(
                scene=None,
                traces=nn_traces(frame, self.animation),
                annotations=nn_annotations(frame, self.animation, show=self.animation.show_network),
            )
        ]
