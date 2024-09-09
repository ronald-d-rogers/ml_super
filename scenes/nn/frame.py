from typing import List

from base import AnimationFrame, NodeView
from scenes.base import Scene, Frame
from scenes.nn.annotations import nn_annotations
from scenes.nn.traces import nn_traces


class NeuralNetworkScene(Scene):
    name = ["neural_network"]
    scene_types = ["scatter"]
    height = 768

    def create_scenes(self, view: NodeView, frame: AnimationFrame) -> List[Frame]:
        return []

    def update_scenes(self, view: NodeView, frame: AnimationFrame) -> List[Frame]:
        return [
            Frame(
                scene=None,
                traces=nn_traces(frame, self.animation),
                annotations=nn_annotations(frame, self.animation, show=self.animation.show_network),
            )
        ]
