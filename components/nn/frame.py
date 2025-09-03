from typing import List

from base import AnimationFrame, LayoutComponent, NodeView
from base import ViewFrameUpdate
from components.nn.annotations import nn_annotations
from components.nn.traces import nn_traces


class NeuralNetworkComponent(LayoutComponent):
    name = ["neural_network"]
    types = ["scatter"]
    height = 768

    def create_component(self, view: NodeView, frame: AnimationFrame) -> list[ViewFrameUpdate]:
        return []

    def update_component(self, view: NodeView, frame: AnimationFrame) -> list[ViewFrameUpdate]:
        return [
            ViewFrameUpdate(
                view=None,
                data=nn_traces(frame, self.animation),
                annotations=nn_annotations(frame, self.animation, show=self.animation.show_network),
            )
        ]
