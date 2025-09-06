from base import AnimationFrame, LayoutComponent, ParameterView, Plot2DFrame
from components.nn.annotations import nn_annotations
from components.nn.traces import nn_traces


class NeuralNetworkComponent(LayoutComponent):
    names = ["neural_network"]
    types = ["scatter"]
    height = 768

    def create_component(self, frame: AnimationFrame, view: ParameterView) -> list[Plot2DFrame]:
        return []

    def update_component(self, frame: AnimationFrame, view: ParameterView) -> list[Plot2DFrame]:
        return [
            Plot2DFrame(
                data=nn_traces(frame, self.animation),
                annotations=nn_annotations(frame, self.animation, show=self.animation.show_network),
            )
        ]
