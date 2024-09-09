from typing import List, NamedTuple

from base import Animation, AnimationFrame, NodeView

import plotly.graph_objs as go
from plotly.basedatatypes import BaseTraceType


class Frame(NamedTuple):
    scene: go.layout.Scene
    traces: List[BaseTraceType]
    annotations: List[go.layout.Annotation]


class Scene:
    names: List[str]
    scene_types: List[str]
    height: int

    @property
    def columns(self):
        return len(self.scene_types)

    def __init__(self, animation: Animation, scene_types: List[str]) -> go.Scene:
        self.animation = animation
        self.scene_types = scene_types

    def create_scenes(self, view: NodeView, frame: AnimationFrame) -> List[Frame]:
        raise NotImplementedError

    def update_scenes(self, view: NodeView, frame: AnimationFrame) -> List[Frame]:
        raise NotImplementedError
