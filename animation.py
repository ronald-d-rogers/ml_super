from adapters.plotly import animate as animate_plotly
from base import AnimationFrame


def animate(
    frames: list[AnimationFrame],
    framework: str = "plotly",
    **kwargs,
):
    if framework == "plotly":
        return animate_plotly(frames, **kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}")
