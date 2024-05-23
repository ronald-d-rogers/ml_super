from typing import NamedTuple


class Theme(NamedTuple):
    class_colors: dict
    feature_colors: dict
    focused_feature_colors: dict
    target_color: str


themes = dict(
    super=Theme(
        class_colors={0: "#FF474C", 1: "#ADD8E6"},
        feature_colors={0: "#FFDEB4", 1: "#B2A3FF"},
        focused_feature_colors={0: "#FFDA6B", 1: "#7860FF"},
        target_color="black",
    )
)


default_theme = themes["super"]
