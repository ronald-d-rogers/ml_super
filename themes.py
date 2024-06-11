from typing import NamedTuple


class Theme(NamedTuple):
    class_colors: dict
    feature_colors: dict
    feature_text_colors: dict
    focused_feature_colors: dict
    target_color: str
    target_text_color: str


themes = dict(
    super=Theme(
        class_colors=["#FF474C", "#ADD8E6"],
        feature_colors=["#FFDEB4", "#B2A3FF"],
        focused_feature_colors=["#FFDA6B", "#7860FF"],
        feature_text_colors=["white", "white"],
        target_color="black",
        target_text_color="white",
    ),
    pastel=Theme(
        class_colors=["#FF474C", "#ADD8E6"],
        feature_colors=["#DADFE5", "#F2EEE8"],
        focused_feature_colors=["#9FC2EA", "#E2D3BC"],
        feature_text_colors=["white", "white"],
        target_color="#F0F0F0",
        target_text_color="black",
    ),
)


default_theme: Theme = themes["super"]
