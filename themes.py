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
        class_colors={0: "#FF474C", 1: "#ADD8E6"},
        feature_colors={0: "#FFDEB4", 1: "#B2A3FF"},
        focused_feature_colors={0: "#FFDA6B", 1: "#7860FF"},
        feature_text_colors={0: "white", 1: "white"},
        target_color="black",
        target_text_color="white",
    ),
    pastel=Theme(
        class_colors={0: "#FF474C", 1: "#ADD8E6"},
        feature_colors={0: "#DADFE5", 1: "#F2EEE8"},
        focused_feature_colors={0: "#9FC2EA", 1: "#E2D3BC"},
        feature_text_colors={0: "white", 1: "white"},
        target_color="#F0F0F0",
        target_text_color="black",
    ),
)


default_theme: Theme = themes["super"]
