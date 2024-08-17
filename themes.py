from typing import NamedTuple, List


class Theme(NamedTuple):
    text_color: str
    background_color: str
    class_colors: List[str]
    feature_colors: List[str]
    feature_grid_colors: List[str]
    feature_text_colors: List[str]
    focused_feature_colors: List[str]
    feature_note_border_color: str
    target_color: str
    target_grid_color: str
    target_text_color: str
    marker_text_color: str
    target_marker_text_color: str
    focused_target_marker_text_color: str
    note_text_color: str
    note_background_color: str
    note_border_color: str


themes = dict(
    super=Theme(
        text_color="#2B3F60",
        background_color="white",
        class_colors=["#FF474C", "#ADD8E6"],
        feature_colors=["#FFDEB4", "#B2A3FF"],
        feature_grid_colors=[None, None],
        focused_feature_colors=["#FFDA6B", "#7860FF"],
        feature_text_colors=["#2B3F60", "#2B3F60"],
        feature_note_border_color="rgba(0,0,0,.8)",
        target_color="black",
        target_grid_color=None,
        target_text_color="#2B3F60",
        marker_text_color="#2B3F60",
        focused_target_marker_text_color="#000000",
        target_marker_text_color="#FFFFFF",
        note_text_color="#2B3F60",
        note_background_color="rgba(255,255,255,.8)",
        note_border_color="rgba(0,0,0,.8)",
    ),
    pastel=Theme(
        text_color="#2B3F60",
        background_color="white",
        class_colors=["#FF474C", "#ADD8E6"],
        feature_colors=["#DADFE5", "#F2EEE8"],
        feature_grid_colors=[None, None],
        focused_feature_colors=["#9FC2EA", "#E2D3BC"],
        feature_text_colors=["#2B3F60", "#2B3F60"],
        feature_note_border_color="rgba(0,0,0,.8)",
        target_color="#F0F0F0",
        target_grid_color=None,
        target_text_color="#2B3F60",
        marker_text_color="#2B3F60",
        focused_target_marker_text_color="#000000",
        target_marker_text_color="#FFFFFF",
        note_text_color="#2B3F60",
        note_background_color="rgba(255,255,255,.8)",
        note_border_color="rgba(0,0,0,.8)",
    ),
)


default_theme: Theme = themes["super"]
