from dataclasses import dataclass
from typing import List


@dataclass
class Theme:
    text_color: str = None
    font_family: str = "Comic Sans MS, Droid Sans, sans-serif"
    font_size: int = 24
    background_color: str = None
    class_colors: List[str] = None
    feature_colors: List[str] = None
    feature_grid_colors: List[str] = None
    feature_text_colors: List[str] = None
    focused_feature_colors: List[str] = None
    feature_note_border_color: str = None
    target_color: str = None
    target_grid_color: str = None
    target_text_color: str = None
    marker_text_color: str = None
    target_marker_text_color: str = None
    focused_target_marker_text_color: str = None
    note_text_color: str = None
    note_background_color: str = None
    note_border_color: str = None
    marker_size: int = 30
    show_label_names = True
    label_precision = 3
    label_yshift = 0
    cost_label_yshift = 0
    cost_label_xshift = 0
    cost_label_xanchor = "left"
    label_font_size = 40
    parameters_line_width = 6


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
    dark=Theme(
        text_color="#DDDDDD",
        background_color="rgba(0,0,0,0)",
        class_colors=["#FF474C", "#ADD8E6"],
        feature_grid_colors=["#999999", "#999999"],
        feature_colors=["#333333", "#666666"],
        feature_text_colors=["#555555", "#888888"],
        focused_feature_colors=["#444444", "#777777"],
        feature_note_border_color="rgba(0,0,0,.8)",
        marker_text_color="black",
        note_text_color="white",
        note_background_color="rgba(0,0,0,.8)",
        note_border_color="rgba(255,255,255,.8)",
        target_color="black",
        target_grid_color="#999999",
        target_text_color="#DDDDDD",
        target_marker_text_color="white",
        focused_target_marker_text_color="black",
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


def merge_themes(*themes: Theme) -> Theme:
    merged = Theme()
    for theme in themes:
        for field in theme.__dataclass_fields__:
            value = getattr(theme, field)
            if value is not None:
                setattr(merged, field, value)
    return merged
