from animations.logistic_regression import get_animation


def test_get_animation():
    # Test case 1: Test the default behavior of the function
    frames = get_animation()
    assert len(frames) > 0

    # Test case 2: Test with custom chapters and resolution
    chapters = ["targets", "logistic", "weights"]
    resolution = 10
    frames = get_animation(chapters=chapters, resolution=resolution)
    assert len(frames) > 0
