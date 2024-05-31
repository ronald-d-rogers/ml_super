from utils import hex_to_rgb, lerp_rgb, rgb_to_str, str_to_rgb, interp_rgb


def test_str_to_rgb():
    # Test case 1: str_to_rgb with input "rgb(0, 0, 0)"
    result = str_to_rgb("rgb(0, 0, 0)")
    assert result == (0, 0, 0)

    # Test case 2: str_to_rgb with input "rgb(255, 255, 255)"
    result = str_to_rgb("rgb(255, 255, 255)")
    assert result == (255, 255, 255)

    # Test case 3: str_to_rgb with input "rgb(100, 50, 75)"
    result = str_to_rgb("rgb(100, 50, 75)")
    assert result == (100, 50, 75)

    # Test case 4: str_to_rgb with input "rgb(0, 255, 0)"
    result = str_to_rgb("rgb(0, 255, 0)")
    assert result == (0, 255, 0)

    # Test case 5: str_to_rgb with input "rgb(128, 128, 128)"
    result = str_to_rgb("rgb(128, 128, 128)")
    assert result == (128, 128, 128)


def test_hex_to_rgb():
    # Test case 1: hex_to_rgb with input "#000000"
    result = hex_to_rgb("#000000")
    assert result == (0, 0, 0)

    # Test case 2: hex_to_rgb with input "#FFFFFF"
    result = hex_to_rgb("#FFFFFF")
    assert result == (255, 255, 255)

    # Test case 3: hex_to_rgb with input "#FF0000"
    result = hex_to_rgb("#FF0000")
    assert result == (255, 0, 0)

    # Test case 4: hex_to_rgb with input "#00FF00"
    result = hex_to_rgb("#00FF00")
    assert result == (0, 255, 0)

    # Test case 5: hex_to_rgb with input "#0000FF"
    result = hex_to_rgb("#0000FF")
    assert result == (0, 0, 255)


def test_rgb_to_str():
    # Test case 1: rgb_to_str with input (0, 0, 0)
    result = rgb_to_str((0, 0, 0))
    assert result == "rgb(0, 0, 0)"

    # Test case 2: rgb_to_str with input (255, 255, 255)
    result = rgb_to_str((255, 255, 255))
    assert result == "rgb(255, 255, 255)"

    # Test case 3: rgb_to_str with input (100, 50, 75)
    result = rgb_to_str((100, 50, 75))
    assert result == "rgb(100, 50, 75)"

    # Test case 4: rgb_to_str with input (0, 255, 0)
    result = rgb_to_str((0, 255, 0))
    assert result == "rgb(0, 255, 0)"

    # Test case 5: rgb_to_str with input (128, 128, 128)
    result = rgb_to_str((128, 128, 128))
    assert result == "rgb(128, 128, 128)"


def test_interp_rgb():
    # Test case 1: interpolate between (0, 0, 0) and (255, 255, 255) with n = 3
    result = interp_rgb((0, 0, 0), (255, 255, 255), 3)
    assert result == [(0, 0, 0), (127.5, 127.5, 127.5), (255, 255, 255)]

    # Test case 2: interpolate between (0, 0, 0) and (255, 255, 255) with n = 1
    result = interp_rgb((0, 0, 0), (255, 255, 255), 1)
    assert result == [(127.5, 127.5, 127.5)]

    # Test case 3: interpolate between (255, 0, 0) and (0, 255, 0) with n = 2
    result = interp_rgb((255, 0, 0), (0, 255, 0), 3)
    assert result == [(255, 0, 0), (127.5, 127.5, 0), (0, 255, 0)]

    # Test case 4: interpolate between (0, 0, 0) and (0, 0, 0) with n = 5
    result = interp_rgb((0, 0, 0), (0, 0, 0), 5)
    assert result == [(0, 0, 0)] * 5


def test_lerp_rgb():
    # Test case 1: lerp between (0, 0, 0) and (255, 255, 255) with frac = 0.5
    result = lerp_rgb((0, 0, 0), (255, 255, 255), 0.5)
    assert result == (127.5, 127.5, 127.5)

    # Test case 2: lerp between (100, 50, 75) and (200, 150, 175) with frac = 0.25
    result = lerp_rgb((100, 50, 75), (200, 150, 175), 0.25)
    assert result == (125.0, 75.0, 100.0)

    # Test case 3: lerp between (0, 0, 0) and (255, 255, 255) with frac = 1.0
    result = lerp_rgb((0, 0, 0), (255, 255, 255), 1.0)
    assert result == (255, 255, 255)

    # Test case 4: lerp between (255, 0, 0) and (0, 255, 0) with frac = 0.75
    result = lerp_rgb((255, 0, 0), (0, 255, 0), 0.75)
    assert result == (63.75, 191.25, 0.0)

    # Test case 5: lerp between (0, 0, 0) and (0, 0, 0) with frac = 0.0
    result = lerp_rgb((0, 0, 0), (0, 0, 0), 0.0)
    assert result == (0, 0, 0)
