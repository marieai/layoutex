import pytest

from layoutex.layout_provider import LayoutProvider, get_layout_provider


def test_fixed_layout_provider():
    layout = get_layout_provider("fixed", 10, 100)
    assert isinstance(layout, LayoutProvider)

    layouts = layout.get_layouts(1)
    print("Total layouts")
    print(layouts)


def test_unknown_layout_provider():
    with pytest.raises(ValueError):
        get_layout_provider("unknown", 10, 100)
