import pytest

from layoutex.content_provider import get_content_provider
from layoutex.layout_provider import LayoutProvider, get_layout_provider


def test_text_content_provider():
    component = {
        "content_type": "text",
        "bbox": [0, 0, 512, 512],
        "sizing": ["FULL_WIDTH", "FULL_HEIGHT"],
    }

    provider = get_content_provider(component["content_type"], assets_dir="./assets")
    assert provider

    content = provider.get_content(
        component, bbox_mode="absolute", baseline_font_size=16
    )

    assert content
    assert content.mask is not None
    assert content.image is not None
