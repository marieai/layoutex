import pytest

from layoutex.content_provider import get_content_provider


def test_text_content_provider():
    component = {
        "content_type": "paragraph",
        "bbox": [0, 0, 1024, 1024],
        "sizing": ["FULL_WIDTH", "FULL_HEIGHT"],
    }

    provider = get_content_provider(component["content_type"], assets_dir="./assets")
    assert provider

    image, mask = provider.get_content(
        component, bbox_mode="absolute", baseline_font_size=16
    )

    assert mask is not None
    assert image is not None

    assert image.size == mask.size


def test_text_content_inverted_provider():

    component = {
        "content_type": "text",
        "bbox": [0, 0, 1024, 80],
        "sizing": ["FULL_WIDTH", "LINE_HEIGHT"],
    }

    provider = get_content_provider(component["content_type"], assets_dir="./assets")
    assert provider

    image, mask = provider.get_content(
        component, bbox_mode="absolute", baseline_font_size=16
    )

    assert mask is not None
    assert image is not None

    assert image.size == mask.size


def test_table_content_provider():
    component = {
        "content_type": "table",
        "bbox": [0, 0, 1024, 1024],
        "sizing": ["FULL_WIDTH", "FULL_HEIGHT"],
    }

    provider = get_content_provider(component["content_type"], assets_dir="./assets")
    assert provider

    image, mask = provider.get_content(
        component, bbox_mode="absolute", baseline_font_size=16
    )

    assert mask is not None
    assert image is not None

    assert image.size == mask.size


def test_figure_content_provider():
    component = {
        "content_type": "figure",
        "bbox": [0, 0, 512, 256],
        "sizing": ["FULL_WIDTH", "FULL_HEIGHT"],
    }

    provider = get_content_provider(component["content_type"], assets_dir="./assets")
    assert provider

    image, mask = provider.get_content(
        component, bbox_mode="absolute", baseline_font_size=16
    )

    assert mask is not None
    assert image is not None

    assert image.size == mask.size


def test_title_content_provider():
    component = {
        "content_type": "title",
        "bbox": [0, 0, 512, 40],
        "sizing": ["FULL_WIDTH", "FULL_HEIGHT"],
    }

    provider = get_content_provider(component["content_type"], assets_dir="./assets")
    assert provider

    image, mask = provider.get_content(
        component, bbox_mode="absolute", baseline_font_size=16
    )

    assert mask is not None
    assert image is not None

    assert image.size == mask.size


def test_table_content_provider_masked_001():
    component = {
        "content_type": "table",
        "bbox": [0, 0, 1024, 1024],
        "sizing": ["FULL_WIDTH", "FULL_HEIGHT"],
    }

    provider = get_content_provider(component["content_type"], assets_dir="./assets")
    assert provider

    image, mask = provider.get_content(
        component, bbox_mode="absolute", baseline_font_size=16
    )

    assert mask is not None
    assert image is not None

    assert image.size == mask.size
