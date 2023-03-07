import pytest

from layoutex.layout_provider import LayoutProvider, get_layout_provider


def test_fixed_layout_provider():
    layout = get_layout_provider("fixed", 10, 100)
    assert isinstance(layout, LayoutProvider)
    doc_count = 1
    documents = layout.get_layouts(
        document_count=doc_count, solidity=0.5, expected_components=["figure", "table"]
    )
    assert len(documents) == doc_count


def test_unknown_layout_provider():
    with pytest.raises(ValueError):
        get_layout_provider("unknown", 10, 100)


