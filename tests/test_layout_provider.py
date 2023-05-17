import pytest

from layoutex.layout_provider import LayoutProvider
from layoutex.layout_provider_factory import LayoutProviderFactory


def test_fixed_layout_provider():
    layout = LayoutProviderFactory.get_basic("fixed", 10, 100)
    assert isinstance(layout, LayoutProvider)
    doc_count = 1
    documents = layout.get_layouts(
        document_count=doc_count,
        solidity=0.5,
        expected_components=["figure", "table"],
        target_size=1024 * 1,
    )
    assert len(documents) == doc_count


def test_unknown_layout_provider():
    with pytest.raises(ValueError):
        LayoutProviderFactory.get_basic("unknown", 10, 100)
