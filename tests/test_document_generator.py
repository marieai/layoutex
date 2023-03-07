import pytest

from layoutex.document_generator import DocumentGenerator
from layoutex.layout_provider import LayoutProvider, get_layout_provider


def test_document_generator():
    layout_provider = get_layout_provider("fixed", 10, 100)
    generator = DocumentGenerator(
        layout_provider=layout_provider,
        target_size=1024*1,
        doc_count=10,
        solidity=0.5,
        expected_components=["figure", "table"],
    )
    assert generator

    for document in generator:
        print(document)
