import pytest

from layoutex.document_generator import DocumentGenerator
from layoutex.layout_provider import LayoutProvider, get_layout_provider


@staticmethod
@pytest.mark.asyncio
def test_document_generator():
    layout_provider = get_layout_provider("fixed", 10, 100)
    generator = DocumentGenerator(
        layout_provider=layout_provider,
        target_size=2048,
        solidity=0.5,
        expected_components=["figure", "table"],
    )
    assert generator

    for i in range(10):
        document = generator.render(i)
        print(document)
        img, mask, layout = document.image, document.mask, document.layout
        img.save(f"/tmp/samples/rendered_{i}.png")
        mask.save(f"/tmp/samples/rendered_{i}_mask.png")
