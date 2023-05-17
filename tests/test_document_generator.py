import logging
import pytest
import os

from layoutex.document_generator import DocumentGenerator
from layoutex.layout_provider import LayoutProviderConfig
from layoutex.layout_provider_factory import LayoutProviderFactory
from layoutex.logger import configure_logger
from layoutex.io_util import prepare_directory

configure_logger(filename="test_document_generator.log")
logger = logging.getLogger(__name__)

@staticmethod
@pytest.mark.asyncio
def test_document_generator():
    logger.info("Testing DocumentGenerator...")
    output_directory = prepare_directory("/tmp/samples")
    layout_provider = LayoutProviderFactory.get(LayoutProviderConfig(
        type="fixed",
        max_objects=10,
        max_length=100,
        dataset_path="./assets/datasets/publaynet/annotations/val.json", 
        samples_dir=output_directory
    ))
    generator = DocumentGenerator(
        layout_provider=layout_provider,
        target_size=2048,
        solidity=0.5,
        expected_components=["figure", "table"],
        assets_dir="./assets"
    )
    assert generator

    for i in range(1, 20):
        document = generator.render(i)
        logger.debug(document)
        img, mask, layout = document.image, document.mask, document.layout
        
        if img is not None and mask is not None:
            img.save(os.path.join(output_directory, f"rendered_{i}.png"))
            mask.save(os.path.join(output_directory, f"rendered_{i}_mask.png"))