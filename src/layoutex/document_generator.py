import asyncio
import os
import threading
import logging
from threading import Thread

import numpy as np
from typing_extensions import AsyncIterable

from layoutex.content_provider_factory import ContentProviderFactory
from layoutex.content_provider import (
    get_images_from_dir,
)
from layoutex.document import Document
from layoutex.layout_provider import LayoutProvider

import asyncio
from codetiming import Timer
from PIL import Image, ImageDraw, ImageOps

from layoutex.layout_transformer.utils import gen_colors

logger = logging.getLogger(__name__)


class DocumentGenerator(object):
    """A object that represents a document generator"""

    def __init__(
        self,
        layout_provider: LayoutProvider,
        target_size: int,
        solidity: float,
        expected_components: list,
        assets_dir: str = "./assets",
    ):
        self.layout_provider = layout_provider
        self.count = 0
        self.target_size = target_size
        self.expected_components = expected_components
        self.solidity = solidity
        self.assets_dir = os.path.expanduser(assets_dir)

    async def render_documents(self, count: int) -> AsyncIterable[Document]:
        for i in range(count):
            yield self.render(i)

    def __aiter__(self):
        self.iter_keys = iter(range(100))
        return self

    async def __anext__(self):
        try:
            # extract the keys one at a time
            key = next(self.iter_keys)
        except StopIteration:
            raise StopAsyncIteration
        return self.render(key)

    def render(self, task_id: int) -> Document:
        retry_count = 0
        overlays = get_images_from_dir(
            os.path.join(self.assets_dir, "patches", "overlay")
        )

        overlays = [
            overlay.resize((self.target_size, self.target_size)) for overlay in overlays
        ]
        rng = np.random.default_rng(threading.get_native_id())

        while True:
            try:
                """
                Render a document
                Args:
                    task_id: unique id for the task

                Returns: a document

                """
                from timeit import default_timer as timer

                start = timer()
                layouts = self.layout_provider.get_layouts(
                    target_size=self.target_size,
                    document_count=1,
                    solidity=self.solidity,
                    expected_components=self.expected_components,
                )
                logger.debug(f"using first layout of {len(layouts)}")
                layout = layouts[0]
                width = self.target_size
                height = self.target_size

                # create empty PIL image
                colors = gen_colors(6)
                generated_doc = Image.new(
                    "RGBA", (width, height), color=(255, 255, 255)
                )

                # get random overlay if needed
                if rng.integers(0, 2) == 1:
                    logger.debug("getting random overlay...")
                    # idx = np.random.randint(0, len(overlays))
                    idx = rng.integers(0, len(overlays))
                    logger.debug(f"Using overlay {idx} of {len(overlays)}")
                    overlay = overlays[idx]

                    # check if rotation is needed
                    if rng.integers(0, 2) == 1:
                        logger.debug("randomly rotating overlay...")
                        # get random rotation angle from set of angles
                        # angle = np.random.choice([0, 90, 180, 270])
                        angle = rng.choice([0, 90, 180, 270])
                        logger.debug(f"Using rotation angle {angle}")
                        overlay = overlay.rotate(angle, expand=True)

                    logger.debug("blending overlay into generated doc...")
                    # blend overlay with document
                    generated_doc = Image.blend(generated_doc, overlay, 0.5)

                generated_mask = Image.new(
                    "RGB", (width, height), color=(255, 255, 255)
                )
                canvas = ImageDraw.Draw(generated_doc, "RGBA")
                for i, component in enumerate(layout):
                    provider = ContentProviderFactory.get(
                        component["content_type"], assets_dir=self.assets_dir
                    )

                    # bounding box mode is relative to the whole document and not the component
                    x1, y1, x2, y2 = np.array(component["bbox"]).astype(np.int32)
                    component["bbox"] = [0, 0, x2 - x1, y2 - y1]
                    # convert from relative to absolute coordinates
                    cat_id = component["category_id"]
                    logger.debug(f"computed category_id = {cat_id}")
                    logger.debug(f"bbox dimensions      = (x1=0, y1=0, x2={x2 - x1}, y2={y2 - y1})")

                    if False:
                        col = colors[cat_id]
                        canvas.rectangle(
                            [x1, y1, x2, y2],
                            outline=tuple(col) + (200,),
                            fill=tuple(col) + (64,),
                            width=2,
                        )

                    logger.debug("getting content from provider...")
                    image, mask = provider.get_content(
                        component, bbox_mode="absolute", baseline_font_size=25
                    )

                    logger.debug("pasting content on generated doc and mask...")

                    generated_doc.paste(image, (x1, y1))
                    generated_mask.paste(mask, (x1, y1))

                end = timer()
                delta = end - start
                logger.info(f"document generation took {delta} seconds")

                # convert to cv2 and binarize the original image
                import cv2

                def convert_pil_to_cv2(pil_img):
                    open_cv_image = np.array(pil_img)
                    # Convert RGB to BGR
                    open_cv_image = open_cv_image[:, :, ::-1].copy()
                    return open_cv_image

                def binarize(pil_img):
                    img = convert_pil_to_cv2(pil_img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.threshold(
                        img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                    )[1]
                    return Image.fromarray(img)

                logger.debug("binarizing generated doc...")
                generated_doc = binarize(generated_doc)

                return Document(task_id, generated_doc, generated_mask, layout)
            except Exception as e:
                retry_count += 1
                if retry_count > 3:
                    logger.error(
                        f"Failed to generate document after {retry_count} retries : {e} "
                    )
                    return Document(task_id, None, None, None)

    async def task(self, task_id: int):
        return self.render(task_id)
