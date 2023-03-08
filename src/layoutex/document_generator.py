import asyncio

import numpy as np
from typing_extensions import AsyncIterable

from layoutex.content_provider import ContentProvider, get_content_provider
from layoutex.document import Document
from layoutex.layout_provider import LayoutProvider

import asyncio
from codetiming import Timer
from PIL import Image, ImageDraw, ImageOps

from layoutex.layout_transformer.utils import gen_colors


class DocumentGenerator(object):
    """A object that represents a document generator"""

    def __init__(
        self,
        layout_provider: LayoutProvider,
        target_size: int,
        solidity: float,
        expected_components: list,
    ):
        self.layout_provider = layout_provider
        self.count = 0
        self.target_size = target_size
        self.expected_components = expected_components
        self.solidity = solidity

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
                layout = layouts[0]
                width = self.target_size
                height = self.target_size
                # create empty PIL image
                colors = gen_colors(6)
                generated_doc = Image.new("RGB", (width, height), color=(255, 255, 255))
                generated_mask = Image.new(
                    "RGB", (width, height), color=(255, 255, 255)
                )
                canvas = ImageDraw.Draw(generated_doc, "RGBA")
                for i, component in enumerate(layout):
                    provider = get_content_provider(
                        component["content_type"], assets_dir="./assets"
                    )

                    # bounding box mode is relative to the whole document and not the component
                    bbox = np.array(component["bbox"]).astype(np.int32)
                    x1, y1, x2, y2 = bbox
                    bbox2 = [0, 0, x2 - x1, y2 - y1]
                    component["bbox"] = bbox2
                    # convert from relative to absolute coordinates
                    cat_id = component["category_id"]

                    if False:
                        col = colors[cat_id]
                        canvas.rectangle(
                            [x1, y1, x2, y2],
                            outline=tuple(col) + (200,),
                            fill=tuple(col) + (64,),
                            width=2,
                        )

                    image, mask = provider.get_content(
                        component, bbox_mode="absolute", baseline_font_size=16
                    )

                    generated_doc.paste(image, (x1, y1))
                    generated_mask.paste(mask, (x1, y1))
                #
                # generated_doc.save(f"/tmp/samples/rendered_{task_id}.png")
                # generated_mask.save(f"/tmp/samples/rendered_{task_id}_mask.png")

                end = timer()
                delta = end - start
                print(f"Document generation took {delta} seconds")
                return Document(task_id, generated_doc, generated_mask, layout)
            except Exception as e:
                retry_count += 1
                if retry_count > 3:
                    print(f"Failed to generate document after {retry_count} retries")
                    return Document(task_id, None, None, None)

    async def task(self, task_id: int):
        return self.render(task_id)
