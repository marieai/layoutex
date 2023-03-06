"""
class representing a content provider
"""
import os
import random
import string
from typing import Union

import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageOps

from layoutex.content import (
    Content,
    TableContent,
    FigureContent,
    ParagraphContent,
    ListContent,
)

from faker import Faker


def get_images_from_dir(asset_dir) -> list[Image]:
    assets = []

    for filename in os.listdir(asset_dir):
        try:
            img_path = os.path.join(asset_dir, filename)
            src_img = Image.open(img_path)

            assets.append(src_img)
        except Exception as e:
            raise e

    return assets


def draw_text_with_mask(
    canvas, canvas_mask, text, xy, font, document_size, clip_mask=False
):
    (left, top, right, bottom) = canvas.textbbox((0, 0), text, font)
    word_width = right - left
    word_height = bottom - top

    # text has to be within the bounds otherwise return same image
    x = xy[0]
    y = xy[1]

    img_w = document_size[0]
    img_h = document_size[1]

    adj_y = y + word_height
    adj_w = x + word_width

    # print(f'size : {img_h},  {adj_y},  {word_width}, {word_height} : {xy}')
    if adj_y > img_h or adj_w > img_w:
        return False, (0, 0)

    stroke_width = 0
    stroke_fill = 'black'
    mask_fill = 'black'
    fill = 'black'

    if False and np.random.choice([0, 1], p=[0.9, 0.1]):
        stroke_width = np.random.randint(1, 4)
        stroke_fill = 'black'
        fill = 'white'
        mask_fill = 'red'

    canvas.text(
        xy,
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    try:
        mask, offset = font.getmask2(text, stroke_width=0, stroke_fill=stroke_fill)
        bitmap_image = Image.frombytes(mask.mode, mask.size, bytes(mask))
        # bitmap_image.save(f'/tmp/samples/mask-{text}.png')

        # filter non-black pixels
        # if we don't do this, the mask will anit-aliased and will have some non-black pixels
        if clip_mask:
            bitmap_image = np.array(bitmap_image)
            # bitmap_image[bitmap_image != 0] = 255
            bitmap_image[bitmap_image >= 75] = 255
            bitmap_image[bitmap_image < 75] = 0
            bitmap_image = Image.fromarray(bitmap_image)

        # bitmap_image.save(f'/tmp/samples/mask-{text}-after.png')
        # print(f'offset : {offset} : {xy},  adjust : {offset[0]}, {offset[1]}')

        adjust = (offset[0], offset[1])
        xy = (xy[0] + adjust[0], xy[1] + adjust[1])
        canvas_mask.bitmap(xy, bitmap_image, fill=mask_fill)
        return True, (word_width, word_height)

    except Exception as e:
        return False, (0, 0)


class ContentProvider(object):
    """A object that represents a content provider"""

    def __init__(self):
        pass

    def get_content(
        self,
        component: dict,
        bbox_mode: str = "absolute",
        baseline_font_size: int = 16,
    ) -> Union[TableContent, FigureContent, ParagraphContent, ListContent]:
        pass

    def _convert_relative_bbox(self, bbox):
        pass


class TextContentProvider(ContentProvider):
    """A object that represents a text content provider"""

    def __init__(self, assets_dir: str = None):
        super().__init__()
        self.font_path = os.path.join(assets_dir, "fonts")
        self.patches_full = get_images_from_dir(
            os.path.join(assets_dir, "patches", "full")
        )
        self.patches_medium = get_images_from_dir(
            os.path.join(assets_dir, "patches", "medium")
        )
        self.patches_small = get_images_from_dir(
            os.path.join(assets_dir, "patches", "small")
        )

        self.fonts = os.listdir(self.font_path)
        self.faker = Faker()
        self.fake_names_only = Faker(["it_IT", "en_US", "es_MX", "en_IN"])  # 'de_DE',

    def get_content(
        self,
        component: dict,
        bbox_mode: str = "absolute",
        baseline_font_size: int = 16,
    ) -> ParagraphContent:
        """Get content"""
        print(f"Getting text content for {component}")
        print(component)
        component_type = component["content_type"]
        assert component_type == "text"
        # convert bbox to absolute
        bbox = component["bbox"]
        if bbox_mode == "relative":
            bbox = self._convert_relative_bbox(bbox)

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        density = 0.8
        # create new Pil image to draw on
        pil_img = Image.new('RGB', (w, h), (255, 255, 255))
        pil_img_mask = Image.new('RGB', (w, h), (255, 255, 255))

        background = self.patches_full[np.random.randint(0, len(self.patches_full))]
        background = background.resize((w, h))
        pil_img.paste(background, (0, 0))

        canvas = ImageDraw.Draw(pil_img)
        canvas_mask = ImageDraw.Draw(pil_img_mask)

        pil_img_mask.save('/tmp/samples/canvas.png')
        pil_img.save('/tmp/samples/canvas-mask.png')

        font_path = os.path.join(self.font_path, np.random.choice(self.fonts))
        font_size_est = baseline_font_size + np.random.randint(0, 10)
        font_baseline = np.random.randint(0, 12)
        font = ImageFont.truetype(font_path, font_size_est)
        font_wh = font.getsize("A")
        font_w = font_wh[0]
        font_h = font_wh[1]

        # esimate the number of lines that can fit in the image based on the font size and height of the image

        line_height = font_h + (font_baseline * density)
        print(line_height)
        num_lines = int(h / line_height)
        print(f"Number of lines: {num_lines}")
        estimated_start_x = 0
        for i in range(num_lines):
            # generate random text
            start_x = estimated_start_x
            start_y = i * line_height

            tries = 0
            while True:
                if tries > 3:
                    break

                text = self._generate_text(start_x, w, font_w)
                pos = (start_x, start_y)
                valid, word_size = draw_text_with_mask(
                    canvas, canvas_mask, text, pos, font, (w, h), True
                )

                if not valid:
                    tries += 1
                    continue

                txt_w = word_size[0] + np.random.randint(font_w, font_w * 4)
                # txt_w = word_size[0] + font_w
                print(f' {start_x}, {start_y} :  : {word_size}')
                start_x = start_x + txt_w
                if start_x > w:
                    tries += 1
                    continue

            # draw text
            # canvas.text((0, i * line_height), text, font=font, fill=(0, 0, 0))
            # canvas_mask.text((0, i * line_height), text, font=font, fill=(0, 0, 0))

            # text_size = draw_text_with_mask(
            #     canvas, canvas_mask, text, (0, i * line_height), font, (w, h), True
            # )
            # print(f'text size : {text_size}')

        pil_img.save('/tmp/samples/canvas.png')
        pil_img_mask.save('/tmp/samples/canvas-mask.png')

        return None
        # return ParagraphContent()

    def _generate_text(self, current_x, document_width, font_width):
        """Generate text"""

        print(
            f'current_x: {current_x}, document_width: {document_width}, font_width: {font_width}'
        )

        # Faker.seed(0)
        def g_1():
            estimated_words = int(
                (document_width - current_x) / font_width * 5
            )  # 5 chars per word
            return self.faker.sentence(nb_words=estimated_words)

        def g_2():
            # https://datatest.readthedocs.io/en/stable/how-to/date-time-str.html
            patterns = [
                "%Y%m%d",
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%d.%m.%Y",
                "%d %B %Y",
                "%b %d, %Y",
            ]

            if np.random.choice([0, 1], p=[0.3, 0.7]):
                pattern = random.choice(patterns)
                sel_reg = random.choice(["-", " - ", " ", " to ", " thought "])
                d1 = self.faker.date(pattern=pattern)
                d2 = self.faker.date(pattern=pattern)
                label_text = f"{d1}{sel_reg}{d2}"
            else:
                label_text = self.faker.date(pattern=random.choice(patterns))

            return label_text

        def g_3():
            return self.fake_names_only.name()

        def g_4():
            N = random.choice([4, 6, 8, 10, 12])
            return "".join(random.choices(string.digits, k=N))

        def g_5():
            return self.fake_names_only.phone_number()

        def g_6():
            label_text = self.faker.domain_name()
            if np.random.choice([0, 1], p=[0.5, 0.5]):
                label_text = self.faker.company_email()
            return label_text

        def g_7():
            label_text = self.faker.pricetag()
            if np.random.choice([0, 1], p=[0.5, 0.5]):
                label_text = label_text.replace("$", "")
            return label_text

        generator = random.choice([g_1, g_2, g_3, g_4, g_5, g_6, g_7])

        return generator()


def get_content_provider(content_type: str, assets_dir: str) -> ContentProvider:
    """Get a content provider"""
    if content_type == "text":
        return TextContentProvider(assets_dir=assets_dir)
    raise ValueError(f"Unknown content type {content_type}")
