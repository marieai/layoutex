"""
class representing a content provider
"""
import os
import random
import string
from typing import Union, Tuple, List, Optional

import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageOps
import PIL

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
    stroke_fill = "black"
    mask_fill = "black"
    fill = "black"

    if False and np.random.choice([0, 1], p=[0.9, 0.1]):
        stroke_width = np.random.randint(1, 4)
        stroke_fill = "black"
        fill = "white"
        mask_fill = "red"

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

    def __init__(self, assets_dir: str = None):
        self.assets_dir = assets_dir
        self.font_path = os.path.join(assets_dir, "fonts")
        self.fonts = os.listdir(self.font_path)

        self.patches_full = get_images_from_dir(
            os.path.join(assets_dir, "patches", "full")
        )
        self.patches_half = get_images_from_dir(
            os.path.join(assets_dir, "patches", "half")
        )

        self.patches_quarter = get_images_from_dir(
            os.path.join(assets_dir, "patches", "quarter")
        )

        self.faker = Faker()
        self.fake_names_only = Faker(
            ["it_IT", "en_US", "en_IN"]
        )  # "es_MX",  # 'de_DE',

    def get_content(
        self,
        component: dict,
        bbox_mode: str = "absolute",
        baseline_font_size: int = 16,
    ) -> tuple[Image, Image]:
        # Union[TableContent, FigureContent, ParagraphContent, ListContent]:
        pass

    def _convert_relative_bbox(self, bbox):
        pass

    def create_image_and_mask(
        self, component: dict, bbox_mode: str = "absolute"
    ) -> tuple[Image, Image, ImageDraw, ImageDraw]:
        bbox = component["bbox"]
        if bbox_mode == "relative":
            bbox = self._convert_relative_bbox(bbox)

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # create new Pil image to draw on
        pil_img = Image.new("RGB", (w, h), (255, 255, 255))
        pil_img_mask = Image.new("RGB", (w, h), (255, 255, 255))
        canvas = ImageDraw.Draw(pil_img)
        canvas_mask = ImageDraw.Draw(pil_img_mask)

        return pil_img, pil_img_mask, canvas, canvas_mask

    def measure_fonts(
        self, baseline_font_size, density
    ) -> tuple[ImageFont, int, int, int, int]:
        """
        Measure fonts to get the baseline and height
        Args:
            density:
            baseline_font_size:

        Returns:

        """
        font_path = os.path.join(self.font_path, np.random.choice(self.fonts))
        font_size_est = baseline_font_size + np.random.randint(0, 10)
        font_baseline = np.random.randint(0, 12)
        font = ImageFont.truetype(font_path, font_size_est)
        font_wh = font.getsize("A")
        font_w = font_wh[0]
        font_h = font_wh[1]

        line_height = font_h + (font_baseline * density)
        return font, font_baseline, font_h, font_w, line_height

    def render_text(
        self,
        baseline_font_size,
        canvas,
        canvas_mask,
        density,
        height,
        width,
        spacing="dense",
        tabular=False,
    ):

        font, font_baseline, font_h, font_w, line_height = self.measure_fonts(
            baseline_font_size, density
        )
        # estimate number of lines that can fit in the image based on the font size and height of the image
        num_lines = int(height / line_height)
        # print(f"Number of lines: {num_lines}")

        estimated_colum_size = int(width / 6)
        estimated_start_x = 0
        max_tries = 10

        for i in range(num_lines):
            # generate random text
            start_x = estimated_start_x
            start_y = i * line_height
            tries = 0
            column = 0

            while True:
                if tries > max_tries:
                    break

                # generate text and measure it
                text = self._generate_text(start_x, width, font_w, tabular=tabular)
                (left, top, right, bottom) = canvas.textbbox((0, 0), text, font)
                word_width = right - left
                word_height = bottom - top

                # check if the text fits in the image
                if spacing == "dense":
                    txt_spacing = word_width + np.random.randint(font_w, font_w * 4)
                elif spacing == "loose":
                    txt_spacing = word_width + np.random.randint(font_w * 4, font_w * 8)
                elif spacing == "normal":
                    txt_spacing = word_width + font_w
                elif spacing == "even":
                    if word_width > estimated_colum_size:
                        tries += 1
                        continue
                    start_x = column * estimated_colum_size
                    txt_spacing = 0
                else:
                    raise ValueError(f"Spacing {spacing} not supported")

                # txt_w = word_size[0] + font_w
                print(f" {start_x}, {start_y} :  : {word_width}, {word_height}")

                if start_x + txt_spacing > width:
                    tries += 1
                    continue

                pos = (start_x, start_y)
                valid, word_size = draw_text_with_mask(
                    canvas, canvas_mask, text, pos, font, (width, height), True
                )

                start_x = start_x + txt_spacing
                if not valid:
                    tries += 1
                    continue
                column += 1

            # draw text
            # canvas.text((0, i * line_height), text, font=font, fill=(0, 0, 0))
            # canvas_mask.text((0, i * line_height), text, font=font, fill=(0, 0, 0))

            # text_size = draw_text_with_mask(
            #     canvas, canvas_mask, text, (0, i * line_height), font, (w, h), True
            # )
            # print(f'text size : {text_size}')

    def _generate_text(
        self, current_x, document_width, font_width, tabular: bool = False
    ) -> str:
        """Generate text"""
        #
        # print(
        #     f"current_x: {current_x}, document_width: {document_width}, font_width: {font_width}"
        # )

        # Faker.seed(0)
        def g_1():
            estimated_words = int(
                (document_width - current_x) / font_width * 5
            )  # 5 chars per word
            return self.faker.sentence(nb_words=estimated_words)

        def g_date():
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

            if tabular:
                label_text = self.faker.date(pattern=random.choice(patterns))
                return label_text

            if np.random.choice([0, 1], p=[0.3, 0.7]):
                pattern = random.choice(patterns)
                sel_reg = random.choice(["-", " - ", " ", " to ", " thought "])
                d1 = self.faker.date(pattern=pattern)
                d2 = self.faker.date(pattern=pattern)
                label_text = f"{d1}{sel_reg}{d2}"
            else:
                label_text = self.faker.date(pattern=random.choice(patterns))

            return label_text

        def g_names():
            return self.fake_names_only.name()

        def g_accounts():
            N = random.choice([1, 4, 6, 8, 10, 12])
            return "".join(random.choices(string.digits, k=N))

        def g_5():
            return self.fake_names_only.phone_number()

        def g_6():
            label_text = self.faker.domain_name()
            if np.random.choice([0, 1], p=[0.5, 0.5]):
                label_text = self.faker.company_email()
            return label_text

        def g_amount():
            label_text = self.faker.pricetag()
            if np.random.choice([0, 1], p=[0.5, 0.5]):
                label_text = label_text.replace("$", "")
            return label_text

        if tabular:
            generator = random.choice([g_date, g_names, g_amount, g_accounts])
        else:
            generator = random.choice(
                [g_1, g_date, g_names, g_accounts, g_5, g_6, g_amount]
            )

        return generator()

    def overlay_background(self, img, mask, h, w, component: dict):
        background = self.patches_full[np.random.randint(0, len(self.patches_full))]
        background = background.resize((w, h))
        img.paste(background, (0, 0))


class TextContentProvider(ContentProvider):
    """A object that represents a text content provider"""

    def __init__(self, assets_dir: str = None):
        super().__init__(assets_dir=assets_dir)

    def get_content(
        self,
        component: dict,
        bbox_mode: str = "absolute",
        baseline_font_size: int = 16,
        density: float = 0.8,
    ) -> tuple[Image, Image]:
        """Get content"""
        print(f"Getting text content for {component}")
        img, mask, canvas, canvas_mask = self.create_image_and_mask(
            component, bbox_mode
        )

        bbox = component["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        self.overlay_background(img, mask, h, w, component)
        # self.render_text(
        #     baseline_font_size, canvas, canvas_mask, density, h, w, "dense", False
        # )

        img.save("/tmp/samples/canvas.png")
        mask.save("/tmp/samples/canvas-mask.png")

        return img, mask


class TableContentProvider(ContentProvider):
    """A object that represents a text content provider"""

    def __init__(self, assets_dir: str = None):
        super().__init__(assets_dir=assets_dir)

    def get_content(
        self,
        component: dict,
        bbox_mode: str = "absolute",
        baseline_font_size: int = 16,
        density: float = 0.8,
    ) -> tuple[Image, Image]:
        """Get content"""
        print(f"Getting table content for {component}")
        img, mask, canvas, canvas_mask = self.create_image_and_mask(
            component, bbox_mode
        )

        bbox = component["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        self.overlay_background(img, mask, h, w, component)

        self.render_text(
            baseline_font_size, canvas, canvas_mask, density, h, w, "even", True
        )

        img.save("/tmp/samples/canvas.png")
        mask.save("/tmp/samples/canvas-mask.png")

        return img, mask

    def overlay_background(self, img, mask, h, w, component: dict):
        tables_dir = os.path.join(self.assets_dir, "tables")
        dirs = [
            d
            for d in os.listdir(tables_dir)
            if os.path.isdir(os.path.join(tables_dir, d))
        ]

        # pick a random directory
        table_dir = os.path.join(tables_dir, random.choice(dirs))
        table_dir = os.path.join(self.assets_dir, "tables", "table-008")
        table_patches = {}
        for filename in os.listdir(table_dir):
            try:
                group = filename.split("/")[-1].split(".")[1]
                if group not in table_patches:
                    table_patches[group] = []
                table_patches[group].append(filename)
            except Exception as e:
                raise e

        def load_one_of(table_patches, group: str, w: int):
            patch_path = os.path.join(table_dir, random.choice(table_patches[group]))
            img = Image.open(patch_path)
            # resize the patch to the width of the image keeping the aspect ratio
            return img.resize((w, int(img.height * w / img.width)))

        # create a random table by selecting a random patch for each group of patches and pasting them together
        def get_random_table_patch():
            table_background = Image.new("RGB", (w, h))
            start_y = 0

            if "header" in table_patches:
                header_img = load_one_of(table_patches, "header", w)
                table_background.paste(header_img, (0, start_y), header_img)
                start_y = header_img.height

            if "row" in table_patches:
                while start_y < h:
                    row_img = load_one_of(table_patches, "row", w)
                    table_background.paste(row_img, (0, start_y), row_img)
                    start_y = start_y + row_img.height

            if "body" in table_patches:
                while start_y < h:
                    row_img = load_one_of(table_patches, "body", w)
                    table_background.paste(row_img, (0, start_y), row_img)
                    start_y = start_y + row_img.height

            if "footer" in table_patches:
                footer_img = load_one_of(table_patches, "footer", w)
                table_background.paste(
                    footer_img, (0, h - footer_img.height), footer_img
                )
            return table_background

        background = get_random_table_patch()
        background.save("/tmp/samples/table.png")
        background = background.resize((w, h))
        img.paste(background, (0, 0))


class FigureContentProvider(ContentProvider):
    """A object that represents a text content provider"""

    def __init__(self, assets_dir: str = None):
        super().__init__(assets_dir=assets_dir)

    def get_content(
        self,
        component: dict,
        bbox_mode: str = "absolute",
        baseline_font_size: int = 16,
        density: float = 0.8,
    ) -> tuple[Image, Image]:
        """Get content"""
        print(f"Getting table content for {component}")
        img, mask, canvas, canvas_mask = self.create_image_and_mask(
            component, bbox_mode
        )

        bbox = component["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        self.overlay_background(img, mask, h, w, component)

        # self.render_text(
        #     baseline_font_size, canvas, canvas_mask, density, h, w, "loose", True
        # )

        img.save("/tmp/samples/canvas.png")
        mask.save("/tmp/samples/canvas-mask.png")

        return img, mask

    def overlay_background(self, img, mask, h, w, component: dict):
        from barcode.writer import ImageWriter
        from barcode import generate
        import io

        fp = io.BytesIO()
        generate("code128", self.faker.company(), writer=ImageWriter(), output=fp)
        barcode = Image.open(fp)

        s = 0.8
        r = w / barcode.width
        barcode = barcode.resize((int(w * s), int(barcode.height * r * s)))

        # rotating a image 90 deg counter clockwise
        # barcode = barcode.rotate(90, PIL.Image.NEAREST, expand=1)

        barcode_w, barcode_h = barcode.size
        barcode_pos = (
            w // 2 - barcode_w // 2,
            h // 2 - barcode_h // 2,
        )

        img.paste(barcode, barcode_pos)
        mask.paste(barcode, barcode_pos)


def get_content_provider(content_type: str, assets_dir: str) -> ContentProvider:
    """Get a content provider"""
    if content_type == "text":
        return TextContentProvider(assets_dir=assets_dir)

    if content_type == "table":
        return TableContentProvider(assets_dir=assets_dir)

    if content_type == "figure":
        return FigureContentProvider(assets_dir=assets_dir)

    raise ValueError(f"Unknown content type {content_type}")
