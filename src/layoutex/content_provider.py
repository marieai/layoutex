"""
class representing a content provider
"""
import os
import random
import string
from functools import cache

import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageStat
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


@cache
def get_text_dimensions(text_string, font) -> tuple[int, int]:
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()
    bbox = font.getmask(text_string).getbbox()
    text_width = bbox[2]
    text_height = bbox[3] + descent

    return text_width, text_height


def draw_text_with_mask(
    img,
    mask,
    canvas,
    canvas_mask,
    text,
    xy,
    font,
    document_size,
    clip_mask=False,
    inverted=False,
):
    # t_len = canvas.textlength(text, font)
    # (left, top, right, bottom) = canvas.textbbox((0, 0), text, font)
    # word_width = right - left
    # word_height = bottom - top

    word_width, word_height = get_text_dimensions(text, font)
    # print(
    #     f'word_width: {word_width}, word_height: {word_height}, text_width : {text_width} {text_height}'
    # )

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

    if inverted:
        # based on the mean of the image, decide how to draw the inverted text
        im1 = img.crop(
            (
                x - word_width // 2,
                y - word_height // 2,
                x + word_width * 2,
                y + word_height * 2,
            )
        )
        # im1 = im1.convert("L")
        import cv2

        img = np.array(im1)
        # numpy blur to remove noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        m = np.mean(img)
        del im1
        del img
        method = 0
        if m < 20:  # very dark background
            stroke_width = 0
            stroke_fill = "black"
            fill = "white"
            mask_fill = "black"
            method = 0
        elif m < 60:  # dark background
            stroke_width = np.random.randint(2, 4)
            stroke_fill = (int(m / 3), int(m / 3), int(m / 3))  # "black"
            fill = "white"
            mask_fill = "black"
            method = 1
        elif 60 < m < 90:  # light background
            stroke_width = 2
            stroke_fill = (int(m / 3), int(m / 3), int(m / 3))
            fill = "white"
            mask_fill = "black"
            method = 2
        elif 90 < m < 150:  # light background
            stroke_width = 1
            stroke_fill = "white"
            fill = "black"
            mask_fill = "black"
            method = 3
        else:  # very light background
            stroke_width = 0
            stroke_fill = "white"
            fill = "black"
            mask_fill = "black"
            method = 4

        # print(m, method, text)

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
        print(f"Error : {e}")
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

        self.blocks_inverted = get_images_from_dir(os.path.join(assets_dir, "blocks"))
        # self.blocks_inverted_bump_map = create_bump_map(self.blocks_inverted)

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

        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])

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
        img,
        mask,
        canvas,
        canvas_mask,
        density,
        height,
        width,
        spacing="dense",
        tabular=False,
        fit_font_to_bbox=False,
        inverted=False,
    ):
        header = False
        if fit_font_to_bbox:
            baseline_font_size = int(height * 0.8)
            header = True

        font, font_baseline, font_h, font_w, line_height = self.measure_fonts(
            baseline_font_size, density
        )
        # estimate number of lines that can fit in the image based on the font size and height of the image
        num_lines = int(height / line_height)
        # print(f"Number of lines: {num_lines}")

        estimated_colum_size = int(width / np.random.randint(5, 10))
        estimated_start_x = 0
        max_tries = 50

        if fit_font_to_bbox:
            num_lines = 1
        tries = 0

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
                text = self._generate_text(
                    start_x, width, font_w, tabular=tabular, header=header
                )

                if text is None or text == "":
                    tries += 1
                    continue

                word_width = font_w * len(text)

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

                space_left = width - start_x
                min_word_width = font_w * 4

                if space_left < min_word_width and tries > 5:
                    # print(f"Space left: {space_left} < {min_word_width}")
                    break

                if start_x + txt_spacing > width:
                    # print(f"Start x + txt spacing > width : {tries}")
                    tries += 1
                    continue

                pos = (start_x, start_y)
                valid, word_size = draw_text_with_mask(
                    img,
                    mask,
                    canvas,
                    canvas_mask,
                    text,
                    pos,
                    font,
                    (width, height),
                    True,
                    inverted=inverted,
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
        self,
        current_x,
        document_width,
        font_width,
        tabular: bool = False,
        header: bool = False,
    ) -> str:
        """Generate text"""
        #
        # print(
        #     f"current_x: {current_x}, document_width: {document_width}, font_width: {font_width}"
        # )

        # Faker.seed(0)
        def g_text():
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
        elif header:
            generator = random.choice([g_text, self.faker.company])
        else:
            generator = random.choice(
                [g_text, g_date, g_names, g_accounts, g_5, g_6, g_amount]
            )

        text = generator()
        if np.random.random() > 0.5:
            text = text.upper()
        return text

    def overlay_background(
        self,
        img,
        mask,
        canvas,
        canvas_mask,
        h,
        w,
        component: dict,
        inverted: bool = False,
    ):
        if inverted:
            background = self.blocks_inverted[
                np.random.randint(0, len(self.blocks_inverted))
            ]
            # background = background.resize((w, h))
        else:
            #
            # get component size and decide if we need to use half or full page

            # "FULL_WIDTH", "LINE_HEIGHT"
            sizing_x = component["sizing"][0]
            sizing_y = component["sizing"][1]

            if sizing_y == "QUARTER_HEIGHT":
                background = self.patches_half[
                    np.random.randint(0, len(self.patches_half))
                ]
            elif sizing_y == "HALF_HEIGHT":
                background = self.patches_half[
                    np.random.randint(0, len(self.patches_half))
                ]
            else:
                background = self.patches_full[
                    np.random.randint(0, len(self.patches_full))
                ]

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
        img, mask, canvas, canvas_mask = self.create_image_and_mask(
            component, bbox_mode
        )

        bbox = component["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # candidates for inverted text
        # "FULL_WIDTH", "LINE_HEIGHT"
        sizing_x = component["sizing"][0]
        sizing_y = component["sizing"][1]

        inverted = False
        if sizing_x in ["FULL_WIDTH", "HALF_WIDTH"] and sizing_y == "LINE_HEIGHT":
            # 50% chance of inverted text
            if np.random.choice([0, 1], p=[0.5, 0.5]):
                inverted = True

        self.overlay_background(
            img, mask, canvas, canvas_mask, h, w, component, inverted=inverted
        )
        self.render_text(
            baseline_font_size,
            img,
            mask,
            canvas,
            canvas_mask,
            density,
            h,
            w,
            "dense",
            tabular=False,
            fit_font_to_bbox=False,
            inverted=inverted,
        )

        # img.save("/tmp/samples/canvas.png")
        # mask.save("/tmp/samples/canvas-mask.png")

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
        img, mask, canvas, canvas_mask = self.create_image_and_mask(
            component, bbox_mode
        )

        bbox = component["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        self.overlay_background(img, mask, h, w, component)

        self.render_text(
            baseline_font_size,
            img,
            mask,
            canvas,
            canvas_mask,
            density,
            h,
            w,
            "even",
            True,
        )

        # img.save("/tmp/samples/canvas.png")
        # mask.save("/tmp/samples/canvas-mask.png")

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
        # table_dir = os.path.join(self.assets_dir, "tables", "table-008")
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
        # background.save("/tmp/samples/table.png")
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

        self.overlay_background(img, mask, canvas, canvas_mask, h, w, component)

        # self.render_text(
        #     baseline_font_size, canvas, canvas_mask, density, h, w, "loose", True
        # )

        # img.save("/tmp/samples/canvas.png")
        # mask.save("/tmp/samples/canvas-mask.png")

        return img, mask

    def overlay_background(self, img, mask, canvas, canvas_mask, h, w, component: dict):
        def generator_barcode():
            from barcode.writer import ImageWriter
            from barcode import generate
            import io

            fp = io.BytesIO()
            generate("code128", self.faker.company(), writer=ImageWriter(), output=fp)
            barcode = Image.open(fp)
            barcode_w, barcode_h = barcode.size

            if h > barcode_h:
                s = 0.8
                r = w / barcode.width
                barcode = barcode.resize((int(w * s), int(barcode.height * r * s)))
            else:
                # resize to the height of the image
                barcode = barcode.resize((int(barcode_w * h / barcode_h), h))

            # rotating a image 90 deg counter clockwise
            # barcode = barcode.rotate(90, PIL.Image.NEAREST, expand=1)

            barcode_w, barcode_h = barcode.size
            barcode_pos = (
                w // 2 - barcode_w // 2,
                h // 2 - barcode_h // 2,
            )

            img.paste(barcode, barcode_pos)
            mask.paste(barcode, barcode_pos)

        def generator_qrcode():
            import qrcode

            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(self.faker.company())
            qr.make(fit=True)

            qrcode_img = qr.make_image(fill_color="black", back_color="white")
            qrcode_img = qrcode_img.convert("RGB")
            # qrcode_img.save("/tmp/samples/qrcode.png")
            barcode_w, barcode_h = qrcode_img.size

            if h > barcode_h:
                s = 0.8
                r = w / qrcode_img.width
                qrcode_img = qrcode_img.resize(
                    (int(w * s), int(qrcode_img.height * r * s))
                )
            else:
                # resize to the height of the image
                qrcode_img = qrcode_img.resize((int(barcode_w * h / barcode_h), h))

            img_w, img_h = qrcode_img.size
            qrcode_pos = (
                w // 2 - img_w // 2,
                h // 2 - img_h // 2,
            )

            img.paste(qrcode_img, qrcode_pos)
            mask.paste(qrcode_img, qrcode_pos)

        def generator_logo():
            logos = get_images_from_dir(os.path.join(self.assets_dir, "logos"))
            logo = random.choice(logos)
            logo_w, logo_h = logo.size[0], logo.size[1]
            logo_pos = (0, 0)
            font_size = 16 + np.random.randint(20, 35)
            possible_alignments = ["left", "right", "under"]

            # resize image to half of the height of the image and half of the width of the image
            if h > logo_h:
                # resize the logo to 25% of the image
                # logo = logo.resize((int(w * 0.33), int(logo_W * w * 0.33 / logo_w)))
                logo = logo.resize((int(logo_w * h * 0.5 / logo_h), int(h * 0.5)))
                logo_w, logo_h = logo.size
            else:
                # resize to the height of the image
                logo = logo.resize((int(logo_w * h * 0.8 / logo_h), int(h * 0.8)))
                logo_w, logo_h = logo.size
                logo_pos = (0, h // 2 - logo_h // 2)
                font_size = logo_h - logo_h // 4
                possible_alignments = ["left", "right"]

            font, font_baseline, font_h, font_w, line_height = self.measure_fonts(
                baseline_font_size=font_size, density=0.8
            )
            attempt = 0
            text = self.faker.company()

            while attempt < 10:
                attempt += 1
                if np.random.random() > 0.5:
                    text = text.upper()

                # generate text and measure it
                (left, top, right, bottom) = canvas.textbbox((0, 0), text, font)
                word_width = right - left
                word_height = bottom - top

                # print(f"Logo size: {logo_w}x{logo_h}")
                # print(f"Text size: {word_width}x{word_height}")

                if logo_w + word_width > w:
                    # text is too long, trim it
                    text = text[: int(len(text) - len(text) * 0.2)]
                    continue

                align = np.random.choice(possible_alignments)
                pos = (0, 0)
                if align == "right":
                    pos = (
                        logo_pos[0] + logo_w,
                        logo_pos[1] + logo_h // 2 - word_height // 2,
                    )
                elif align == "left":
                    pos = (logo_pos[0], logo_pos[1] + logo_h // 2 - word_height // 2)
                    logo_pos = (logo_pos[0] + word_width, logo_pos[1])
                elif align == "under":
                    pos = (logo_pos[0], logo_pos[1] + logo_h)

                valid, word_size = draw_text_with_mask(
                    img, mask, canvas, canvas_mask, text, pos, font, (w, h), True
                )

                if valid:
                    break

            img.paste(logo, logo_pos)
            mask.paste(logo, logo_pos)

        generator = random.choice([generator_barcode, generator_qrcode, generator_logo])
        # generator_qrcode()
        # generator_logo()
        generator()


class TitleContentProvider(ContentProvider):
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
        img, mask, canvas, canvas_mask = self.create_image_and_mask(
            component, bbox_mode
        )

        bbox = component["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # self.overlay_background(img, mask, h, w, component)
        self.render_text(
            baseline_font_size,
            img,
            mask,
            canvas,
            canvas_mask,
            density,
            h,
            w,
            "dense",
            False,
            fit_font_to_bbox=True,
        )

        # img.save("/tmp/samples/canvas.png")
        # mask.save("/tmp/samples/canvas-mask.png")

        return img, mask


def get_content_provider(content_type: str, assets_dir: str) -> ContentProvider:
    """Get a content provider"""
    if content_type == "paragraph":
        return TextContentProvider(assets_dir=assets_dir)

    if content_type in ["table", "list"]:
        return TableContentProvider(assets_dir=assets_dir)

    if content_type == "figure":
        return FigureContentProvider(assets_dir=assets_dir)

    if content_type == "title":
        return TitleContentProvider(assets_dir=assets_dir)

    raise ValueError(f"Unknown content type {content_type}")
