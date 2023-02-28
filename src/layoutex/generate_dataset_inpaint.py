import sys, os, glob, time, pdb, cv2
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import shutil
import random
import string
import PIL
from resize_image import resize_image
import config as cfg
from PIL import ImageFont, ImageDraw, Image, ImageOps

import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor

from faker import Faker

fake = Faker()


def q(text=''):
    print(f'>{text}<')
    sys.exit()


data_dir = cfg.data_dir
train_dir = cfg.train_dir
val_dir = cfg.val_dir

imgs_dir = cfg.imgs_dir
noisy_dir = cfg.noisy_dir
debug_dir = cfg.debug_dir
patch_dir = cfg.patch_dir

asset_dir = cfg.asset_dir

train_data_dir = os.path.join(data_dir, train_dir)
val_data_dir = os.path.join(data_dir, val_dir)

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(train_data_dir):
    os.mkdir(train_data_dir)

if not os.path.exists(val_data_dir):
    os.mkdir(val_data_dir)

img_train_dir = os.path.join(data_dir, train_dir, imgs_dir)
noisy_train_dir = os.path.join(data_dir, train_dir, noisy_dir)
debug_train_dir = os.path.join(data_dir, train_dir, debug_dir)

img_val_dir = os.path.join(data_dir, val_dir, imgs_dir)
noisy_val_dir = os.path.join(data_dir, val_dir, noisy_dir)
debug_val_dir = os.path.join(data_dir, val_dir, debug_dir)

dir_list = [
    img_train_dir,
    noisy_train_dir,
    debug_train_dir,
    img_val_dir,
    noisy_val_dir,
    debug_val_dir,
]
for dir_path in dir_list:
    print(f'dir_path = {dir_path}')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def get_word_list():
    f = open(cfg.txt_file_dir, encoding='utf-8', mode="r")
    text = f.read()
    f.close()
    lines_list = str.split(text, '\n')
    while '' in lines_list:
        lines_list.remove('')

    lines_word_list = [str.split(line) for line in lines_list]
    words_list = [words for sublist in lines_word_list for words in sublist]

    return words_list


def __scale_width(img, long_side):
    size = img.shape[:2]
    oh, ow = size
    ratio = oh / ow
    new_width = long_side
    new_height = int(ratio * new_width)

    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def resize_image(image, desired_size, color=(255, 255, 255)):
    '''Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------

    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size
    '''

    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(
            image, (new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC
        )
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image


def get_size(load_size, size):
    w, h = size
    new_w = w

    new_w = load_size
    new_h = load_size * h // w

    return new_w, new_h


def __frame_image(img, size):
    h = img.shape[0]
    w = img.shape[1]

    back = np.ones(size, dtype=np.uint8) * 255
    hh, ww = back.shape

    # print(f'Shape : {size}')
    # print(f'h, w = {h}, {w}')
    # print(f'hh, ww = {hh}, {ww}')

    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((hh - h) / 2)
    xoff = round((ww - w) / 2)
    # print(f'xoff, yoff = {xoff}, {yoff}')

    # use numpy indexing to place the resized image in the center of background image
    result = back.copy()
    result[yoff : yoff + h, xoff : xoff + w] = img

    return result


# 2550px W x 3300
# 128 * 26 -> 3328
# 128 * 20 -> 2560
# 2560 x 3328  WxH


def read_image(image):
    """Read image and convert to OpenCV compatible format"""
    img = None
    if type(image) == str:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR
            img = image
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            img = image[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif type(image) == PIL.Image.Image:  # convert pil to OpenCV
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        raise Exception(f"Unhandled image type : {type(image)}")

    return img


def augment_image(img):
    import random
    import string

    """Augment imag and mask"""
    import imgaug as ia
    import imgaug.augmenters as iaa

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq_shared = iaa.Sequential(
        [
            #  sometimes(iaa.Affine(
            #      scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
            #      shear=(-6, 6),
            #      cval=(0, 0), # Black
            # )),
            # In some images distort local areas with varying strength.
            # sometimes(iaa.PiecewiseAffine(scale=(0.005, 0.01))),
            iaa.PiecewiseAffine(scale=(0.005, 0.01)),
            # iaa.CropAndPad(
            #     percent=(-0.07, 0.2),
            #     # pad_mode=ia.ALL,
            #     # pad_mode=["edge"],
            #     pad_mode=["constant", "edge"],
            #     pad_cval=(0, 0)
            # ),
        ]
    )

    seq = iaa.Sequential(
        [
            # sometimes(iaa.Dropout((0.0, 0.002))),
            iaa.SaltAndPepper(0.001, per_channel=False),
            # Blur each image with varying strength using
            # gaussian blur (sigma between 0 and 3.0),
            # average/uniform blur (kernel size between 2x2 and 7x7)
            # median blur (kernel size between 1x1 and 5x5).
            sometimes(
                iaa.OneOf(
                    [
                        iaa.GaussianBlur((0, 2.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(1, 3)),
                    ]
                )
            ),
            sometimes(iaa.ElasticTransformation(alpha=(0.5, 1.2), sigma=0.25)),
        ],
        random_order=True,
    )

    seq_shared_det = seq_shared.to_deterministic()
    image_aug = seq_shared_det(image=img)

    return image_aug


def get_patches():
    patches = []
    # Resolution widths
    # 128*20 is the base
    resolutions = [128 * 20, 128 * 19.5, 128 * 19, 128 * 18, 128 * 18.5, 128 * 17.0]

    resolutions = [128 * 16]

    # rescale image to height of 1000
    def __rescale_height(img, new_height=1000):
        size = img.shape[:2]
        oh, ow = size
        ratio = oh / ow
        # new_height = 1000
        new_width = int(ratio * new_height)

        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    for filename in os.listdir(patch_dir):
        try:
            img_path = os.path.join(patch_dir, filename)
            src_img = read_image(img_path)

            if len(src_img.shape) == 2:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)

            for k in range(len(resolutions) * 1):
                index = random.randint(0, len(resolutions) - 1)
                res = int(resolutions[index])
                # Scale to our resolution then frame
                img = __scale_width(src_img, res)
                # img = __rescale_height(src_img, res)

                print(f'img shape : {img.shape}')

                if True:
                    h = max(img.shape[0], 3328)
                    w = max(img.shape[1], 2560)

                    h = 2660
                    w = 2048
                    img = cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)
                    # img = __frame_image(img, (h, w)) # HxW

                patches.append(img)
        except Exception as e:
            raise e

    return patches


def get_images_from_dir(asset_dir):
    assets = []

    for filename in os.listdir(asset_dir):
        try:
            img_path = os.path.join(asset_dir, filename)
            src_img = read_image(img_path)

            if len(src_img.shape) == 2:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)

            assets.append(src_img)
        except Exception as e:
            raise e

    return assets


words_list = get_word_list()
patches_list = get_patches()

logo_list = get_images_from_dir(os.path.join(asset_dir, 'logos'))
image_blocks = get_images_from_dir(os.path.join(asset_dir, 'blocks'))

print('\nnumber of words in the txt file: ', len(words_list))
print('number of patches: ', len(patches_list))
print('number of logos: ', len(logo_list))
print('number of image_blocks: ', len(image_blocks))

# scale factor
scale_h, scale_w = 1, 1

img_count = 1
word_count = 0
num_imgs = int(cfg.num_synthetic_imgs)  # max number of synthetic images to be generated
train_num = int(num_imgs * cfg.train_percentage)  # training percent
print('\nnum_imgs : ', num_imgs)
print('train_num: ', train_num)


def get_text():
    global word_count, words_list
    # text to be printed on the blank image
    num_words = np.random.randint(1, 6)

    # renew the word list in case we run out of words
    if (word_count + num_words) >= len(words_list):
        print('===\nrecycling the words_list')
        words_list = get_word_list()
        word_count = 0

    print_text = ''
    for _ in range(num_words):
        index = np.random.randint(0, len(words_list))
        print_text += str.split(words_list[index])[0] + ' '
        # print_text += str.split(words_list[word_count])[0] + ' '
        word_count += 1
    print_text = print_text.strip()  # to get rif of the last space
    return print_text


def get_phone():
    "Generate phone like string"

    letters = string.digits
    sep = np.random.choice([True, False], p=[0.5, 0.5])
    c = 10
    if sep:
        c = 3
        d = 3
        z = 4

    n = ''.join(random.choice(letters) for i in range(c))
    if sep:
        n += '-'
        n += ''.join(random.choice(letters) for i in range(d))
        n += '-'
        n += ''.join(random.choice(letters) for i in range(z))

    return n


def drawTrueTypeTextOnImage(pil_img, canvas, canvas_mask, text, xy, size, font=None):
    """
    Print True Type fonts using PIL and convert image back into OpenCV
    """

    # check if the image is PIL
    if not isinstance(pil_img, Image.Image):
        raise Exception('Image is not PIL')

    if font is None:
        # fontFace = np.random.choice([ "FreeMono.ttf", "FreeMonoBold.ttf", "oldfax.ttf", "FreeMonoBold.ttf", "FreeSans.ttf", "Old_Rubber_Stamp.ttf"])
        fontFace = np.random.choice(
            ["FreeMono.ttf", "FreeMonoBold.ttf", "FreeMonoBold.ttf", "FreeSans.ttf"]
        )
        fontFace = np.random.choice(
            [
                "FreeMono.ttf",
                "FreeMonoBold.ttf",
                "FreeSans.ttf",
                "ColourMePurple.ttf",
                "Pelkistettyatodellisuutta.ttf",
                "SpotlightTypewriterNC.ttf",
            ]
        )

        fonts = os.listdir('./assets/fonts/truetype-simple')
        fontFace = np.random.choice(fonts)
        fontPath = os.path.join("./assets/fonts/truetype-simple", fontFace)

        font = ImageFont.truetype(fontPath, size)

    (left, top, right, bottom) = canvas.textbbox((0, 0), text, font)
    size_width = right - left
    size_height = bottom - top

    # text has to be within the bounds otherwise return same image
    x = xy[0]
    y = xy[1]

    img_w = pil_img.size[0]
    img_h = pil_img.size[1]

    adj_y = y + size_height
    adj_w = x + size_width

    # print(f'size : {img_h},  {adj_y},  {size_width}, {size_height} : {xy}')
    if adj_y > img_h or adj_w > img_w:
        return False, (0, 0)

    stroke_width = 0
    stroke_fill = 'black'
    mask_fill = 'black'
    fill = 'black'

    if np.random.choice([0, 1], p=[0.9, 0.1]):
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

    if True:
        mask, offset = font.getmask2(text, stroke_width=0, stroke_fill=stroke_fill)
        bitmap_image = Image.frombytes(mask.mode, mask.size, bytes(mask))

        # bitmap_image.save(f'/tmp/masks/mask-{text}.png')
        adjust = (offset[0], offset[1])
        xy = (xy[0] + adjust[0], xy[1] + adjust[1])

        # filter non-black pixels
        # if we don't do this, the mask will anit-aliased and will have some non-black pixels
        bitmap_image = np.array(bitmap_image)
        bitmap_image[bitmap_image != 0] = 255
        bitmap_image = Image.fromarray(bitmap_image)
        canvas_mask.bitmap(xy, bitmap_image, fill=mask_fill)

    return True, (size_width, size_height)


def print_lines_aligned(pil_img, boxes):
    # check if the image is PIL
    if not isinstance(pil_img, Image.Image):
        raise Exception('Image is not PIL')

    # get PIL image size
    pil_w, pil_h = pil_img.size

    w = pil_w
    h = pil_h

    def getUpperOrLowerText(txt):
        if np.random.choice([0, 1], p=[0.4, 0.6]):
            return txt.upper()
        return txt.lower().capitalize()

    def make_txt():
        # get a line of text
        txt = get_text()

        if np.random.choice([0, 1], p=[0.8, 0.2]):
            txt = get_text()
        else:
            letters = string.digits
            c = np.random.randint(1, 9)
            txt = ''.join(random.choice(letters) for i in range(c))

        return getUpperOrLowerText(txt)

    fonts = os.listdir('./assets/fonts/truetype-simple')
    fontFace = np.random.choice(fonts)
    fontPath = os.path.join("./assets/fonts/truetype-simple", fontFace)

    # Pass the image to PIL
    # pil_im = Image.fromarray(cv2Image)

    # create new Pil image to draw on
    pil_img_mask = Image.new('RGB', (w, h), (255, 255, 255))

    canvas = ImageDraw.Draw(pil_img)
    canvas_mask = ImageDraw.Draw(pil_img_mask)
    # Determine the text start position

    xy = (np.random.randint(0, w / 12), np.random.randint(0, h / 16))

    if np.random.choice([True, False], p=[0.5, 0.5]):
        # create header
        header_h = 300

        print("w : ", w, "h : ", h)
        print("xy : ", xy)

        # get random logo from the logo_list
        logo = logo_list[np.random.randint(0, len(logo_list))]
        # opencv hxw
        logo_h, logo_w = logo.shape[0], logo.shape[1]

        # augment the logo if needed
        logo = augment_image(logo) if np.random.choice([0, 1], p=[0.5, 0.5]) else logo
        logo = Image.fromarray(logo)
        logo_w, logo_h = logo.size
        logo_pos = (
            np.random.randint(50, w / 8),
            np.random.randint(50, 50 + min(header_h - logo_h, logo_h)),
        )
        print("logo_pos : ", logo_pos, "logo_w : ", logo_w, "logo_h : ", logo_h)
        # paste the logo on the image
        pil_img.paste(logo, logo_pos)
        pil_img_mask.paste(logo, logo_pos)

        # draw text AND Logo

        company = fake.company()
        company = company[: min(len(company), 16)]

        valid, wh = drawTrueTypeTextOnImage(
            pil_img,
            canvas,
            canvas_mask,
            company,
            (logo_pos[0] + logo_w, logo_pos[1]),
            np.random.randint(100, 140),
            font=None,
        )

        start_y = logo_pos[1] + wh[1]

        # draw barcode component
        if np.random.choice([True, False], p=[0.5, 0.5]):
            # import EAN13 from barcode module
            from barcode.writer import ImageWriter
            from barcode import generate

            import io

            fp = io.BytesIO()

            generate('code128', fake.company(), writer=ImageWriter(), output=fp)
            pil_barcode = Image.open(fp)
            # center the barcode horizontally
            barcode_w, barcode_h = pil_barcode.size
            barcode_pos = (
                np.random.randint(w // 4, w // 2),
                np.random.randint(start_y + barcode_h // 8, start_y + barcode_h // 4),
            )

            print("barcode_pos : ", barcode_pos)

            pil_img.paste(pil_barcode, barcode_pos)
            pil_img_mask.paste(pil_barcode, barcode_pos)

        else:
            import qrcode

            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=np.random.randint(8, 12),
                border=np.random.randint(2, 4),
            )
            qr.add_data(fake.company())
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")
            img = img.convert('RGB')
            # ranmozize the size of the qr code
            qrs = np.random.randint(120, 200)
            img = img.resize((qrs, qrs))

            barcode_w, barcode_h = img.size
            barcode_pos = (
                np.random.randint(w // 4, w // 2),
                np.random.randint(start_y, start_y + barcode_h // 2),
            )
            pil_img.paste(img, barcode_pos)

        header_h = barcode_pos[1] + barcode_h
        print("header_h : ", header_h)

        xy = [xy[0], header_h + np.random.randint(0, 25)]

    start_x, start_y = xy[0], xy[1]

    # STAMP : genalog
    if False:
        stamp = [np.random.randint(0, len(image_blocks))]
        # opencv hxw
        stamp_h, stamp_w = stamp.shape[0], stamp.shape[1]

        # create new Pil image to draw on
        pil_stamp = Image.fromarray(stamp)
        pil_stamp_mask = Image.new('RGB', (w, h), (255, 255, 255))

        stamp_canvas = ImageDraw.Draw(pil_stamp)
        stamp_canvas_mask = ImageDraw.Draw(pil_stamp_mask)

    while True:
        trueTypeFontSize = np.random.randint(30, 100)
        font = ImageFont.truetype(fontPath, trueTypeFontSize)
        m_h = np.random.randint(trueTypeFontSize, trueTypeFontSize * 2)

        start_x = xy[0]
        while True:
            txt = make_txt()
            pos = (start_x, start_y)
            valid, wh = drawTrueTypeTextOnImage(
                pil_img, canvas, canvas_mask, txt, pos, trueTypeFontSize, font
            )
            txt_w = wh[0] + np.random.randint(60, 120)
            # print(f' {start_x}, {start_y} : {valid}  : {wh}')
            start_x = start_x + txt_w
            if wh[1] > m_h:
                m_h = wh[1]
            if start_x > w:
                break
        start_y = start_y + np.random.randint(m_h // 2, m_h * 1.5)
        if start_y > h:
            break

    box = [xy[0], xy[1], wh[0], wh[1]]
    boxes.append(box)

    # pil_img_mask.save('/tmp/masks/mask-canvas.png')
    return pil_img_mask


def write_pil_images(generated, noisy_img, index):
    img_type = ''
    print(f'Writing {index}, {train_num}')

    if index <= train_num:
        noisy_img.save(
            os.path.join(
                data_dir,
                train_dir,
                imgs_dir,
                'blk_{}.png'.format(str(index).zfill(8), img_type),
            )
        )
        generated.save(
            os.path.join(
                data_dir,
                train_dir,
                noisy_dir,
                'blk_{}.png'.format(str(index).zfill(8), img_type),
            )
        )
    else:
        noisy_img.save(
            os.path.join(
                data_dir,
                val_dir,
                imgs_dir,
                'blk_{}.png'.format(str(index).zfill(8), img_type),
            )
        )
        generated.save(
            os.path.join(
                data_dir,
                val_dir,
                noisy_dir,
                'blk_{}.png'.format(str(index).zfill(8), img_type),
            )
        )


print('\nsynthesizing image data...')
idx = 0


def __process(index):
    print(f'index : {index}')
    try:
        patch_idx = np.random.randint(0, len(patches_list))
        patch = patches_list[patch_idx]
        patch = augment_image(patch)

        h = patch.shape[0]
        w = patch.shape[1]

        boxes = []

        fonts = os.listdir('./assets/fonts/truetype-simple')
        fontFace = np.random.choice(fonts)
        fontPath = os.path.join("./assets/fonts/truetype-simple", fontFace)

        pil_img = Image.new('RGB', (w, h), (255, 255, 255))

        trueTypeFontSize = np.random.randint(30, 120)
        font = ImageFont.truetype(fontPath, trueTypeFontSize)

        pil_mask = print_lines_aligned(pil_img, boxes)
        pil_patch = Image.fromarray(patch)

        pil_img = pil_img.convert('L')
        pil_patch = pil_patch.convert('L')

        img = np.array(pil_img)
        mask = np.array(pil_mask)
        patch = np.array(pil_patch)

        if np.random.choice([True, False], p=[0.5, 0.5]):
            img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            patch = cv2.threshold(patch, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
                1
            ]

        merged = cv2.bitwise_and(patch, img, mask=None)
        merged = Image.fromarray(merged)

        if False:
            merged.save(f"/tmp/masks/merged-{index}.png")
            pil_img.save(f"/tmp/masks/final-{index}.png")
            pil_mask.save(f"/tmp/masks/mask-{index}.png")
            pil_patch.save(f"/tmp/masks/patch-{index}.png")

        write_pil_images(merged, pil_mask, index)
    except Exception as e:
        # print full statcktrace
        import traceback

        traceback.print_exc()
        print(e)


def main():
    # fireup new threads for processing
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for i in range(0, num_imgs):
            executor.submit(__process, i)

    print('All tasks has been finished')


def check_mask_images():
    import numpy as np

    import cv2

    kernel = np.ones((2, 2), np.uint8)

    # load image
    img = cv2.imread("/training_data/images/152630207_0.png")

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of black color in HSV
    lower_val = np.array([0, 0, 0])
    upper_val = np.array([179, 100, 130])

    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_val, upper_val)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)

    # invert the mask to get black letters on white background
    res2 = cv2.bitwise_not(mask)

    # display image

    # cv2.imshow("img", res)
    cv2.imshow("img2", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # check_mask_images()
    main()
