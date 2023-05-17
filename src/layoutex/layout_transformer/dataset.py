import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageOps
import json

from .utils import trim_tokens, gen_colors


class Padding(object):
    def __init__(self, max_length, vocab_size):
        self.max_length = max_length
        self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.pad_token = vocab_size - 1

    def __call__(self, layout):
        # grab a chunk of (max_length + 1) from the layout

        chunk = torch.zeros(self.max_length + 1, dtype=torch.long) + self.pad_token
        # Assume len(item) will always be <= self.max_length:
        chunk[0] = self.bos_token
        chunk[1 : len(layout) + 1] = layout
        chunk[len(layout) + 1] = self.eos_token

        x = chunk[:-1]
        y = chunk[1:]
        return {'x': x, 'y': y}


class MNISTLayout(MNIST):
    def __init__(self, root, train=True, download=True, threshold=32, max_length=None):
        super().__init__(root, train=train, download=download)
        self.vocab_size = 784 + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        self.threshold = threshold
        self.data = [self.img_to_set(img) for img in self.data]
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def __len__(self):
        return len(self.data)

    def img_to_set(self, img):
        fg_mask = img >= self.threshold
        fg_idx = fg_mask.nonzero(as_tuple=False)
        fg_idx = fg_idx[:, 0] * 28 + fg_idx[:, 1]
        return fg_idx

    def render(self, layout):
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        x_coords = layout % 28
        y_coords = layout // 28
        # valid_idx = torch.where((y_coords < 28) & (y_coords >= 0))[0]
        img = np.zeros((28, 28, 3)).astype(np.uint8)
        img[y_coords, x_coords] = 255
        return Image.fromarray(img, 'RGB')

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = self.transform(self.data[idx])
        return layout['x'], layout['y']


class JSONLayout(Dataset):
    def __init__(self, json_path, max_length=None, precision=8):
        with open(json_path, "r") as f:
            data = json.loads(f.read())

        images, annotations, categories = (
            data['images'],
            data['annotations'],
            data['categories'],
        )
        self.size = pow(2, precision)

        self.categories = {c["id"]: c for c in categories}
        self.colors = gen_colors(len(self.categories))

        self.json_category_id_to_contiguous_id = {
            v: i + self.size
            for i, v in enumerate([c["id"] for c in self.categories.values()])
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.vocab_size = self.size + len(self.categories) + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        image_to_annotations = {}
        for annotation in annotations:
            image_id = annotation["image_id"]

            if not (image_id in image_to_annotations):
                image_to_annotations[image_id] = []

            image_to_annotations[image_id].append(annotation)

        self.data = []
        for image in images:
            image_id = image["id"]
            height, width = float(image["height"]), float(image["width"])

            if image_id not in image_to_annotations:
                continue

            ann_box = []
            ann_cat = []
            for ann in image_to_annotations[image_id]:
                x, y, w, h = ann["bbox"]
                ann_box.append([x, y, w, h])
                ann_cat.append(
                    self.json_category_id_to_contiguous_id[ann["category_id"]]
                )

            # Sort boxes
            ann_box = np.array(ann_box)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            ann_box = ann_box[ind]

            ann_cat = np.array(ann_cat)
            ann_cat = ann_cat[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

            # Flatten and add to the dataset
            self.data.append(layout.reshape(-1))

        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)


    def quantize_box(self, boxes, width, height):
        # range of xy is [0, large_side-1]
        # range of wh is [1, large_side]
        # bring xywh to [0, 1]
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - 1
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / (width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / (height - 1)
        boxes = np.clip(boxes, 0, 1)

        # next take xywh to [0, size-1]
        boxes = (boxes * (self.size - 1)).round()

        return boxes.astype(np.int32)

    def __len__(self):
        return len(self.data)

    def render(self, layout):
        # create an all white RGB 256x256 pixel image 
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        # reshape layout as 1D array
        layout = layout.reshape(-1)
        # trim the beginning and end of the array, 
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        # removing any starting or ending tokens until a multiple of 5 tokens remains.
        # reshape back into a 2D array with 5 columns.
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)

        # extract box coords from layout, 
        # selecting all rows and columns starting from the second column.
        box = layout[:, 1:].astype(np.float32)

        # scale x and y coords to 255
        box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 255

        # scale w and h to 256
        box[:, [2, 3]] = box[:, [2, 3]] / self.size * 256

        # w and h are added to the x and y to get the final bbox coords.
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

        # draw rectangles on 256x256 image
        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0]
            color = (
                self.colors[cat - self.size]
                if 0 <= cat - self.size < len(self.colors)
                else [0, 0, 0]
            )
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=tuple(color) + (200,),
                fill=tuple(color) + (64,),
                width=2,
            )

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img

    def normalize_layout(self, layout, target_size):
        # reshape as 1D array
        layout = layout.reshape(-1)
        # trim the beginning and end of the array, 
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        # removing any starting or ending tokens until a multiple of 5 tokens remains.
        # reshape back into a 2D array with 5 columns.
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)

        # extract box coords from layout, 
        # selecting all rows and columns starting from the second column.
        box = layout[:, 1:].astype(np.float32)

        # normalize to fit within target_size
        # x and y (cols 0 and 1) are divded by self.size - 1 to bring them to the range [0, 1]
        # Then they are multiplied by target_size to scale them to the target size.
        box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * target_size
        # w and h (cols 2 and 3) are divded by self.size and multiplied by target_size to bring them
        # to the range of the normalized x and y coords.
        box[:, [2, 3]] = box[:, [2, 3]] / self.size * target_size
        # normalized w and h are added to the normalized x and y to get the final bbox coords.
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

        # build and return normalized layout data
        normalized = []
        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0] - self.size
            normalized.append([cat, x1, y1, x2, y2])
        return normalized

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        return layout['x'], layout['y']
