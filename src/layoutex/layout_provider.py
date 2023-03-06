import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from PIL.ImageFont import ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from layoutex.content import Content, ContentType, ParagraphContent
from layoutex.layout_transformer.dataset import JSONLayout
from layoutex.layout_transformer.model import GPTConfig, GPT
import torch
from torch.nn import functional as F

from PIL import Image, ImageDraw, ImageOps

from layoutex.layout_transformer.utils import trim_tokens, gen_colors


def get_layout_provider(name: str, max_objects: int, max_length: int):
    if name == "fixed":
        return FixedLayoutProvider(max_objects, max_length)
    elif name == "generated":
        return GeneratedLayoutProvider(max_objects, max_length)
    else:
        raise ValueError(f"Unknown layout provider: {name}")


class LayoutProvider(ABC):
    """
    Layout provider base class
    """

    def __init__(self, max_objects: int, max_length: int):
        self.max_length = max_length
        self.max_objects = max_objects

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the layout provider
        """
        ...

    @abstractmethod
    def get_layouts(
        self,
        document_count: int,
        solidity: float = 0.5,
        expected_components: Optional[list[str]] = None,
    ) -> list[Content]:
        """
        Get the layout of the document  to be generated
        Args:
            document_count:  The number of documents to be generated
            solidity: The solidity of the document to be generated in relation to the expected components
            expected_components: The components that are expected to be in the document
        Returns
            The layout of the document to be generated as a list of Content objects
        """
        ...


class GeneratedLayoutProvider(LayoutProvider):
    def __init__(self, max_objects: int, max_length: int):
        super().__init__(max_objects, max_length)

        # load model and use it for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load(
            "/home/greg/dev/marieai/layoutex/src/layoutex/layout_transformer/logs/publaynet/checkpoints/publaynet.pth",
            self.device,
        )
        train_json = "~/datasets/publaynet/annotations/val.json"
        self.dataset = JSONLayout(os.path.expanduser(train_json))

    @property
    def name(self) -> str:
        """
        Get the name of the layout provider
        Returns:
            The name of the layout provider
        """
        return "generated"

    def get_layouts(self, document_count: int) -> list[Content]:
        model = self.model
        dataset = self.dataset
        loader = DataLoader(
            dataset, shuffle=True, pin_memory=True, batch_size=1, num_workers=1
        )

        samples_dir = "/home/greg/dev/marieai/layoutex/src/layoutex/layout_transformer/logs/publaynet/samples"
        samples_dir = "/tmp/samples"
        gen_name = 0
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (x, y) in pbar:
            gen_name += 1
            fixed_x = x[: min(4, len(x))]
            fixed_y = y[: min(4, len(y))]
            # inputs
            layouts = fixed_x.detach().cpu().numpy()
            input_layouts = [dataset.render(layout) for layout in layouts]

            if True:
                for i, layout in enumerate(layouts):
                    layout = dataset.render(layout)
                    layout.save(
                        os.path.join(samples_dir, f"{gen_name:02d}_{i:02d}_input.png")
                    )

            # reconstruction
            x_cond = fixed_x.to(self.device)
            logits, _ = model(x_cond)
            probs = F.softmax(logits, dim=-1)
            _, y = torch.topk(probs, k=1, dim=-1)
            layouts = (
                torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
            )
            recon_layouts = [dataset.render(layout) for layout in layouts]

            if True:
                for i, layout in enumerate(layouts):
                    layout = dataset.render(layout)
                    layout.save(
                        os.path.join(samples_dir, f"{gen_name:02d}_{i:02d}_recon.png")
                    )

            # for i, layout in enumerate(layouts):
            #     layout = self.train_dataset.render(layout)
            #     layout.save(os.path.join(self.config.samples_dir, f'recon_{epoch:02d}_{i:02d}.png'))

    def load(self, model_path: str, device: str) -> object:
        # {1: {'supercategory': '', 'id': 1, 'name': 'text'}, 2: {'supercategory': '', 'id': 2, 'name': 'title'},
        #  3: {'supercategory': '', 'id': 3, 'name': 'list'}, 4: {'supercategory': '', 'id': 4, 'name': 'table'},
        #  5: {'supercategory': '', 'id': 5, 'name': 'figure'}}

        mconf = GPTConfig(
            264,
            517,
            n_layer=6,
            n_head=8,
            n_embd=512,
        )  # a GPT-1

        # model = GPT(mconf)
        model = GPT(mconf).to(device)
        # load checkpoint from pth file
        model.load_state_dict(torch.load(model_path))
        print("Loaded model from {}".format(model_path))
        return model


def normalize_bbox_1000(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


#


def estimate_component_sizing(box, target_size, margin_size):
    """
    estimate the component sizing based on the bounding box size and the target size of the document including margins
    """
    x1, y1, x2, y2 = box
    target_size = target_size - 2 * margin_size
    w = int(x2 - x1)
    h = y2 - y1
    ratio_w = w / target_size
    ratio_h = h / target_size

    component_w = "FULL_WIDTH"
    component_h = "FULL_HEIGHT"

    if ratio_w > 0.75:
        component_w = "FULL_WIDTH"
    elif ratio_w > 0.5:
        component_w = "TWO_THIRDS_WIDTH"
    elif ratio_w > 0.25:
        component_w = "HALF_WIDTH"
    elif ratio_w > 0.01:
        component_w = "QUARTER_WIDTH"

    if ratio_h > 0.75:
        component_h = "FULL_HEIGHT"
    elif ratio_h > 0.25:
        component_h = "HALF_HEIGHT"
    elif ratio_h > 0.05:
        component_h = "QUARTER_HEIGHT"
    elif ratio_h > 0.01:
        component_h = "LINE_HEIGHT"

    return component_w, component_h


class FixedLayoutProvider(LayoutProvider):
    def __init__(self, max_objects: int, max_length: int):
        super().__init__(max_objects, max_length)
        train_json = "~/datasets/publaynet/annotations/val.json"
        self.dataset = JSONLayout(os.path.expanduser(train_json))

    @property
    def name(self) -> str:
        """
        Get the name of the layout provider
        Returns:
            The name of the layout provider
        """
        return "fixed"

    def get_layouts(
        self,
        document_count: int,
        solidity: float = 0.5,
        expected_components: Optional[list[str]] = None,
    ) -> list[list[dict]]:
        if expected_components is None:
            expected_components = ["table"]

        dataset = self.dataset
        loader = DataLoader(
            dataset, shuffle=True, pin_memory=True, batch_size=1, num_workers=1
        )

        colors = gen_colors(6)  # category_colors
        samples_dir = "/tmp/samples"
        documents = []
        idx = 0
        pbar = tqdm(enumerate(loader), total=len(loader))

        for it, (x, y) in pbar:
            if idx >= document_count:
                break

            fixed_x = x[: min(4, len(x))]
            fixed_y = y[: min(4, len(y))]

            layouts = fixed_x.detach().cpu().numpy()
            target_size = 1024
            generated_layouts = []
            layout = layouts[0]

            normalized_layout = dataset.normalize_layout(
                layout, target_size=target_size
            )

            img = Image.new('RGB', (target_size, target_size), color=(255, 255, 255))
            draw = ImageDraw.Draw(img, 'RGBA')

            has_table = False
            has_figure = False
            has_list = False

            target_area = target_size * target_size
            layout_area = 0
            for normalized in normalized_layout:
                # print(normalized)
                cat = normalized[0]
                box = normalized[1:]
                col = colors[cat]
                x1, y1, x2, y2 = box

                print(cat, box)
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=tuple(col) + (200,),
                    fill=tuple(col) + (64,),
                    width=2,
                )

                # get area of the box
                area = (x2 - x1) * (y2 - y1)
                layout_area += area

                # 0  {1: {'supercategory': '', 'id': 1, 'name': 'text'},
                # 1  2: {'supercategory': '', 'id': 2, 'name': 'title'},
                # 2  3: {'supercategory': '', 'id': 3, 'name': 'list'},
                # 3  4: {'supercategory': '', 'id': 4, 'name': 'table'},
                # 4  5: {'supercategory': '', 'id': 5, 'name': 'figure'}}

                component_sizing = estimate_component_sizing(box, target_size, 60)

                # convert category to Content Type
                if cat == 0:
                    content_type = ContentType.PARAGRAPH
                elif cat == 1:
                    content_type = ContentType.TITLE
                elif cat == 2:
                    content_type = ContentType.LIST
                    has_list = True
                elif cat == 3:
                    content_type = ContentType.TABLE
                    has_table = True
                elif cat == 4:
                    content_type = ContentType.FIGURE
                    has_figure = True
                else:
                    raise ValueError(f"Unknown category {cat}")

                info = {
                    "content_type": content_type,
                    "bbox": box,
                    "sizing": component_sizing,
                }

                draw.text(
                    (x1, y1),
                    f"{content_type} {component_sizing}",
                    fill=(0, 0, 0, 255),
                    # font=ImageFont.truetype("arial.ttf", 20),
                )

                generated_layouts.append(info)

            if expected_components is not None:
                if "table" in expected_components and not has_table:
                    continue
                if "figure" in expected_components and not has_figure:
                    continue
                if "list" in expected_components and not has_list:
                    continue

            solid_area = layout_area / target_area
            if solid_area <= solidity:
                continue

            # for i, layout in enumerate(layouts):
            # rendered = dataset.render(layout)
            # rendered.save(os.path.join(samples_dir, f"{idx:02d}_{0:02d}_input.png"))
            img.save(os.path.join(samples_dir, f"{idx:02d}_{0:02d}_rescaled.png"))
            idx += 1
            documents.append(generated_layouts)

        return documents
