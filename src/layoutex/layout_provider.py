import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from layoutex.content import Content, ContentType
from layoutex.layout_transformer.dataset import JSONLayout
from layoutex.layout_transformer.model import GPTConfig, GPT
from layoutex.layout_transformer.utils import gen_colors
from layoutex.component_util import estimate_component_sizing

from typing import List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class LayoutProviderConfig:
    """
    Layout provider configuration object.
    """

    def __init__(self, type: str, max_objects: int, max_length: int, *args, **settings):
        # get the name of the layout provider
        self.type = type

        # get the max objects?
        self.max_objects = max_objects

        # get the max length?
        self.max_length = max_length
        # get the checkpoint
        default_checkpoint_path = (
            "./src/layoutex/layout_transformer/logs/publaynet/checkpoints/publaynet.pth"
        )
        self.checkpoint_path = settings.get("checkpoint_path", default_checkpoint_path)

        # get the dataset
        default_dataset_path = "./assets/datasets/publaynet/annotations/val.json"
        self.dataset_path = settings.get(
            "dataset_path", os.path.expanduser(default_dataset_path)
        )

        # get the samples directory
        default_samples_dir = "/tmp/samples"
        self.samples_dir = settings.get("samples_dir", default_samples_dir)


class LayoutProvider(ABC):
    """
    Layout provider base class
    """

    def __init__(self, layout_provider_config):
        self.max_length = layout_provider_config.max_length
        self.max_objects = layout_provider_config.max_objects
        self.config = layout_provider_config

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
        target_size: int,
        document_count: int,
        solidity: float = 0.5,
        expected_components: Optional[list] = None,
    ) -> List:  # list[Content]:
        """
        Get the layout of the document to be generated
        Args:
            target_size: The target size of the document to be generated(in pixels)
            document_count:  The number of documents to be generated
            solidity: The solidity of the document to be generated in relation to the expected components
            expected_components: The components that are expected to be in the document
        Returns
            The layout of the document to be generated as a list of Content objects
        """
        ...


class GeneratedLayoutProvider(LayoutProvider):
    def __init__(self, layout_provider_config):
        super().__init__(layout_provider_config)

        # load model and use it for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load(self.config.checkpoint_path, self.device)
        self.dataset = JSONLayout(os.path.expanduser(self.config.dataset_path))

    @property
    def name(self) -> str:
        """
        Get the name of the layout provider
        Returns:
            The name of the layout provider
        """
        return "generated"

    def get_layouts(self, document_count: int) -> List:  # list[Content]:
        model = self.model
        dataset = self.dataset
        loader = DataLoader(
            dataset, shuffle=True, pin_memory=False, batch_size=1, num_workers=1
        )

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
                        os.path.join(
                            self.config.samples_dir, f"{gen_name:02d}_{i:02d}_input.png"
                        )
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
                        os.path.join(
                            self.config.samples_dir, f"{gen_name:02d}_{i:02d}_recon.png"
                        )
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
        logger.info("Loaded model from {}".format(model_path))
        return model


class FixedLayoutProvider(LayoutProvider):
    def __init__(self, layout_provider_config):
        super().__init__(layout_provider_config)

        self.dataset = JSONLayout(os.path.expanduser(self.config.dataset_path))
        total = len(self.dataset)
        logger.info(f"total samples : {total}")

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
        target_size: int,
        document_count: int,
        solidity: float = 0.5,
        expected_components: Optional[list] = None,
    ) -> List:  # list[list[dict]]:
        logger.debug(f"get_layouts(target_size={target_size}, document_count={document_count}, solidity={solidity}, expected_components={expected_components})")
        
        if expected_components is None:
            logger.warning("No expected components...defaulting to ['table']")
            expected_components = ["table"]

        dataset = self.dataset

        colors = gen_colors(6)  # category_colors
        documents = []
        idx = 0
        # pbar = tqdm(enumerate(loader), total=len(loader))
        total = len(dataset)

        build_layout_image_for_debugging = False

        # get a random sample from the dataset
        rng = np.random.default_rng(threading.get_native_id())

        # for i in range(document_count):
        while len(documents) < document_count:
            # item = int(random.random() * total)
            # idx = np.random.randint(0, total - 1)
            idx = rng.integers(0, total - 1)

            logger.debug(f"random dataset index = {idx}")
            # idx = 9
            logger.debug(f"item: {idx}")
            (x, y) = self.dataset[idx]


            # (x, y) = self.dataset[item]
            # (x, y) = self.dataset[np.random.randint(0, total - 1)]
            fixed_x = x
            fixed_y = y
            layout = fixed_x.detach().cpu().numpy()
            fixed_y.detach().cpu().numpy()

            generated_layouts = []
            # layout = layouts[0] #only with dataloader

            normalized_layout = dataset.normalize_layout(
                layout, target_size=target_size
            )

            if build_layout_image_for_debugging:
                img = Image.new(
                    "RGB", (target_size, target_size), color=(255, 255, 255)
                )
                draw = ImageDraw.Draw(img, "RGBA")

            forced_figure = False
            added_components = set()

            target_area = target_size * target_size
            logger.debug(f"computed target_area = {target_area}")
            layout_area = 0

            for i, normalized in enumerate(normalized_layout):
                logger.debug(normalized)
                cat = normalized[0]
                box = normalized[1:]
                col = colors[cat]
                x1, y1, x2, y2 = box

                logger.debug(f"computed category    = {['PARAGRAPH', 'TITLE', 'LIST', 'TABLE', 'FIGURE'][cat]}")
                logger.debug(f"computed color       = {col}")
                logger.debug(f"bbox dimensions      = (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

                if build_layout_image_for_debugging:
                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline=tuple(col) + (200,),
                        fill=tuple(col) + (64,),
                        width=2,
                    )

                # get area of the box
                area = (x2 - x1) * (y2 - y1)
                layout_area += area
                logger.debug(f"computed area        = {area}")
                logger.debug(f"computed layout area = {layout_area}")

                # 0  {1: {'supercategory': '', 'id': 1, 'name': 'text'},
                # 1  2: {'supercategory': '', 'id': 2, 'name': 'title'},
                # 2  3: {'supercategory': '', 'id': 3, 'name': 'list'},
                # 3  4: {'supercategory': '', 'id': 4, 'name': 'table'},
                # 4  5: {'supercategory': '', 'id': 5, 'name': 'figure'}}

                component_sizing = estimate_component_sizing(box, target_size, 60)
                component_w, component_h = component_sizing
                logger.debug(f"computed component_w = ({component_w})")
                logger.debug(f"computed component_h = ({component_h})")
                # convert category to Content Type
                if cat == 0:
                    content_type = ContentType.PARAGRAPH # text
                elif cat == 1:
                    content_type = ContentType.TITLE
                elif cat == 2:
                    content_type = ContentType.LIST
                elif cat == 3:
                    content_type = ContentType.TABLE
                elif cat == 4:
                    content_type = ContentType.FIGURE
                else:
                    raise ValueError(f"Unknown category {cat}")
                
                # Force a figure if able.
                if (not forced_figure
                    and (component_w == "HALF_WIDTH" and component_h == "QUARTER_HEIGHT")):
                    logger.info(f"forcing FIGURE content for component #{i}")
                    content_type = ContentType.FIGURE
                    forced_figure = True
                
                if content_type == ContentType.TABLE:
                    if component_w == "HALF_WIDTH" and component_h == "QUARTER_HEIGHT":
                        logger.info("Changing content type from TABLE to FIGURE...")
                        content_type = ContentType.FIGURE
                        # Does this really make sense to remove ContentType.TABLE here?
                        added_components.remove(ContentType.TABLE)

                # due to how the dataset is generated, we change FIGURE to TABLE if the component is too large
                if content_type == ContentType.FIGURE:
                    if component_w == "FULL_WIDTH" and component_h in [
                        "FULL_HEIGHT",
                        "HALF_HEIGHT",
                    ]:
                        logger.info("Changing content type from FIGURE to TABLE...")
                        content_type = ContentType.TABLE

                logger.debug(f"comp. content_type   = {content_type}")
                added_components.add(content_type)

                info = {
                    "content_type": str(content_type.name).lower(),
                    "category_id": cat,
                    "bbox": box,
                    "sizing": component_sizing,
                }

                if build_layout_image_for_debugging:
                    draw.text(
                        (x1, y1),
                        f"{content_type} {component_sizing}",
                        fill=(0, 0, 0, 255),
                        # font=ImageFont.truetype("arial.ttf", 20),
                    )

                generated_layouts.append(info)

            if expected_components is not None:
                if "figure" in expected_components and ContentType.FIGURE not in added_components:
                    logger.warning("figure required but not found...skipping")
                    continue
                if "table" in expected_components and ContentType.TABLE not in added_components:
                    logger.warning("table required but not found...skipping")
                    continue
                if "list" in expected_components and ContentType.LIST not in added_components:
                    logger.warning("list required but not found...skipping")
                    continue

            solid_area = layout_area / target_area
            logger.debug(f"computed solid_area  = {solid_area}")
            if solid_area <= solidity:
                logger.warning("solid_area is below the solidity threshold...skipping")
                continue

            # rendered.save(os.path.join(self.config.samples_dir, f"{idx:02d}_{0:02d}_input.png"))
            if build_layout_image_for_debugging:
                img.save(os.path.join(self.config.samples_dir, f"{idx:02d}_{0:02d}_rescaled.png"))

            idx += 1
            documents.append(generated_layouts)

        return documents
