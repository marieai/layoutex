import os
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from layoutex.content import Content
from layoutex.layout_transformer.dataset import JSONLayout
from layoutex.layout_transformer.model import GPTConfig, GPT
import torch
from torch.nn import functional as F


def get_layout_provider(name: str, max_objects: int, max_length: int):
    if name == "fixed":
        return FixedLayoutProvider(max_objects, max_length)
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
    def get_layouts(self, document_count: int) -> list[Content]:
        """
        Get the layout of the document  to be generated
        Args:
            document_count:  The number of documents to be generated
        Returns
            The layout of the document to be generated as a list of Content objects
        """
        ...


class FixedLayoutProvider(LayoutProvider):
    def __init__(self, max_objects: int, max_length: int):
        super().__init__(max_objects, max_length)

        # load model and use it for inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load(
            "/home/greg/dev/marieai/layoutex/src/layoutex/layout_transformer/logs/publaynet/checkpoints/publaynet.pth",
            self.device,
        )
        train_json = "/home/greg/datasets/publaynet/annotations/val.json"
        self.dataset = JSONLayout(train_json)

    @property
    def name(self) -> str:
        """
        Get the name of the layout provider
        Returns:
            The name of the layout provider
        """
        return "fixed"

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

            print(fixed_x)
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
