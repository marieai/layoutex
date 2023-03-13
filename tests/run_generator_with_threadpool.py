import os
from random import random

import numpy as np

from layoutex.document import Document
from layoutex.document_generator import DocumentGenerator
from layoutex.layout_provider import LayoutProvider, get_layout_provider

import asyncio
from codetiming import Timer

import concurrent.futures as cf
import multiprocessing as mp


def write_images(output_root_dir, image, mask, index, train_num):
    img_type = ""
    print(f"Writing {index}, {train_num}")

    if index <= train_num:
        data_dir = os.path.join(os.path.expanduser(output_root_dir), "train")
    else:
        data_dir = os.path.join(os.path.expanduser(output_root_dir), "test")

    image_dir = os.path.join(data_dir, "image")
    mask_dir = os.path.join(data_dir, "mask")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    mask.save(
        os.path.join(
            mask_dir,
            "blk_{}.png".format(str(index).zfill(8), img_type),
        )
    )

    image.save(
        os.path.join(
            image_dir,
            "blk_{}.png".format(str(index).zfill(8), img_type),
        )
    )


def main():
    layout_sizes = [1700, 2048, 2550]
    layout_size = 2048

    layout_provider = get_layout_provider("fixed", 10, 100)
    generator = DocumentGenerator(
        layout_provider=layout_provider,
        target_size=layout_size,
        solidity=0.5,
        expected_components=["figure", "table"],
    )

    # get cpu count
    num_samples = 100
    train_percentage = 1
    train_num = int(num_samples * train_percentage)  # training percent

    print(f"train_percentage = {train_percentage}")
    print(f"train_num: {train_num}")
    output_root_dir = "~/dev/pytorch-CycleGAN-and-pix2pix/datasets/claim_mask/src"
    # "/tmp/generated"

    def completed(future):
        print(f"Completed :  {future}")
        document = future.result()  # type: Document
        if document is None:
            return
        if not document.is_valid():
            return

        idx = document.task_id
        write_images(
            output_root_dir=output_root_dir,
            image=document.image,
            mask=document.mask,
            index=idx,
            train_num=train_num,
        )

    def batchify(iterable, n=1):
        s = len(iterable)
        for ndx in range(0, s, n):
            yield iterable[ndx : min(ndx + n, s)]

    batch_size = mp.cpu_count() * 4
    with Timer(text="\nTotal elapsed time: {:.3f}"):
        # it is important to batchify the task to avoid memory issues
        with cf.ProcessPoolExecutor(max_workers=batch_size) as executor:
            # batchify the tasks and run in parallel
            for batch in batchify(range(num_samples), batch_size):
                print(f"Batch: {batch}")
                futures = [executor.submit(generator.render, i) for i in batch]
                # futures = executor.map(generator.render, batch)
                for future in cf.as_completed(futures):
                    completed(future)


if __name__ == "__main__":
    main()
