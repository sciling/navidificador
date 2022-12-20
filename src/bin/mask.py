#! /usr/bin/env python

import json
import math
import os
import random
import sys
import time

from hashlib import sha256
from io import BytesIO

from fastapi.logger import logger
from PIL import Image

from bin.server import api
from bin.server import base64_to_image


def create_sha256(data):
    sha = sha256()
    sha.update(data)
    return sha.hexdigest()


def img_to_bytes(img):
    image_bytes = BytesIO()
    img.save(image_bytes, "PNG")
    return image_bytes.getvalue()


def main():
    _, image_fn = sys.argv
    with open(image_fn, "rb") as file:
        img_bytes = file.read()

    run = create_sha256(str(random.random()).encode("ascii"))

    basename = f"tmp/{create_sha256(img_bytes)}/{run}"
    os.makedirs(basename, exist_ok=True)

    img = Image.open(BytesIO(img_bytes))
    img.save(f"{basename}/orig.jpg", exif=run.encode("ascii"))

    sorted_size = sorted(img.size)
    scaled_side_size = max(math.ceil(512 * sorted_size[1] / sorted_size[0]), 512)

    thumb = Image.open(BytesIO(img_bytes))
    thumb.thumbnail((scaled_side_size, scaled_side_size), Image.Resampling.LANCZOS)
    thumb.save(f"{basename}/thumb.jpg", exif=run.encode("ascii"))

    urls = {
        "tiny": "https://api-inference.huggingface.co/models/facebook/maskformer-swin-tiny-ade",
        "small": "https://api-inference.huggingface.co/models/facebook/maskformer-swin-small-ade",
        "base": "https://api-inference.huggingface.co/models/facebook/maskformer-swin-base-ade",
        "large": "https://api-inference.huggingface.co/models/facebook/maskformer-swin-large-ade",
    }

    summary = {"orig": {}, "thumb": {}}

    for size, sdata in summary.items():
        with open(f"{basename}/{size}.jpg", "rb") as file:
            data = file.read()

        for key, url in urls.items():
            start_time = time.time()
            res = api(url, data)
            elapsed_time = time.time() - start_time
            sdata[key] = elapsed_time
            labels = [k["label"] for k in res if "label" in k] if res else None
            for mask in res:
                if "mask" in mask:
                    img = Image.open(BytesIO(base64_to_image(mask["mask"])))
                    img.save(f"{basename}/{size}-{key}-mask-{mask['label']}.jpg", exif=run.encode("ascii"))

            logger.debug(f"RES: ({elapsed_time})s: {size}: {labels}: {key}")

    with open(f"{basename}/data.json", "w") as outfile:
        json.dump(summary, outfile, indent=2, ensure_ascii=False)

    logger.debug(basename)


if __name__ == "__main__":
    main()
