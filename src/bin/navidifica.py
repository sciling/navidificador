#! /usr/bin/env python

import base64
import io
import time

from typing import Optional

import requests
import typer

from PIL import Image

from bin.server import image_to_base64
from navidificador.logging import getLogger


logger = getLogger(__name__)

# URL = "https://navidificador.sciling.com"
URL = "http://localhost:8000"


def main(image: typer.FileBinaryRead, campaign: Optional[str] = "navidad", prompt: Optional[str] = typer.Argument(None), seed: Optional[int] = 5464587):
    data = {
        "image": image_to_base64(image.read()).decode("ascii"),
        "prompt": prompt,
        "campaign": campaign,
        "seed": seed,
    }

    start_time = time.time()
    response = requests.post(f"{URL}/image", json=data, timeout=600)
    elapsed_time = time.time() - start_time

    logger.debug(f"RESPONSE: {response}")
    images = response.json()
    if response.status_code >= 400:
        raise Exception(images)
    logger.debug(f"IMAGES: {images.keys()}")

    for index, item in enumerate(images["images"]):
        image = Image.open(io.BytesIO(base64.b64decode(item["image"])))
        image.save(f"resultImages/image{index}.jpg")

    logger.debug(f"Ending profiling: ({elapsed_time}s)")


if __name__ == "__main__":
    typer.run(main)
