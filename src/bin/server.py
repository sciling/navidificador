import os
import io
import json
import base64
from typing import Union, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
import requests
import magic
from PIL import Image

# from navidificador import logging

load_dotenv()
app = FastAPI()


def api(service, data, mime=None, **kwargs):
    service = service.upper()

    params = {
        "timeout": 10 * 60,
    }

    params.update(kwargs)

    headers = {
        "Authorization": f"Bearer {os.getenv(service + '_HUGGINGFACE_BEARER')}",
    }

    print(f"HEADERS: {headers}, KWARGS: {kwargs}")

    if mime:
        headers['Content-Type'] = mime

    response = requests.post(os.getenv(service + '_HUGGINGFACE_ENDPOINT'), headers=headers, data=data, **params)
    return response.json()


def image_to_base64(image):
    return base64.b64encode(image)


def base64_to_image(image):
    return base64.b64decode(image)


def get_mask(image):
    mime = magic.from_buffer(image, mime=True)
    print(f"image: {mime} {image[:100]}")
    masks = api('mask', mime=mime, data=image)

    try:
        mask = next((base64_to_image(im['mask']) for im in masks if im['label'] == 'person'), None)
    except Exception as e:
        print(e)
        mask = None

    if mask is None:
        print('WARNING: no mask found!')
        with Image.open(io.BytesIO(image)) as im:
            image_bytes = io.BytesIO()
            Image.new('RGB', (im.size)).save(image_bytes, 'PNG')
            mask = image_bytes.getvalue()

    with open('mask.png', 'wb') as file:
        file.write(mask)

    mask_mime = magic.from_buffer(mask, mime=True)
    print(f"mask: {mask_mime} {mask[:100]}")
    return mask


def get_inpaint(image, mask):
    data = {
        'image': image_to_base64(image),
        'mask': image_to_base64(mask),
    }
    output = api('INPAINT', data=data)
    return output


@app.get("/")
async def root():
    return {"message": "Soy el Navidificador!"}


class ImageModel(BaseModel):
    image: str
    prompt: Optional[str] = ''


def process_image(image):
    image = base64_to_image(image.image)
    mask = get_mask(image)
    images = [image_to_base64(mask)]
    return {"images": images}


@app.post("/image")
async def process_image_json(image: ImageModel):
    return process_image(image)


@app.post("/image-file")
async def process_image_file(image_file: UploadFile):
    image = ImageModel(image=image_to_base64(image_file))
    return process_image(image)


def start():
    """Launched with `poetry run start` at root level"""
    import uvicorn  # pylint: disable=import-outside-toplevel
    uvicorn.run("bin.server:app", host="0.0.0.0", port=8000, reload=True)
