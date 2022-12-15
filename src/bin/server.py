import base64
import io
import os
import re
import sys

from typing import List
from typing import Optional

import magic
import openai
import requests

from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.logger import logger
from PIL import Image
from pydantic import BaseModel  # pylint: disable=no-name-in-module

from navidificador import logging  # pylint: disable=unused-import # noqa: F401
from navidificador.fastapi import UserException
from navidificador.fastapi import app
from navidificador.fastapi import responses
from navidificador.profiler import get_profiling_data
from navidificador.profiler import profile


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logger.debug("Python %s", sys.version.replace("\n", " "))


@profile(desc=0)
def api(service, data, **kwargs):
    service = service.upper()

    params = {
        "timeout": 60 * 2,
    }

    params.update(kwargs)

    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_BEARER')}",
    }

    if isinstance(data, (bytes, bytearray)):
        if not kwargs.get("mime", None):
            mime = magic.from_buffer(data, mime=True)
        params["data"] = data
        headers["Content-Type"] = mime
    else:
        params["json"] = data

    response = requests.post(os.getenv(service + "_HUGGINGFACE_ENDPOINT"), headers=headers, **params)

    return response.json()


def clean_spaces(text):
    text = re.sub(r"^\s*", "", text)
    text = re.sub(r"\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def read_b64(filename, ensure_ascii=False):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(base_dir, "..", filename)
    with open(filename, "rb") as file:
        return image_to_base64(file.read(), ensure_ascii=ensure_ascii)


def image_to_base64(image, ensure_ascii=False):
    b64 = base64.b64encode(image)
    if ensure_ascii:
        return b64.decode("ascii")
    return b64


def base64_to_image(image):
    return base64.b64decode(image)


def create_full_mask(image):
    with Image.open(io.BytesIO(image)) as img:
        image_bytes = io.BytesIO()
        Image.new("RGB", (img.size)).save(image_bytes, "PNG")
        return image_bytes.getvalue()


def get_mask(image):
    logger.debug("Invoking the masker...")
    masks = api("mask", image)
    logger.debug(f"The masker found {len(masks)} masks")

    try:
        mask = next((base64_to_image(im["mask"]) for im in masks if im["label"] == "person"), None)
    except Exception:  # pylint: disable=broad-except
        logger.exception()
        mask = None

    if mask is None:
        mask = create_full_mask(image)

    if mask is None:
        raise HTTPException(status_code=500, detail="Could not generate a mask image")

    return mask


def get_inpaint(image, mask):
    data = {
        "inputs": "inpaint",
        "image": image_to_base64(image, ensure_ascii=True),
        "mask": image_to_base64(mask, ensure_ascii=True),
    }
    output = api("inpaint", data)
    return output


def validation_error(target, message, data=None):
    raise UserException(status_code=400, target=target, msg=message)


@profile()
def validate_image_format(image, target, width=(512, 2048), height=(512, 2048), mimes=("image/jpeg", "image/png")):
    mime = magic.from_buffer(image, mime=True)

    if mime not in mimes:
        validation_error(target, f"Invalid mimetype. '{mime}' not in {mimes}.")

    with Image.open(io.BytesIO(image)) as img:

        if img.size[0] < width[0]:
            validation_error(target, f"Width is too small {img.size[0]} < {width[0]}.")

        if img.size[0] > width[1]:
            validation_error(target, f"Width is too small {img.size[0]} < {width[1]}.")

        if img.size[1] < height[0]:
            validation_error(target, f"Width is too small {img.size[1]} < {height[0]}.")

        if img.size[1] > height[1]:
            validation_error(target, f"Width is too small {img.size[1]} < {height[1]}.")

    return True


@app.get("/")
async def root():
    return {"message": "Soy el Navidificador!"}


@app.get("/stats")
async def stats():
    return get_profiling_data()


class ImageModel(BaseModel):
    image: str
    prompt: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "image": read_b64("resources/vader.jpg"),
                "prompt": None,
            }
        }


class ImageResponseModel(BaseModel):
    images: List

    class Config:
        schema_extra = {
            "example": {
                "images": [read_b64("resources/vader.jpg", ensure_ascii=True)[:20] + "..."],
            }
        }


def process_image(image):
    image = base64_to_image(image.image)

    validate_image_format(image, "input image")
    mask = get_mask(image)

    validate_image_format(image, "mask image")
    images = get_inpaint(image, mask)

    for index, item in enumerate(images):
        validate_image_format(base64_to_image(item["image"]), f"result image[{index}]")

        if os.path.isdir("tmp"):
            image = Image.open(io.BytesIO(base64.b64decode(item["image"])))
            image.save(f"tmp/image{index}.jpg")

    return {
        "images": images,
    }


@app.post("/image", response_model=ImageResponseModel, responses=responses)
async def process_image_json(image: ImageModel):
    return process_image(image)


@app.post("/image-file", response_model=ImageResponseModel, responses=responses)
async def process_image_file(image_file: UploadFile):
    image = ImageModel(image=image_to_base64(image_file))
    return process_image(image)


class PoemModel(BaseModel):
    name: str
    description: str
    language: Optional[str] = "es"

    class Config:
        schema_extra = {
            "example": {
                "name": "Cristóbal Colón",
                "description": clean_spaces(
                    """
                    Cristóbal Colón (Cristoforo Colombo, en italiano, o Christophorus Columbus, en latín;
                    de orígenes discutidos, los expertos se inclinan por Génova, República de Génovan donde pudo haber
                    nacido el 31 de octubre de 1451 y se sabe que murió en Valladolid el 20 de mayo de 1506) fue un navegante,
                    cartógrafo, almirante, virrey y gobernador general de las Indias Occidentales al servicio de la Corona de
                    Castilla. Realizó el llamado descubrimiento de América el 12 de octubre de 1492,
                    al llegar a la isla de Guanahani, en las Bahamas.
                """
                ),
                "language": "es",
            }
        }


@app.post("/poem")
async def create_poem(poem: PoemModel):
    """Produces a Christmas poem addressed to a specific person.
    Description should have some details of that person so that the poem can be personalized.
    """

    if poem.language == "en":
        prompt = f"""
        Generate a Christmas story in the form of a Shakespeare sonnet with a person named {poem.name} and the following CV.
        Christmas is very important.

        {poem.description}
        """
    else:
        prompt = f"""
        Genera un cuento de Navidad en forma de soneto de Lope de Vega con una persona llamada {poem.name} y el siguiente currículum.
        La Navidad es muy importante.

        {poem.description}
        """

    response = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, temperature=0.7, max_tokens=256, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, best_of=1
    )

    text = response["choices"][0]["text"]
    return {
        "poem": text,
    }


def start():
    """Launched with `poetry run start` at root level"""
    import uvicorn  # pylint: disable=import-outside-toplevel

    uvicorn.run("bin.server:app", host="0.0.0.0", port=8000, reload=True)
