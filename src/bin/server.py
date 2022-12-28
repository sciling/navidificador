# pylint: disable=too-few-public-methods
import base64
import io
import json
import math
import os
import re
import sys

from hashlib import sha256
from inspect import cleandoc
from typing import List
from typing import Optional

import httpx
import magic
import numpy as np
import openai

from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.logger import logger
from fastapi.staticfiles import StaticFiles
from PIL import ExifTags
from PIL import Image
from pydantic import BaseModel  # pylint: disable=no-name-in-module

from navidificador import logging  # pylint: disable=unused-import # noqa: F401
from navidificador.fastapi import AppException
from navidificador.fastapi import UserException
from navidificador.fastapi import app
from navidificador.fastapi import responses
from navidificador.profiler import get_profiling_data
from navidificador.profiler import profile


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logger.debug("Python %s", sys.version.replace("\n", " "))


def create_sha256(data):
    sha = sha256()
    sha.update(data)
    return sha.hexdigest()


def clean_spaces(text):
    text = re.sub(r"^\s*", "", text)
    text = re.sub(r"\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


campaigns = {
    "navidad": {
        "inpaint": {
            "inputs": "inpaint",
            "prompt": clean_spaces(
                """
                christmas hat, scene from a christmas story, christmas card, thomas kinkade, deviantart, cinematic,
                snowing, 8k, Christmas lights, Christmas colors muted, snow, Christmas tale, santa, Editorial Photography,
                Highly detailed photorealistic, christmas aesthetic, a christmas eve photorealistic painting on the wall,
                canon eos c 3 0 0, ƒ 1. 8, 3 5 mm, no blur
            """
            ),
            "negative-prompt": clean_spaces(
                """
                duplicate, fog, darkness, grain, disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry,
                bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb,
                blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly,
                disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, bad artist
            """
            ),
            "num-samples": 4,
            "strength": 0.40,
            "guidance-scale": 15,
            "inference-steps": 100,
        },
        "poem_prompt": {
            "en": cleandoc(
                """
                Taking into account that Christmas is important, generate a Christmas story in the form of a
                Shakespeare sonnet with maximum four strophes, without title, and with good rhyme, with the following summary:

                {poem.description}
            """
            ),
            "es": cleandoc(
                """
                Teniendo en cuenta que la Navidad es importante, genera un cuento de Navidad en forma de soneto de Machado,
                sin título, con máximo cuatro estrofas y con buena rima y con el siguiente resumen:

                {poem.description}
            """
            ),
        },
    },
    "newyear": {
        "inpaint": {
            "inputs": "inpaint",
            "prompt": clean_spaces(
                """
                confetti, golden shimmer, new year, new year’s eve, deviantart, cinematic,snowing, 8k,
                Editorial Photography, Highly detailed photorealistic, christmas aesthetic,
                a new year's eve photorealistic fireworks on the background, canon eos c 3 0 0, ƒ 1. 8, 3 5 mm, no blur
            """
            ),
            "negative-prompt": clean_spaces(
                """
                duplicate, fog, darkness, grain, disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry,
                bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb,
                blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly,
                disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, bad artist
            """
            ),
            "num-samples": 4,
            "strength": 0.40,
            "guidance-scale": 15,
            "inference-steps": 100,
        },
        "poem_prompt": {
            "en": cleandoc(
                """
                To wish happy new year, write a story in the form of a Shakespeare sonnet,
                cheerful and festive, with the following summary:

                {poem.description}
            """
            ),
            "es": cleandoc(
                """
                Para felicitar el año nuevo, escribe un cuento en forma de soneto de Machado,
                alegre y festivo, con el siguiente resumen:

                {poem.description}
            """
            ),
        },
    },
    "cumpleaños": {
        "inpaint": {
            "inputs": "inpaint",
            "prompt": clean_spaces(
                """
                epic, scene from a superhero movie, deviantart, cinematic,
                snowing, 8k, Christmas lights, Christmas colors muted, snow, Christmas tale, santa, Editorial Photography,
                Highly detailed photorealistic, christmas aesthetic, a christmas eve photorealistic painting on the wall,
                canon eos c 3 0 0, ƒ 1. 8, 3 5 mm, no blur
            """
            ),
            "negative-prompt": clean_spaces(
                """
                duplicate, fog, darkness, grain, disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry,
                bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb,
                blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly,
                disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, bad artist
            """
            ),
            "num-samples": 4,
            "strength": 0.40,
            "guidance-scale": 15,
            "inference-steps": 100,
        },
        "poem_prompt": {
            "en": cleandoc(
                """
                Taking into account that it is this person's birthday, generate a friendship story from the group of friends.
                Congratulating the birthday in the form of a sonnet by Shakespeare with good rhyme and with the following summary:

                {poem.description}
            """
            ),
            "es": cleandoc(
                """
                Teniendo en cuenta que es el cumpleaños de esta persona, genera un cuento de amistad del grupo de amigos
                felicitando al cumpleañero en forma de soneto de Machado con buena rima y con el siguiente resumen:

                {poem.description}
            """
            ),
        },
    },
}


@profile(desc=0)
async def api(service, data, dumpname=None, **kwargs):
    if service.startswith("https://"):
        url = service
    else:
        service = service.upper()
        url = os.getenv(service + "_HUGGINGFACE_ENDPOINT")

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
        logger.debug(f"JSON: {data.keys()}")
        params["json"] = data

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, **params)

    if response.status_code >= 400:
        logger.error(f"RESPONSE ERROR[{service}][{response.status_code}]: {response.json()}")

    if dumpname:
        with open(dumpname, "w", encoding="utf-8") as file:
            json.dump({"response": str(response), "body": response.json()}, file, indent=2, ensure_ascii=False)

    return response.json()


def get_filename(filename):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(base_dir, "..", filename)
    return filename


def read_b64(filename, ensure_ascii=False):
    filename = get_filename(filename)
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
        Image.new("RGB", (img.size)).convert("L").save(image_bytes, "PNG")
        return image_bytes.getvalue()


def image_to_mask(image):
    return np.array(Image.open(io.BytesIO(image)).convert("1"))


KEEP_MASK = {
    "person",
    "individual",
    "someone",
    "somebody",
    "mortal",
    "soul",
    "animal",
    "animate",
    "being",
    "beast",
    "brute",
    "creature",
    "fauna",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


async def get_mask(image, basename=None):
    logger.debug("Invoking the masker...")
    dumpname = f"{basename}/mask-request.json" if basename else None
    masks = await api("mask", image, dumpname=dumpname)
    logger.debug(f"The masker found {len(masks)} masks: { {im['label'] for im in masks} }")

    mask = None
    try:
        pil_masks = [image_to_mask(base64_to_image(im["mask"])) for im in masks if im["label"] in KEEP_MASK]
        if len(pil_masks):
            mask = pil_masks[0]
            for pil in pil_masks[1:]:
                mask += pil

            image_bytes = io.BytesIO()
            Image.fromarray(mask).convert("L").save(image_bytes, "PNG")
            mask = image_bytes.getvalue()

    except Exception:  # pylint: disable=broad-except
        logger.exception("No person mask found")
        mask = None

    if mask is None:
        mask = create_full_mask(image)

    if mask is None:
        raise HTTPException(status_code=500, detail="Could not generate a mask image")

    return mask


async def get_inpaint(image, mask, campaign, seed=5464587, basename=None):
    if "inpaint" not in campaigns[campaign]:
        raise AppException(msg=f"Campaign '{campaign}' does not have inpaint configuration.")

    data = {
        "inputs": "inpaint",
        "image": image_to_base64(image, ensure_ascii=True),
        "mask": image_to_base64(mask, ensure_ascii=True),
        "seed": seed,
    }
    data.update(campaigns[campaign]["inpaint"])
    dumpname = f"{basename}/inpaint-request.json" if basename else None
    output = await api("inpaint", data, dumpname=dumpname)
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
            validation_error(target, f"Width is too big {img.size[0]} > {width[1]}.")

        if img.size[1] < height[0]:
            validation_error(target, f"Height is too small {img.size[1]} < {height[0]}.")

        if img.size[1] > height[1]:
            validation_error(target, f"Height is too big {img.size[1]} < {height[1]}.")

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
    campaign: str = "navidad"
    debug_campaign: Optional[str] = None
    seed: int = 5464587

    class Config:
        schema_extra = {
            "example": {
                "image": read_b64("resources/vader.jpg"),
                "campaign": "navidad",
                "seed": 5464587,
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


def resize(image, max_size, mode=None):
    with Image.open(io.BytesIO(image)) as img:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                exif = img._getexif()  # pylint: disable=protected-access
                if not exif:
                    break

                exif = dict(exif.items())
                if orientation not in exif:
                    break
                orient = exif[orientation]

                if not isinstance(orient, int):
                    break

                if orient == 3:
                    img = img.rotate(180, expand=True)
                elif orient == 6:
                    img = img.rotate(270, expand=True)
                elif orient == 8:
                    img = img.rotate(90, expand=True)

                break

        if mode:
            img = img.convert(mode)

        image_bytes = io.BytesIO()

        sorted_size = sorted(img.size)
        scaled_side_size = max(math.ceil(max_size * sorted_size[1] / sorted_size[0]), max_size)
        img.thumbnail((scaled_side_size, scaled_side_size), Image.Resampling.LANCZOS)
        img.save(image_bytes, "JPEG")
        return image_bytes.getvalue()


@profile()
async def process_image(image: ImageModel):
    campaign = image.campaign
    if image.debug_campaign:
        campaign = image.debug_campaign
    logger.debug(f"SET CAMPAIGN: {campaign} (debug: {bool(image.debug_campaign)})")

    if campaign not in campaigns:
        raise UserException(msg=f"Invalid campaign '{campaign}'", target="campaign")

    img = base64_to_image(image.image)

    sha = create_sha256(img)
    basename = f"tmp/{sha}"
    logger.info(f"IMAGE SHA: {basename}")
    os.makedirs(basename, exist_ok=True)

    with open(f"{basename}/orig", "wb") as file:
        file.write(img)

    img = resize(img, 512, mode="RGB")
    with open(f"{basename}/thumb.jpg", "wb") as file:
        file.write(img)
    validate_image_format(img, "thumb image")

    mask = await get_mask(img, basename=basename)
    with open(f"{basename}/mask.jpg", "wb") as file:
        file.write(mask)
    validate_image_format(mask, "mask image")

    images = await get_inpaint(img, mask, campaign, seed=image.seed, basename=basename)
    if "error" in images:
        raise AppException(msg=images["error"])

    logger.debug(f"IMAGES: {images.keys()}")

    images = images.pop("images", images)

    for index, item in enumerate(images):
        validate_image_format(base64_to_image(item["image"]), f"result image[{index}]")
        with open(f"{basename}/image{index}.jpg", "wb") as file:
            file.write(base64.b64decode(item["image"]))

    return images


@app.post("/image", response_model=ImageResponseModel, responses=responses)
@profile()
async def process_image_json(image: ImageModel):
    return ImageResponseModel(images=await process_image(image))


@app.post("/image-file", response_model=ImageResponseModel, responses=responses)
@profile()
async def process_image_file(image_file: UploadFile):
    image = ImageModel(image=image_to_base64(image_file))
    return ImageResponseModel(images=await process_image(image))


class PoemModel(BaseModel):
    description: str
    language: Optional[str] = "es"
    campaign: str = "navidad"
    debug_campaign: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "campaign": "navidad",
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


class PoemResponseModel(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "the text of the poem",
            }
        }


def limit_poem(text: str, max_lines: int = 16):
    lines = text.split("\n")
    res = []
    n_lines = 0
    n_latest_block = 0
    for line in lines:
        if n_lines >= max_lines:
            break

        if len(line) > 0:
            n_lines += 1
            n_latest_block += 1
        elif len(res) > 0 and n_latest_block <= 1:
            res.pop()
            n_lines -= 1
            n_latest_block = 0

        res.append(line)

    limited = "\n".join(res)
    limited = re.sub(r"^\n*", "", limited)
    limited = re.sub(r"\n*$", "", limited)
    return limited


@app.post("/poem", response_model=PoemResponseModel, responses=responses)
@profile()
async def create_poem(poem: PoemModel):
    """Produces a Christmas poem addressed to a specific person.
    Description should have some details of that person so that the poem can be personalized.
    """
    campaign = poem.campaign
    if poem.debug_campaign:
        campaign = poem.debug_campaign
    logger.debug(f"SET CAMPAIGN: {campaign} (debug: {bool(poem.debug_campaign)})")

    if campaign not in campaigns:
        raise UserException(msg=f"Invalid campaign '{campaign}'", target="campaign")

    campaign_data = campaigns[campaign]

    if poem.language not in campaign_data["poem_prompt"]:
        raise UserException(msg=f"Invalid language '{poem.language}' for campaign '{campaign}'", target="language")

    prompt = campaign_data["poem_prompt"][poem.language].format(**globals(), **locals())

    response = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, temperature=0.7, max_tokens=512, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, best_of=1
    )

    text = response["choices"][0]["text"]
    text = limit_poem(text)
    logger.debug(f"openai response: {text}")
    return PoemResponseModel(text=text)


app.mount("/", StaticFiles(directory=get_filename("templates"), html=True), name="templates")


def start():
    """Launched with `poetry run start` at root level"""
    import uvicorn  # pylint: disable=import-outside-toplevel

    uvicorn.run("bin.server:app", host="0.0.0.0", port=8000, reload=True)
