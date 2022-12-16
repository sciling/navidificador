# pylint: disable=too-few-public-methods,invalid-name
import inspect
import re
import sys

from typing import Dict
from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.logger import logger
from fastapi.responses import JSONResponse
from pydantic import BaseModel  # pylint: disable=no-name-in-module

from navidificador import logging  # pylint: disable=unused-import # noqa: F401


app = FastAPI()


# From https://groups.google.com/g/dev-python/c/Xa2zqMfgxHk/m/LSeLoAkuBwAJ
# It seems to be slow. Use it only if you want to retrieve the original f-string text
class fstring(str):  # noqa: N801
    def __str__(self):
        scope = inspect.currentframe().f_back.f_globals.copy()
        scope.update(inspect.currentframe().f_back.f_locals)
        return self.format(**scope)


# From https://github.com/pydantic/pydantic/issues/1875#issuecomment-964395974
class UserError(BaseModel):
    msg: str
    status_code: int = 400
    target: Optional[str] = None
    data: Optional[Dict] = None

    class Config:
        schema_extra = {
            "example": {
                "status_code": 400,
                "msg": "Invalid size {img.size}",
                "target": "input element that provoked the error",
                "data": None,
            }
        }


# From https://github.com/pydantic/pydantic/issues/1875#issuecomment-964395974
class AppError(BaseModel):
    msg: str
    status_code: int = 500
    trace: Optional[str] = None
    data: Optional[Dict] = None

    class Config:
        schema_extra = {
            "example": {
                "status_code": 500,
                "msg": "Generated mask too small",
            }
        }


responses = {}


def create_exception(model_cls: BaseModel, name: str, get_status_code: lambda exc: 500, get_content=jsonable_encoder, default_status_code=None):
    class ModelException(Exception):
        detail: model_cls

        def __init__(self, *args, **kwargs):
            self.detail = model_cls(*args, **kwargs)

    ModelException.__qualname__ = name

    @app.exception_handler(ModelException)
    async def app_exception_handler(request: Request, exc: ModelException):  # pylint: disable=unused-argument
        logger.debug(f"Capturing <{name}>...")
        return JSONResponse(
            status_code=get_status_code(exc),
            content=get_content(exc),
        )

    logger.debug(f"MODEL: {model_cls} {hasattr(model_cls, 'status_code')}")
    if default_status_code is None and "status_code" in model_cls.__fields__:
        default_status_code = model_cls.__fields__["status_code"].default

    if default_status_code:
        responses[str(default_status_code)] = {"model": model_cls}

    return ModelException


UserException = create_exception(UserError, "UserException", get_status_code=lambda exc: exc.detail.status_code)

AppException = create_exception(AppError, "AppException", get_status_code=lambda exc: exc.detail.status_code)


@app.get("/exception")
async def generate_exception(request: Request, classname: str = None, arg: str = None):
    try:
        cls = getattr(sys.modules[__name__], classname)
    except:  # pylint: disable=bare-except # noqa: E722
        cls = Exception

    params = dict(request.query_params)

    for key in ("classname", "arg"):
        if key in params:
            del params[key]

    for key in params:
        if re.match(r"^\d+$", params[key]):
            params[key] = int(params[key])

    if arg is None:
        raise cls(**params)

    raise cls(arg, **params)


def exception_as_json_response(exc):
    data = {
        "type": exc.__class__.__name__,
    }

    if len(exc.args):
        data["args"] = exc.args

    data.update(exc.__dict__)

    if "detail" in data:
        data["msg"] = data["detail"]
        del data["detail"]

    for key, value in list(data.items()):
        if value is None:
            del data[key]

    try:
        return jsonable_encoder({"detail": data})
    except ValueError:
        logger.exception(f"jsonable_encoder couldn't parse exception '{exc}'")
        return jsonable_encoder({"detail": str(exc)})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):  # pylint: disable=unused-argument
    return JSONResponse(
        status_code=exc.status_code,
        content=exception_as_json_response(exc),
    )


async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug(f"Capturing uncaught exception: {type(exc)}")
        logger.exception("Uncaught exception")
        return JSONResponse(
            status_code=exc.status_code if hasattr(exc, "status_code") else 500,
            content=exception_as_json_response(exc),
        )


app.middleware("http")(catch_exceptions_middleware)
