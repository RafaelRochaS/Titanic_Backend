from fastapi import FastAPI

from .api.routes.router import api_router
from .core.config import API_PREFIX, APP_NAME, APP_VERSION, ENV


def app_setup() -> FastAPI:

    debug = False
    if ENV == "dev":
        debug = True
    api = FastAPI(title=APP_NAME, version=APP_VERSION, debug=debug)
    api.include_router(api_router, prefix=API_PREFIX)

    return api


app = app_setup()
