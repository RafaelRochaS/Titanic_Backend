from fastapi import FastAPI

from .api.routes.router import api_router
from .core.config import API_PREFIX, APP_NAME, APP_VERSION, ENV
from .core.event_handlers import start_app_handler, stop_app_handler


def app_setup() -> FastAPI:

    debug = False
    if ENV == "dev":
        debug = True
    api = FastAPI(title=APP_NAME, version=APP_VERSION, debug=debug)
    api.include_router(api_router, prefix=API_PREFIX)

    api.add_event_handler("startup", start_app_handler(api))
    api.add_event_handler("shutdown", stop_app_handler(api))

    return api


app = app_setup()
