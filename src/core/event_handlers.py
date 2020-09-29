from typing import Callable
from fastapi import FastAPI

from src.core.config import MODEL_PATH
from src.services.models import TitanicModelService


def _startup_model(app: FastAPI) -> None:
    model_path = MODEL_PATH
    model_instance = TitanicModelService(model_path)
    app.state.model = model_instance


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        _startup_model(app)
    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        _shutdown_model(app)
    return shutdown
