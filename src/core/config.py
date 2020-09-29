from starlette.config import Config

APP_VERSION = "0.0.1"
APP_NAME = "Titanic Survivors Prediction"
API_PREFIX = "/api"

config = Config("src/.env")

ENV = config("ENVIRONMENT")
MODEL_PATH = config("MODEL_PATH")
