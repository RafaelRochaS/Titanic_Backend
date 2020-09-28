from starlette.config import Config

APP_VERSION = "0.0.1"
APP_NAME = "Titanic Survivors Prediction"
API_PREFIX = "/api"

config = Config(".env")

ENV = config("ENVIRONMENT")
