import joblib

from src.models.payload import TitanicPredictionPayload, payload_to_list
from src.core.messages import INVALID_PAYLOAD


class TitanicModelService():

    def __init__(self, path):
        self.path = path
        self.__load_model()

    def __load_model(self):
        self.model = joblib.load(self.path)

    def predict(self, payload: TitanicPredictionPayload):
        if payload is None:
            raise ValueError(INVALID_PAYLOAD.format(payload))

        prediction = self.model.predict(payload_to_list(payload))

        return prediction
