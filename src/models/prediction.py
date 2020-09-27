from pydantic import BaseModel


class TitanicPredictionResult(BaseModel):
    survived: bool
