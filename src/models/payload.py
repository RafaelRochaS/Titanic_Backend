from typing import List
from pydantic import BaseModel


class TitanicPredictionPayload(BaseModel):
    passenger_id: int
    pclass: int
    sex: str
    age: int
    sibsp: int
    parch: int
    fare: float
    cabin: str
    embarked: str


def payload_to_list(tpp: TitanicPredictionPayload) -> List:
    return [
        tpp.passenger_id,
        tpp.pclass,
        tpp.sex,
        tpp.age,
        tpp.sibsp,
        tpp.parch,
        tpp.fare,
        tpp.cabin,
        tpp.embarked
    ]
