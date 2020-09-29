from fastapi import APIRouter
from starlette.requests import Request

from src.models.payload import TitanicPredictionPayload
from src.models.prediction import TitanicPredictionResult
from src.services.models import TitanicModelService

router = APIRouter()


@router.post("/predict", response_model=TitanicPredictionResult,
             name="predict")
def predict(
    request: Request,
    block_data: TitanicPredictionPayload = None
) -> TitanicPredictionResult:

    model: TitanicModelService = request.app.state.model
    prediction: TitanicPredictionResult = model.predict(block_data)

    return prediction
