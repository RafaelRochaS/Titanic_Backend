from fastapi import APIRouter

router = APIRouter()


@router.get("/predict")
def get_predict() -> dict:
    return {'status': 'down'}
