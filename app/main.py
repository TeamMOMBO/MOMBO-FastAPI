from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.models import load_fasttext_model
from app.utils import correct_ingredient

# FastText 모델 로드
model = load_fasttext_model()

# FastAPI 초기화
app = FastAPI()


# 입력 데이터 형식 정의
class OCRResult(BaseModel):
    ingredients: List[str]


# OCR 결과를 교정하는 API 엔드포인트
@app.post("/correct_ingredients/")
async def correct_ingredients(ocr_result: OCRResult):
    corrected_ingredients = []

    for ingredient in ocr_result.ingredients:
        corrected = correct_ingredient(model, ingredient)
        corrected_ingredients.append(corrected)

    return {"corrected_ingredients": corrected_ingredients}