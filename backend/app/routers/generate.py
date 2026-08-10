from fastapi import APIRouter, HTTPException

from ..schemas import GenerateRequest, GenerateResponse, ModelInfo
from ..services.llm.registry import available_models, get_provider
from ..services.prompt import SYSTEM_PROMPT, build_user_prompt

router = APIRouter(prefix="/api", tags=["generate"])


@router.get("/models", response_model=list[ModelInfo])
def list_models():
    return available_models()


@router.post("/generate", response_model=GenerateResponse)
def generate_verdict(request: GenerateRequest):
    try:
        provider = get_provider(request.model)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")

    user_prompt = build_user_prompt(request.input)
    text = provider.generate(SYSTEM_PROMPT, user_prompt)
    return GenerateResponse(model=request.model, text=text)
