from __future__ import annotations

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from shared.config import get_settings
from shared.db import get_session
from shared.logger import configure_logging, logger
from shared.schemas import Timeframe
from .train import build_dataset, train_model

configure_logging()
settings = get_settings()

app = FastAPI(title="Trainer Service", version="0.1.0")


class TrainRequest(BaseModel):
    timeframe: Timeframe = Timeframe.m5
    horizon_minutes: int = 30
    model_type: str = "gbm"  # default to gbm; "logreg" optional


class TrainResponse(BaseModel):
    model_version: str | None
    metrics: dict
    top_features: list[str] | None


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("trainer_service started", env=settings.app_env)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest, session=Depends(get_session)) -> TrainResponse:
    async for db in session:
        df = await build_dataset(db, timeframe=req.timeframe, horizon_minutes=req.horizon_minutes)
    meta, metrics = train_model(df, model_type=req.model_type)
    version = meta["version"] if meta else None
    return TrainResponse(model_version=version, metrics=metrics, top_features=meta.get("top_features") if meta else None)

