from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from shared.models import MarketBar, MarketFeatures, NewsScore, PatternSignal
from shared.schemas import Timeframe


def _tf_minutes(tf: Timeframe) -> int:
    return {"1m": 1, "5m": 5, "15m": 15, "1h": 60}[tf.value]


def _load_to_df(rows: Sequence) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    first = rows[0]
    if hasattr(first, "_mapping"):  # SQLAlchemy Row
        data = [dict(r._mapping) for r in rows]
    elif isinstance(first, dict):
        data = rows
    else:
        data = [r.__dict__ for r in rows]
    df = pd.DataFrame(data)
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if "ts_scored" in df:
        df["ts_scored"] = pd.to_datetime(df["ts_scored"], utc=True)
    return df


async def build_dataset(session, timeframe: Timeframe = Timeframe.m5, horizon_minutes: int = 30):
    bars_res = await session.execute(
        MarketBar.__table__.select().where(MarketBar.timeframe == timeframe).order_by(MarketBar.ts)
    )
    feats_res = await session.execute(
        MarketFeatures.__table__.select().where(MarketFeatures.timeframe == timeframe).order_by(MarketFeatures.ts)
    )
    pat_res = await session.execute(
        PatternSignal.__table__.select().where(PatternSignal.timeframe == timeframe).order_by(PatternSignal.ts)
    )
    news_res = await session.execute(NewsScore.__table__.select().order_by(NewsScore.ts_scored))

    df_bars = _load_to_df(bars_res.mappings().all())
    df_feats = _load_to_df(feats_res.mappings().all())
    df_pat = _load_to_df(pat_res.mappings().all())
    df_news = _load_to_df(news_res.mappings().all())

    if df_bars.empty:
        return pd.DataFrame()

    tf_min = _tf_minutes(timeframe)
    horizon_steps = max(1, horizon_minutes // tf_min)
    df_bars = df_bars.sort_values(["symbol", "ts"])
    df_bars["future_close"] = df_bars.groupby("symbol")["close"].shift(-horizon_steps)
    df_bars["future_ret"] = (df_bars["future_close"] - df_bars["close"]) / df_bars["close"]
    df_bars["label"] = np.where(
        df_bars["future_ret"] > 0.001, "long", np.where(df_bars["future_ret"] < -0.001, "short", "hold")
    )

    df = df_bars.merge(df_feats.drop(columns=["id", "timeframe"], errors="ignore"), on=["symbol", "ts"], how="left")
    df = df.merge(df_pat.drop(columns=["id", "timeframe"], errors="ignore"), on=["symbol", "ts"], how="left")

    if not df_news.empty:
        df = pd.merge_asof(
            df.sort_values("ts"),
            df_news[["ts_scored", "news_bias", "news_confidence"]].rename(columns={"ts_scored": "ts_news"}),
            left_on="ts",
            right_on="ts_news",
            direction="backward",
        )
    else:
        df["news_bias"] = 0.0
        df["news_confidence"] = 0.0
    return df


def train_model(df: pd.DataFrame, model_type: str = "logreg"):
    if df.empty:
        return None, {}
    feature_cols = [
        "atr_pct",
        "ema20",
        "ema50",
        "ema200",
        "rsi",
        "returns",
        "vol",
        "spread",
        "market_bias",
        "market_confidence",
        "news_bias",
        "news_confidence",
    ]
    for col in feature_cols:
        if col not in df:
            df[col] = 0.0
    X = df[feature_cols].fillna(0.0)
    y = df["label"]
    if y.nunique() < 2:
        return None, {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_type == "gbm":
        base = GradientBoostingClassifier()
    else:
        base = LogisticRegression(max_iter=400, multi_class="auto", n_jobs=-1)

    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    classes = model.classes_
    conf = proba.max(axis=1)
    preds = classes[proba.argmax(axis=1)]
    mask = conf >= 0.7
    if mask.any():
        precision = (preds[mask] == y_test.to_numpy()[mask]).mean()
    else:
        precision = 0.0
    metrics = {
        "precision_conf_ge_0.7": float(precision),
        "n_conf_ge_0.7": int(mask.sum()),
        "test_size": int(len(y_test)),
        "model_type": model_type,
    }

    # top features (importance or coef)
    top_features: list[str] = []
    try:
        if hasattr(base, "feature_importances_"):
            importances = base.feature_importances_
            top_idx = np.argsort(importances)[::-1][:5]
            top_features = [feature_cols[i] for i in top_idx]
        elif hasattr(base, "coef_"):
            coefs = np.abs(base.coef_).mean(axis=0)
            top_idx = np.argsort(coefs)[::-1][:5]
            top_features = [feature_cols[i] for i in top_idx]
    except Exception:
        top_features = []

    ts_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_{ts_str}.joblib"
    artifact = {
        "model": model,
        "features": feature_cols,
        "classes": list(classes),
        "version": ts_str,
        "top_features": top_features,
        "metrics": metrics,
        "model_type": model_type,
    }
    joblib.dump(artifact, model_path)

    # auto-pick best: compare precision_conf_ge_0.7 with existing best meta if any
    best_path = model_dir / "best_model.json"
    best_score = -1
    if best_path.exists():
        try:
            with open(best_path) as f:
                best_meta = json.load(f)
            best_score = best_meta.get("metrics", {}).get("precision_conf_ge_0.7", -1)
        except Exception:
            best_score = -1
    if metrics["precision_conf_ge_0.7"] >= best_score:
        joblib.dump(artifact, model_dir / "best_model.joblib")
        with open(best_path, "w") as f:
            json.dump(
                {
                    "version": ts_str,
                    "metrics": metrics,
                    "features": feature_cols,
                    "classes": list(classes),
                    "top_features": top_features,
                    "model_type": model_type,
                    "model_path": str(model_dir / "best_model.joblib"),
                },
                f,
                indent=2,
            )

    meta = {
        "version": ts_str,
        "metrics": metrics,
        "model_path": str(model_path),
        "features": feature_cols,
        "classes": list(classes),
        "top_features": top_features,
        "model_type": model_type,
    }
    with open(model_dir / f"model_{ts_str}.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta, metrics
