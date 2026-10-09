"""Delay prediction machine learning module for Supply Prescript.

Provides end-to-end capabilities:
- Feature extraction & strict leakage prevention
- Preprocessing pipelines
- Model training & multi-model comparison
- Evaluation reporting & visualization
- Inference, calibrated delay probability & operational risk scoring
- Optional database persistence
"""

from src.models.delay_prediction.config import DelayPredictionConfig
from src.models.delay_prediction.predict import predict_delay, DelayPredictor

__all__ = [
    "DelayPredictionConfig",
    "predict_delay",
    "DelayPredictor",
]
