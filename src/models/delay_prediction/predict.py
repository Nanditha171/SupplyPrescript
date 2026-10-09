"""Inference, delay probability estimation, risk band scoring, and database persistence.

Provides:
- DelayPredictor class (cached pipeline loader, zero retraining during inference)
- predict_delay function for single shipment dicts or batch DataFrames
- Operational risk level mapping (LOW, MEDIUM, HIGH, CRITICAL)
- Optional SQLAlchemy database persistence adapter
- CLI batch scoring entry point
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.delay_prediction.config import DelayPredictionConfig, default_config
from src.models.delay_prediction.features import select_pre_event_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def map_probability_to_risk_level(
    prob: float,
    thresholds: Optional[Dict[str, tuple]] = None,
) -> str:
    """Classify delay probability into operational risk bands.

    Default Bands:
    - LOW: [0.00, 0.30)
    - MEDIUM: [0.30, 0.60)
    - HIGH: [0.60, 0.80)
    - CRITICAL: [0.80, 1.00]

    Args:
        prob: Delay probability between 0.0 and 1.0.
        thresholds: Custom threshold mappings if provided.

    Returns:
        Risk level string: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.
    """
    prob_clamped = max(0.0, min(1.0, float(prob)))

    if thresholds is not None:
        for level, (low, high) in thresholds.items():
            if low <= prob_clamped < high or (high >= 1.0 and prob_clamped >= high):
                return level

    if prob_clamped < 0.30:
        return "LOW"
    elif prob_clamped < 0.60:
        return "MEDIUM"
    elif prob_clamped < 0.80:
        return "HIGH"
    else:
        return "CRITICAL"


class DelayPredictor:
    """Production predictor instance with cached serialized model pipeline."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        config: DelayPredictionConfig = default_config,
    ):
        self.config = config
        self.model_path = Path(model_path) if model_path else config.model_path
        self.metadata_path = Path(metadata_path) if metadata_path else config.metadata_path
        self._pipeline: Optional[Pipeline] = None
        self._metadata: Dict[str, Any] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load fitted pipeline and metadata from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at '{self.model_path}'. "
                "Train the model first using 'python -m src.models.delay_prediction.train'."
            )

        logger.info(f"Loading predictive pipeline from {self.model_path}...")
        self._pipeline = joblib.load(self.model_path)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {
                "model_name": self.config.model_name,
                "model_version": self.config.model_version,
                "classification_threshold": self.config.classification_threshold,
            }

    @property
    def model_version(self) -> str:
        return self._metadata.get("model_version", self.config.model_version)

    @property
    def model_name(self) -> str:
        return self._metadata.get("model_name", self.config.model_name)

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Generate delay predictions for a batch of pre-event shipment records.

        Args:
            df: Input DataFrame containing pre-event features.
            threshold: Probability decision threshold (defaults to config).

        Returns:
            DataFrame containing original identifiers and prediction columns.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")

        clf_threshold = threshold if threshold is not None else self.config.classification_threshold

        # Extract pre-event features
        X, _, _ = select_pre_event_features(df, self.config)

        # Generate probabilities
        if hasattr(self._pipeline, "predict_proba"):
            probabilities = self._pipeline.predict_proba(X)[:, 1]
        else:
            probabilities = self._pipeline.predict(X).astype(float)

        # Clamp probabilities strictly between 0 and 1
        probabilities = np.clip(probabilities, 0.0, 1.0)
        predicted_delays = (probabilities >= clf_threshold).astype(bool)
        risk_scores = np.round(probabilities, 4)
        risk_levels = [map_probability_to_risk_level(p) for p in probabilities]

        timestamp_iso = datetime.now(timezone.utc).isoformat()

        # Build output dataframe
        result_df = pd.DataFrame(index=df.index)

        # Retain available identifier columns
        for id_col in self.config.identifier_columns:
            if id_col in df.columns:
                result_df[id_col] = df[id_col]

        # Synthesize event_id if not present
        if "event_id" not in result_df.columns:
            if "date" in df.columns and "supplier_id" in df.columns and "product_id" in df.columns:
                result_df["event_id"] = (
                    "EVT_"
                    + df["date"].astype(str)
                    + "_"
                    + df["supplier_id"].astype(str)
                    + "_"
                    + df["product_id"].astype(str)
                )
            else:
                result_df["event_id"] = [f"EVT_{i+1:06d}" for i in range(len(df))]

        result_df["delay_probability"] = np.round(probabilities, 4)
        result_df["predicted_delay"] = predicted_delays
        result_df["risk_score"] = risk_scores
        result_df["risk_level"] = risk_levels
        result_df["model_name"] = self.model_name
        result_df["model_version"] = self.model_version
        result_df["prediction_timestamp"] = timestamp_iso

        return result_df

    def predict_single(
        self,
        record: Union[Dict[str, Any], pd.Series],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate prediction for a single shipment record.

        Args:
            record: Dictionary or Series with pre-event features.
            threshold: Optional decision threshold.

        Returns:
            Dictionary with prediction results.
        """
        if isinstance(record, dict):
            df_single = pd.DataFrame([record])
        elif isinstance(record, pd.Series):
            df_single = pd.DataFrame([record.to_dict()])
        else:
            raise TypeError(f"Unsupported record type: {type(record)}. Expected dict or pd.Series.")

        res_df = self.predict_batch(df_single, threshold=threshold)
        return res_df.iloc[0].to_dict()


def predict_delay(
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    model_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    config: DelayPredictionConfig = default_config,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Convenience functional interface for delay prediction inference.

    Args:
        data: Single shipment record (dict) or batch (DataFrame / list of dicts).
        model_path: Optional path to serialized model pipeline.
        threshold: Decision threshold.
        config: Configuration parameters.

    Returns:
        DataFrame for batch predictions, or Dict for single record.
    """
    predictor = DelayPredictor(model_path=model_path, config=config)

    if isinstance(data, dict):
        return predictor.predict_single(data, threshold=threshold)
    elif isinstance(data, list):
        df_batch = pd.DataFrame(data)
        return predictor.predict_batch(df_batch, threshold=threshold)
    elif isinstance(data, pd.DataFrame):
        return predictor.predict_batch(data, threshold=threshold)
    else:
        raise TypeError(f"Unsupported input data type: {type(data)}.")


def persist_predictions_to_database(
    predictions_df: pd.DataFrame,
) -> Tuple[int, Optional[str]]:
    """Persist prediction results to SQLAlchemy database using existing models.

    Args:
        predictions_df: DataFrame output from predict_batch.

    Returns:
        Tuple of (count of saved records, error message if any).
    """
    try:
        from src.database.connection import SessionLocal
        from src.database.models import Prediction, SupplyEvent
        from sqlalchemy import select
    except ImportError as e:
        msg = f"SQLAlchemy database modules not importable: {e}"
        logger.warning(msg)
        return 0, msg

    persisted_count = 0
    db = SessionLocal()
    try:
        for _, row in predictions_df.iterrows():
            event_code = str(row.get("event_id", ""))
            # Query existing supply event by code if possible
            stmt = select(SupplyEvent).where(SupplyEvent.event_id == event_code)
            event_obj = db.execute(stmt).scalars().first()

            event_db_id = event_obj.id if event_obj else None

            if event_db_id is None:
                # If no matching event in DB, we skip or link to first available
                # or log information without crashing
                logger.debug(f"No existing SupplyEvent found in DB for '{event_code}'.")
                continue

            prediction_record = Prediction(
                event_id=event_db_id,
                model_name=str(row.get("model_name", "supply_delay_classifier")),
                model_version=str(row.get("model_version", "1.0.0")),
                prediction_type="DELAY",
                delay_probability=float(row.get("delay_probability", 0.0)),
                predicted_delay_days=0.0,  # Day 6 is binary delay classifier; 0.0 placeholder per schema
                risk_score=float(row.get("risk_score", 0.0)),
                risk_level=str(row.get("risk_level", "LOW")),
            )
            db.add(prediction_record)
            persisted_count += 1

        db.commit()
        logger.info(f"Successfully persisted {persisted_count} predictions to the database.")
        return persisted_count, None
    except Exception as exc:
        db.rollback()
        err_msg = f"Failed to persist predictions to database: {exc}"
        logger.error(err_msg)
        return 0, err_msg
    finally:
        db.close()


def run_prediction_cli(
    input_csv: Path,
    output_csv: Path,
    threshold: Optional[float] = None,
    persist: bool = False,
    config: DelayPredictionConfig = default_config,
) -> pd.DataFrame:
    """Execute batch inference via CLI and export results to CSV."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found at '{input_csv}'.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading input dataset from {input_csv}...")
    input_df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(input_df)} records for inference.")

    predictor = DelayPredictor(config=config)
    predictions_df = predictor.predict_batch(input_df, threshold=threshold)

    predictions_df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {len(predictions_df)} prediction records to {output_csv}.")

    if persist:
        count, err = persist_predictions_to_database(predictions_df)
        if err:
            logger.warning(f"Database persistence note: {err}")
        else:
            logger.info(f"Database persistence completed: {count} records saved.")

    return predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate delay predictions for supply chain shipments.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input pre-event dataset CSV")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to write predictions CSV")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Classification decision threshold")
    parser.add_argument("--persist", "-p", action="store_true", help="Persist predictions to database")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    run_prediction_cli(
        input_csv=in_path,
        output_csv=out_path,
        threshold=args.threshold,
        persist=args.persist,
    )
