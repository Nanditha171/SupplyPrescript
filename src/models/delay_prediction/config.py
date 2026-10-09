"""Configuration module for Delay Prediction in Supply Prescript."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class DelayPredictionConfig:
    """Central configuration for delay prediction training, evaluation, and inference."""

    # Project directories and filepaths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])
    dataset_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "processed" / "supply_prescript_processed_dataset.csv"
    )
    model_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "models" / "delay_prediction"
    )
    report_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "reports" / "model_evaluation"
    )
    predictions_output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "data" / "predictions"
    )

    # File names
    model_filename: str = "delay_classifier.joblib"
    preprocessor_filename: str = "preprocessing_pipeline.joblib"
    metadata_filename: str = "model_metadata.json"

    # Target definition
    target_column: str = "delay_occurred"
    date_column: str = "date"

    # Modeling parameters
    model_name: str = "supply_delay_classifier"
    model_version: str = "1.0.0"
    random_seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.20  # Proportion of training data reserved for validation
    classification_threshold: float = 0.50
    model_selection_metric: str = "average_precision"  # PR-AUC / Average Precision

    # Risk classification thresholds
    # LOW: < 0.30, MEDIUM: 0.30 - < 0.60, HIGH: 0.60 - < 0.80, CRITICAL: >= 0.80
    risk_thresholds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "LOW": (0.00, 0.30),
            "MEDIUM": (0.30, 0.60),
            "HIGH": (0.60, 0.80),
            "CRITICAL": (0.80, 1.01),
        }
    )

    # Approved pre-event feature candidates (available before shipment outcome)
    approved_numerical_features: List[str] = field(
        default_factory=lambda: [
            "supplier_reliability",
            "demand_units",
            "inventory_units",
            "supplier_capacity_units",
            "planned_lead_time_days",
            "unit_cost_usd",
            "transport_cost_usd",
            "secondary_supplier_premium_pct",
            "inventory_coverage_ratio",
            "inventory_gap_units",
            "shortage_flag",
            "capacity_utilization",
            "demand_pressure",
            "supplier_risk_score",
            "cost_per_demand_unit",
            "secondary_cost_difference",
            "air_freight_premium",
            "previous_delay_days",
            "previous_delay_flag",
            "rolling_average_delay",
            "rolling_average_demand",
            "rolling_average_inventory",
            "demand_outlier_flag",
            "inventory_outlier_flag",
            "cost_outlier_flag",
            "lead_time_outlier_flag",
        ]
    )

    approved_categorical_features: List[str] = field(
        default_factory=lambda: [
            "supplier_country",
            "category",
            "supplier_id",
        ]
    )

    # Forbidden post-event leakage columns
    leakage_columns: List[str] = field(
        default_factory=lambda: [
            "actual_lead_time_days",
            "delay_days",
            "lead_time_variance_days",
            "lead_time_risk",
            "service_risk",
            "delay_occurred",
            "actual_delivery_date",
            "actual_cost",
        ]
    )

    # Identifier and metadata columns (kept for tracking, not model input)
    identifier_columns: List[str] = field(
        default_factory=lambda: [
            "date",
            "supplier_id",
            "supplier_name",
            "product_id",
            "product_name",
            "event_id",
        ]
    )

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    @property
    def preprocessor_path(self) -> Path:
        return self.model_dir / self.preprocessor_filename

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / self.metadata_filename


# Global default configuration instance
default_config = DelayPredictionConfig()
