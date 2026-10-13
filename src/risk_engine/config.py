"""Configuration management and startup validation for Supply Prescript Risk Engine.

Strictly validates:
- Dimension weights are non-negative and sum to 1.0 within tolerance.
- Does not silently normalize invalid weights or invalid thresholds.
- Validates all boundary ordering and operational thresholds at startup.
"""

from pathlib import Path
from typing import Dict, Literal, Optional, Union
import yaml
from pydantic import BaseModel, Field, model_validator


class LevelRange(BaseModel):
    min_score: float = Field(..., ge=0.0, le=1.0)
    max_score: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_range(self) -> "LevelRange":
        if self.min_score > self.max_score:
            raise ValueError(
                f"Invalid score boundary: min_score ({self.min_score}) cannot exceed "
                f"max_score ({self.max_score})."
            )
        return self


class RiskLevelsConfig(BaseModel):
    low: LevelRange
    medium: LevelRange
    high: LevelRange
    critical: LevelRange

    @model_validator(mode="after")
    def validate_continuity(self) -> "RiskLevelsConfig":
        if self.low.min_score != 0.0:
            raise ValueError(f"Low risk level min_score must start at 0.0, got {self.low.min_score}")
        if self.critical.max_score != 1.0:
            raise ValueError(f"Critical risk level max_score must end at 1.0, got {self.critical.max_score}")
        if not (self.low.max_score <= self.medium.min_score + 0.01):
            raise ValueError("Gap or overlap detected between Low and Medium risk boundaries.")
        if not (self.medium.max_score <= self.high.min_score + 0.01):
            raise ValueError("Gap or overlap detected between Medium and High risk boundaries.")
        if not (self.high.max_score <= self.critical.min_score + 0.01):
            raise ValueError("Gap or overlap detected between High and Critical risk boundaries.")
        return self


class DelayProbabilityConfig(BaseModel):
    low_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    medium_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    high_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    classification_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "DelayProbabilityConfig":
        if not (self.low_threshold < self.medium_threshold < self.high_threshold):
            raise ValueError(
                f"Delay probability thresholds must be strictly increasing: "
                f"low ({self.low_threshold}) < medium ({self.medium_threshold}) < high ({self.high_threshold})"
            )
        return self


class SupplierRiskConfig(BaseModel):
    low_reliability_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    critical_reliability_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    high_past_delay_rate_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_ordering(self) -> "SupplierRiskConfig":
        if self.critical_reliability_threshold > self.low_reliability_threshold:
            raise ValueError(
                f"critical_reliability_threshold ({self.critical_reliability_threshold}) cannot exceed "
                f"low_reliability_threshold ({self.low_reliability_threshold})"
            )
        return self


class InventoryRiskConfig(BaseModel):
    low_coverage_threshold: float = Field(default=1.0, ge=0.0)
    critical_coverage_threshold: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="after")
    def validate_ordering(self) -> "InventoryRiskConfig":
        if self.critical_coverage_threshold > self.low_coverage_threshold:
            raise ValueError(
                f"critical_coverage_threshold ({self.critical_coverage_threshold}) cannot exceed "
                f"low_coverage_threshold ({self.low_coverage_threshold})"
            )
        return self


class CapacityRiskConfig(BaseModel):
    moderate_utilization_threshold: float = Field(default=0.70, ge=0.0)
    high_utilization_threshold: float = Field(default=0.85, ge=0.0)
    critical_utilization_threshold: float = Field(default=1.00, ge=0.0)

    @model_validator(mode="after")
    def validate_ordering(self) -> "CapacityRiskConfig":
        if not (
            self.moderate_utilization_threshold
            < self.high_utilization_threshold
            <= self.critical_utilization_threshold
        ):
            raise ValueError(
                "Capacity utilization thresholds must satisfy: moderate < high <= critical"
            )
        return self


class ServiceRiskConfig(BaseModel):
    criticality_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "CRITICAL": 1.00,
            "HIGH": 0.75,
            "MEDIUM": 0.50,
            "LOW": 0.25,
        }
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "ServiceRiskConfig":
        for k, v in self.criticality_weights.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"Service criticality weight for '{k}' must be in [0, 1], got {v}")
        return self


class ScoringWeightsConfig(BaseModel):
    delay_weight: float = Field(..., ge=0.0, le=1.0)
    supplier_weight: float = Field(..., ge=0.0, le=1.0)
    inventory_weight: float = Field(..., ge=0.0, le=1.0)
    capacity_weight: float = Field(..., ge=0.0, le=1.0)
    service_weight: float = Field(..., ge=0.0, le=1.0)
    missing_dimension_policy: Literal["renormalize", "require_all"] = "renormalize"

    @model_validator(mode="after")
    def validate_sum(self) -> "ScoringWeightsConfig":
        total = (
            self.delay_weight
            + self.supplier_weight
            + self.inventory_weight
            + self.capacity_weight
            + self.service_weight
        )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Scoring weights must sum to 1.0 within numerical tolerance. "
                f"Calculated sum: {total:.6f} (weights: delay={self.delay_weight}, "
                f"supplier={self.supplier_weight}, inventory={self.inventory_weight}, "
                f"capacity={self.capacity_weight}, service={self.service_weight}). "
                "Silent normalization is prohibited."
            )
        return self


class AlertsConfig(BaseModel):
    high_severity_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    critical_severity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_ordering(self) -> "AlertsConfig":
        if self.high_severity_threshold > self.critical_severity_threshold:
            raise ValueError(
                f"high_severity_threshold ({self.high_severity_threshold}) cannot exceed "
                f"critical_severity_threshold ({self.critical_severity_threshold})"
            )
        return self


class ModelPathsConfig(BaseModel):
    model_path: Path = Path("models/delay_prediction/delay_classifier.joblib")
    metadata_path: Path = Path("models/delay_prediction/model_metadata.json")
    evaluation_metrics_path: Path = Path("reports/model_evaluation/metrics.json")


class DataPathsConfig(BaseModel):
    processed_dataset_path: Path = Path("data/processed/supply_prescript_processed_dataset.csv")


class RiskEngineConfig(BaseModel):
    version: str = "1.0.0"
    approved_date: str = "2026-10-12"
    risk_levels: RiskLevelsConfig
    delay_probability: DelayProbabilityConfig
    supplier: SupplierRiskConfig
    inventory: InventoryRiskConfig
    capacity: CapacityRiskConfig
    service: ServiceRiskConfig
    scoring: ScoringWeightsConfig
    alerts: AlertsConfig
    model: ModelPathsConfig = Field(default_factory=ModelPathsConfig)
    data: DataPathsConfig = Field(default_factory=DataPathsConfig)


def load_risk_config(config_path: Optional[Union[str, Path]] = None) -> RiskEngineConfig:
    """Load and validate Risk Engine configuration from YAML.

    Args:
        config_path: Optional path to YAML configuration file. Defaults to config/risk_config.yaml.

    Returns:
        RiskEngineConfig instance with strictly validated parameters.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If weights or thresholds are invalid.
    """
    if config_path is None:
        # Search relative to repo root
        default_path = Path("config/risk_config.yaml")
        if not default_path.exists():
            # Try finding from current file location
            default_path = Path(__file__).resolve().parents[2] / "config" / "risk_config.yaml"
        config_path = default_path
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Risk configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"Invalid YAML content in {config_path}; expected a mapping/dictionary.")

    return RiskEngineConfig(**raw_data)
