"""Pydantic schemas and data validation models for Supply Prescript Risk Engine.

Provides models for:
- RiskAssessmentInput
- RiskAssessmentResult
- RiskAlert
- RiskEngineSummary
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskReasonCode(str, Enum):
    HIGH_DELAY_PROBABILITY = "HIGH_DELAY_PROBABILITY"
    CRITICAL_DELAY_PROBABILITY = "CRITICAL_DELAY_PROBABILITY"
    LOW_SUPPLIER_RELIABILITY = "LOW_SUPPLIER_RELIABILITY"
    CRITICAL_SUPPLIER_RELIABILITY = "CRITICAL_SUPPLIER_RELIABILITY"
    HIGH_HISTORICAL_SUPPLIER_DELAYS = "HIGH_HISTORICAL_SUPPLIER_DELAYS"
    INVENTORY_SHORTAGE = "INVENTORY_SHORTAGE"
    CRITICAL_INVENTORY_SHORTAGE = "CRITICAL_INVENTORY_SHORTAGE"
    LOW_INVENTORY_COVERAGE = "LOW_INVENTORY_COVERAGE"
    HIGH_CAPACITY_UTILIZATION = "HIGH_CAPACITY_UTILIZATION"
    CRITICAL_CAPACITY_UTILIZATION = "CRITICAL_CAPACITY_UTILIZATION"
    CRITICAL_PRODUCT_EXPOSURE = "CRITICAL_PRODUCT_EXPOSURE"


class RiskAssessmentInput(BaseModel):
    """Input payload representing pre-event features for a shipment event."""

    event_id: str = Field(..., description="Unique shipment event identifier")
    delay_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted delay probability")
    supplier_reliability: float = Field(..., ge=0.0, le=1.0, description="Supplier reliability index")
    historical_delay_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Strictly past delay rate")
    inventory_coverage_ratio: Optional[float] = Field(None, ge=0.0, description="Inventory units / Demand units")
    inventory_units: Optional[float] = Field(None, ge=0.0, description="Warehouse stock units")
    demand_units: Optional[float] = Field(None, ge=0.0, description="Expected shipment demand units")
    shortage_flag: Optional[int] = Field(None, ge=0, le=1, description="Binary shortage indicator")
    supplier_capacity_units: Optional[float] = Field(None, ge=0.0, description="Supplier maximum capacity")
    capacity_utilization: Optional[float] = Field(None, ge=0.0, description="Demand / Capacity ratio")
    product_criticality: Optional[str] = Field(None, description="Product criticality tier (LOW, MEDIUM, HIGH, CRITICAL)")
    category: Optional[str] = Field(None, description="Product category")
    supplier_id: Optional[str] = Field(None, description="Supplier identifier")
    supplier_name: Optional[str] = Field(None, description="Supplier display name")
    product_id: Optional[str] = Field(None, description="Product SKU identifier")
    product_name: Optional[str] = Field(None, description="Product display name")
    date: Optional[str] = Field(None, description="Operational event date")

    model_config = ConfigDict(extra="ignore")


class RiskAssessmentResult(BaseModel):
    """Output risk evaluation result for a shipment, ready for Optimization Engine."""

    event_id: str = Field(..., description="Unique shipment event identifier")
    delay_probability: float = Field(..., ge=0.0, le=1.0, description="Model delay probability")
    predicted_delay: bool = Field(..., description="Binary delay classification using validated threshold")
    delay_risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalized delay risk score")
    supplier_risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalized supplier risk score")
    inventory_risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalized inventory risk score")
    capacity_risk_score: float = Field(..., ge=0.0, le=1.0, description="Normalized capacity risk score")
    service_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Normalized service risk score")
    composite_risk_score: float = Field(..., ge=0.0, le=1.0, description="Weighted composite risk score")
    risk_level: str = Field(..., description="Operational risk band: LOW, MEDIUM, HIGH, CRITICAL")
    risk_reasons: List[str] = Field(default_factory=list, description="Machine reason codes")
    risk_reason_descriptions: List[str] = Field(default_factory=list, description="Human-readable reason explanations")
    unavailable_dimensions: List[str] = Field(default_factory=list, description="List of omitted/unavailable risk dimensions")
    model_version: str = Field(..., description="Trained model release version")
    configuration_version: str = Field(default="1.0.0", description="Risk engine configuration version")
    assessment_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO UTC assessment timestamp",
    )

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            raise ValueError(f"Invalid risk_level: {v}. Must be LOW, MEDIUM, HIGH, or CRITICAL.")
        return v_upper

    @property
    def risk_score(self) -> float:
        """Alias for composite_risk_score conforming to optimization engine contract."""
        return self.composite_risk_score

    def to_optimization_contract(self) -> Dict[str, Any]:
        """Produce the standardized Day 9 optimization interface dictionary."""
        return {
            "event_id": self.event_id,
            "delay_probability": self.delay_probability,
            "risk_score": self.composite_risk_score,
            "risk_level": self.risk_level,
            "supplier_risk_score": self.supplier_risk_score,
            "inventory_risk_score": self.inventory_risk_score,
            "capacity_risk_score": self.capacity_risk_score,
            "service_risk_score": self.service_risk_score,
            "risk_reasons": self.risk_reasons,
            "model_version": self.model_version,
            "assessment_timestamp": self.assessment_timestamp,
        }

    model_config = ConfigDict(from_attributes=True)


class RiskAlert(BaseModel):
    """Structured alert record for priority or warning supply disruption risks."""

    event_id: str = Field(..., description="Referenced shipment event code")
    alert_type: str = Field(..., description="Type identifier: PRIORITY_DISRUPTION_ALERT, WARNING_DISRUPTION_ALERT, MONITORING_NOTICE")
    severity: str = Field(..., description="Alert severity: CRITICAL, HIGH, MEDIUM, LOW")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite risk score triggering alert")
    alert_message: str = Field(..., description="Human-readable alert notification summary")
    triggered_reasons: List[str] = Field(default_factory=list, description="Reason codes that triggered alert")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO UTC creation timestamp",
    )

    @property
    def deduplication_key(self) -> str:
        """Key used to prevent duplicate alerts on repeated batch executions."""
        return f"{self.event_id}::{self.alert_type}"

    model_config = ConfigDict(from_attributes=True)


class RiskEngineSummary(BaseModel):
    """Aggregated execution summary across the processed shipment cohort."""

    total_shipments_assessed: int = Field(..., ge=0)
    processed_count: int = Field(..., ge=0)
    skipped_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    risk_level_counts: Dict[str, int] = Field(default_factory=dict)
    alert_counts: Dict[str, int] = Field(default_factory=dict)
    average_composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    model_version: str = Field(...)
    configuration_version: str = Field(...)
    classification_threshold: float = Field(...)
    execution_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_config = ConfigDict(from_attributes=True)
