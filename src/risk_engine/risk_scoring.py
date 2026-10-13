"""Composite Risk Scoring and Explanatory Reason Generation.

Calculates weighted composite risk score across operational dimensions,
applies missing-dimension policies ('renormalize' or 'require_all'),
maps composite scores to operational risk levels, and generates
auditable machine-readable codes and human-readable explanations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.risk_engine.config import RiskEngineConfig
from src.risk_engine.operational_risk import (
    calculate_supplier_risk,
    calculate_inventory_risk,
    calculate_capacity_risk,
    calculate_service_risk,
)
from src.risk_engine.probability_risk import (
    compute_delay_risk_score,
    is_predicted_delayed,
)
from src.risk_engine.schemas import (
    RiskAssessmentInput,
    RiskAssessmentResult,
    RiskReasonCode,
)


def map_score_to_risk_level(score: float, config: RiskEngineConfig) -> str:
    """Map normalized composite risk score to configured operational risk band.

    Args:
        score: Normalized risk score in [0.0, 1.0].
        config: RiskEngineConfig containing risk_levels definitions.

    Returns:
        Risk level string: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.
    """
    s = float(np.clip(score, 0.0, 1.0))
    levels = config.risk_levels

    # Check from critical downwards to prevent edge boundary misclassification
    if s >= levels.critical.min_score:
        return "CRITICAL"
    elif s >= levels.high.min_score:
        return "HIGH"
    elif s >= levels.medium.min_score:
        return "MEDIUM"
    else:
        return "LOW"


def generate_risk_reasons(
    delay_prob: float,
    delay_risk_score: float,
    supplier_reliability: float,
    historical_delay_rate: Optional[float],
    coverage_ratio: Optional[float],
    shortage_flag: Optional[int],
    capacity_utilization: Optional[float],
    product_criticality: Optional[str],
    config: RiskEngineConfig,
) -> Tuple[List[str], List[str]]:
    """Generate rule-triggered machine reason codes and human-readable explanations.

    Every reason is triggered strictly by an actual input value exceeding
    an explicit operational rule.

    Returns:
        Tuple of (reason_codes, reason_descriptions).
    """
    codes: List[str] = []
    descriptions: List[str] = []

    # 1. Delay Probability Rules
    if delay_prob >= config.delay_probability.high_threshold:
        codes.append(RiskReasonCode.CRITICAL_DELAY_PROBABILITY.value)
        descriptions.append(
            f"Critical predicted probability of shipment delay ({delay_prob:.1%} >= {config.delay_probability.high_threshold:.0%})."
        )
    elif delay_prob >= config.delay_probability.medium_threshold:
        codes.append(RiskReasonCode.HIGH_DELAY_PROBABILITY.value)
        descriptions.append(
            f"High predicted probability of shipment delay ({delay_prob:.1%} >= {config.delay_probability.medium_threshold:.0%})."
        )

    # 2. Supplier Reliability Rules
    if supplier_reliability < config.supplier.critical_reliability_threshold:
        codes.append(RiskReasonCode.CRITICAL_SUPPLIER_RELIABILITY.value)
        descriptions.append(
            f"Supplier reliability ({supplier_reliability:.2f}) is critically below threshold ({config.supplier.critical_reliability_threshold:.2f})."
        )
    elif supplier_reliability < config.supplier.low_reliability_threshold:
        codes.append(RiskReasonCode.LOW_SUPPLIER_RELIABILITY.value)
        descriptions.append(
            f"Supplier reliability ({supplier_reliability:.2f}) is below target threshold ({config.supplier.low_reliability_threshold:.2f})."
        )

    # 3. Historical Delay Rate
    if historical_delay_rate is not None and historical_delay_rate >= config.supplier.high_past_delay_rate_threshold:
        codes.append(RiskReasonCode.HIGH_HISTORICAL_SUPPLIER_DELAYS.value)
        descriptions.append(
            f"Supplier demonstrates high historical delay frequency ({historical_delay_rate:.1%} in past shipments)."
        )

    # 4. Inventory Coverage & Shortage Rules
    if coverage_ratio is not None:
        if coverage_ratio < config.inventory.critical_coverage_threshold:
            codes.append(RiskReasonCode.CRITICAL_INVENTORY_SHORTAGE.value)
            descriptions.append(
                f"Critical inventory shortfall: stock coverage ({coverage_ratio:.2f}) is below {config.inventory.critical_coverage_threshold:.2f} of required demand."
            )
        elif coverage_ratio < config.inventory.low_coverage_threshold:
            codes.append(RiskReasonCode.LOW_INVENTORY_COVERAGE.value)
            descriptions.append(
                f"Low inventory coverage ({coverage_ratio:.2f}) leaves limited buffer for shipment lead time fluctuations."
            )
    if shortage_flag == 1 and RiskReasonCode.CRITICAL_INVENTORY_SHORTAGE.value not in codes:
        codes.append(RiskReasonCode.INVENTORY_SHORTAGE.value)
        descriptions.append("Current on-hand warehouse inventory is insufficient to cover shipment demand.")

    # 5. Capacity Utilization Rules
    if capacity_utilization is not None:
        if capacity_utilization >= config.capacity.critical_utilization_threshold:
            codes.append(RiskReasonCode.CRITICAL_CAPACITY_UTILIZATION.value)
            descriptions.append(
                f"Supplier capacity utilization ({capacity_utilization:.1%}) meets or exceeds 100% capacity limit."
            )
        elif capacity_utilization >= config.capacity.high_utilization_threshold:
            codes.append(RiskReasonCode.HIGH_CAPACITY_UTILIZATION.value)
            descriptions.append(
                f"Supplier capacity utilization ({capacity_utilization:.1%}) is near maximum operating limits ({config.capacity.high_utilization_threshold:.0%})."
            )

    # 6. Service / Critical Product Exposure Rules
    if product_criticality and product_criticality.upper() in {"CRITICAL", "HIGH"}:
        if (coverage_ratio is not None and coverage_ratio < config.inventory.low_coverage_threshold) or shortage_flag == 1:
            codes.append(RiskReasonCode.CRITICAL_PRODUCT_EXPOSURE.value)
            descriptions.append(
                f"High-criticality product ({product_criticality.upper()}) is exposed to potential disruption under tight inventory."
            )

    return codes, descriptions


def calculate_composite_risk_score(
    delay_risk_score: float,
    supplier_risk_score: float,
    inventory_risk_score: float,
    capacity_risk_score: float,
    service_risk_score: Optional[float],
    config: RiskEngineConfig,
) -> Tuple[float, List[str]]:
    """Calculate weighted composite risk score respecting missing-dimension policy.

    Args:
        delay_risk_score: Normalized delay risk [0, 1].
        supplier_risk_score: Normalized supplier risk [0, 1].
        inventory_risk_score: Normalized inventory risk [0, 1].
        capacity_risk_score: Normalized capacity risk [0, 1].
        service_risk_score: Normalized service risk [0, 1] or None.
        config: RiskEngineConfig with weights and missing dimension policy.

    Returns:
        Tuple of (composite_score in [0.0, 1.0], list of unavailable dimension names).

    Raises:
        ValueError: If 'require_all' policy is violated or no usable dimensions remain.
    """
    dimensions: Dict[str, Tuple[Optional[float], float]] = {
        "delay": (delay_risk_score, config.scoring.delay_weight),
        "supplier": (supplier_risk_score, config.scoring.supplier_weight),
        "inventory": (inventory_risk_score, config.scoring.inventory_weight),
        "capacity": (capacity_risk_score, config.scoring.capacity_weight),
        "service": (service_risk_score, config.scoring.service_weight),
    }

    unavailable_dims: List[str] = [
        dim for dim, (score, _) in dimensions.items() if score is None
    ]

    policy = config.scoring.missing_dimension_policy

    if unavailable_dims and policy == "require_all":
        raise ValueError(
            f"Missing dimension policy 'require_all' failed: "
            f"dimensions {unavailable_dims} are unavailable."
        )

    available_dims = {
        dim: (score, weight)
        for dim, (score, weight) in dimensions.items()
        if score is not None
    }

    if not available_dims:
        raise ValueError("No usable risk dimensions remain for composite risk scoring.")

    total_available_weight = sum(w for _, (_, w) in available_dims.items())

    if total_available_weight <= 0.0:
        raise ValueError("Sum of available weights is zero; cannot compute composite score.")

    # Weighted sum divided by available weight (renormalization)
    weighted_sum = sum(score * weight for _, (score, weight) in available_dims.items())
    composite = weighted_sum / total_available_weight

    return round(float(np.clip(composite, 0.0, 1.0)), 4), unavailable_dims


def assess_single_shipment(
    inp: RiskAssessmentInput,
    config: RiskEngineConfig,
    model_version: str = "1.0.0",
) -> RiskAssessmentResult:
    """Perform complete risk evaluation for a single shipment record.

    Workflow:
    1. Validate delay probability & compute delay risk score
    2. Evaluate decision threshold (predicted_delay)
    3. Compute operational risk scores (supplier, inventory, capacity, service)
    4. Compute composite risk score & missing dimension policy
    5. Classify operational risk level
    6. Generate triggered risk reasons
    7. Return structured RiskAssessmentResult
    """
    # 1. Delay risk & binary prediction
    delay_score = compute_delay_risk_score(inp.delay_probability)
    pred_delay = is_predicted_delayed(
        inp.delay_probability,
        classification_threshold=config.delay_probability.classification_threshold,
    )

    # 2. Operational dimensions
    supplier_score = calculate_supplier_risk(
        reliability=inp.supplier_reliability,
        historical_delay_rate=inp.historical_delay_rate,
        thresholds=config.supplier,
    )

    cov_ratio = inp.inventory_coverage_ratio
    if cov_ratio is None and inp.inventory_units is not None and inp.demand_units is not None:
        cov_ratio = (
            999.0 if inp.demand_units == 0 else inp.inventory_units / inp.demand_units
        )

    inventory_score = calculate_inventory_risk(
        coverage_ratio=cov_ratio,
        inventory_units=inp.inventory_units,
        demand_units=inp.demand_units,
        thresholds=config.inventory,
    )

    cap_units = (
        inp.supplier_capacity_units
        if inp.supplier_capacity_units is not None
        else (inp.demand_units / inp.capacity_utilization if (inp.demand_units and inp.capacity_utilization) else 0.0)
    )
    capacity_score = calculate_capacity_risk(
        demand_units=inp.demand_units or 0.0,
        capacity_units=cap_units or 0.0,
        thresholds=config.capacity,
    )

    service_score = calculate_service_risk(
        product_criticality=inp.product_criticality,
        coverage_ratio=cov_ratio,
        shortage_flag=inp.shortage_flag,
        thresholds=config.service,
    )

    # 3. Composite score
    composite_score, unavail_dims = calculate_composite_risk_score(
        delay_risk_score=delay_score,
        supplier_risk_score=supplier_score,
        inventory_risk_score=inventory_score,
        capacity_risk_score=capacity_score,
        service_risk_score=service_score,
        config=config,
    )

    # 4. Risk Level
    risk_level = map_score_to_risk_level(composite_score, config)

    # 5. Risk Reasons
    cap_util = inp.capacity_utilization
    if cap_util is None and inp.demand_units and cap_units:
        cap_util = inp.demand_units / cap_units if cap_units > 0 else 1.0

    reason_codes, reason_descs = generate_risk_reasons(
        delay_prob=inp.delay_probability,
        delay_risk_score=delay_score,
        supplier_reliability=inp.supplier_reliability,
        historical_delay_rate=inp.historical_delay_rate,
        coverage_ratio=cov_ratio,
        shortage_flag=inp.shortage_flag,
        capacity_utilization=cap_util,
        product_criticality=inp.product_criticality,
        config=config,
    )

    return RiskAssessmentResult(
        event_id=inp.event_id,
        delay_probability=inp.delay_probability,
        predicted_delay=pred_delay,
        delay_risk_score=delay_score,
        supplier_risk_score=supplier_score,
        inventory_risk_score=inventory_score,
        capacity_risk_score=capacity_score,
        service_risk_score=service_score,
        composite_risk_score=composite_score,
        risk_level=risk_level,
        risk_reasons=reason_codes,
        risk_reason_descriptions=reason_descs,
        unavailable_dimensions=unavail_dims,
        model_version=model_version,
        configuration_version=config.version,
    )
