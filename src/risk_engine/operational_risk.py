"""Operational Risk Scoring Module.

Calculates normalized operational risk scores in [0.0, 1.0] across:
1. Supplier Risk (reliability, past-only historical delays)
2. Inventory Risk (inventory coverage ratio, shortage exposure)
3. Capacity Risk (demand pressure, capacity utilization, zero-capacity handling)
4. Service Risk (product criticality modulated by shortage exposure)
"""

import math
from typing import Optional, Union
import numpy as np
import pandas as pd

from src.risk_engine.config import (
    SupplierRiskConfig,
    InventoryRiskConfig,
    CapacityRiskConfig,
    ServiceRiskConfig,
)


def compute_past_supplier_delay_rates(
    df: pd.DataFrame,
    date_col: str = "date",
    supplier_col: str = "supplier_id",
    delay_col: str = "delay_occurred",
) -> pd.Series:
    """Compute historical supplier delay rates strictly from past records.

    LEAKAGE PREVENTION GUARANTEE:
    Uses shift(1) prior to calculating expanding average. The current shipment's
    actual delay outcome is NEVER included in its own historical feature.

    Args:
        df: Input DataFrame containing shipment records.
        date_col: Date column name for chronological sorting.
        supplier_col: Supplier ID column name.
        delay_col: Actual delay outcome column.

    Returns:
        pd.Series: Expanding mean of past delay occurrences for each supplier.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    work_df = df.copy()
    if date_col in work_df.columns:
        work_df[date_col] = pd.to_datetime(work_df[date_col])
        work_df = work_df.sort_values(by=[date_col]).reset_index()
    else:
        work_df = work_df.reset_index()

    # If delay_col not present, return 0.0 series
    if delay_col not in work_df.columns:
        return pd.Series(0.0, index=df.index)

    # Convert delay to numeric binary
    work_df["_delay_num"] = pd.to_numeric(work_df[delay_col], errors="coerce").fillna(0).astype(int)

    # Compute expanding mean strictly shifted by 1 per supplier
    def _calc_supplier_history(series: pd.Series) -> pd.Series:
        return series.shift(1).expanding(min_periods=1).mean().fillna(0.0)

    past_rates = (
        work_df.groupby(supplier_col, group_keys=False)["_delay_num"]
        .apply(_calc_supplier_history)
        .round(4)
    )

    # Map back to original index
    result = pd.Series(index=work_df["index"], data=past_rates.values)
    return result.reindex(df.index).fillna(0.0)


def calculate_supplier_risk(
    reliability: float,
    historical_delay_rate: Optional[float] = None,
    thresholds: Optional[SupplierRiskConfig] = None,
) -> float:
    """Calculate normalized supplier risk score in [0.0, 1.0].

    Formula:
    - Base Risk = 1.0 - Supplier Reliability
    - If historical past delay rate is available:
        Supplier Risk = 0.70 * (1 - Reliability) + 0.30 * Past Delay Rate

    Args:
        reliability: Supplier reliability score in [0.0, 1.0].
        historical_delay_rate: Optional delay rate strictly from prior shipments.
        thresholds: Configurable supplier thresholds.

    Returns:
        float: Normalized score in [0.0, 1.0].

    Raises:
        ValueError: If reliability is invalid or outside [0, 1].
    """
    if not isinstance(reliability, (int, float)) or isinstance(reliability, bool):
        raise TypeError(f"Supplier reliability must be numeric, got {type(reliability).__name__}")

    rel_float = float(reliability)
    if math.isnan(rel_float) or math.isinf(rel_float):
        raise ValueError(f"Supplier reliability must be finite, got {rel_float}")

    if not (0.0 <= rel_float <= 1.0):
        raise ValueError(f"Supplier reliability must be in [0.0, 1.0], got {rel_float}")

    base_risk = 1.0 - rel_float

    if historical_delay_rate is not None and not math.isnan(historical_delay_rate):
        hist_rate = float(historical_delay_rate)
        if not (0.0 <= hist_rate <= 1.0):
            raise ValueError(f"Historical delay rate must be in [0.0, 1.0], got {hist_rate}")
        combined = 0.70 * base_risk + 0.30 * hist_rate
    else:
        combined = base_risk

    return round(float(np.clip(combined, 0.0, 1.0)), 4)


def calculate_inventory_risk(
    coverage_ratio: Optional[float] = None,
    inventory_units: Optional[float] = None,
    demand_units: Optional[float] = None,
    thresholds: Optional[InventoryRiskConfig] = None,
) -> float:
    """Calculate normalized inventory risk score in [0.0, 1.0].

    Uses Day 3 definition:
        inventory_coverage_ratio = inventory_units / demand_units

    Thresholds:
    - coverage < 0.5 (critical): critical stockout exposure -> [0.80, 1.00]
    - 0.5 <= coverage < 1.0 (low): elevated exposure -> [0.40, 0.80)
    - coverage >= 1.0: adequate stock buffer -> [0.00, 0.40)

    Args:
        coverage_ratio: Optional direct ratio (inventory / demand).
        inventory_units: Optional warehouse available units.
        demand_units: Optional shipment demand units.
        thresholds: Configurable inventory thresholds.

    Returns:
        float: Normalized score in [0.0, 1.0].

    Raises:
        ValueError: If inputs are missing, negative, or invalid.
    """
    crit_thresh = thresholds.critical_coverage_threshold if thresholds else 0.50
    low_thresh = thresholds.low_coverage_threshold if thresholds else 1.00

    if coverage_ratio is not None:
        cov = float(coverage_ratio)
        if math.isnan(cov) or math.isinf(cov):
            raise ValueError(f"Inventory coverage ratio must be finite, got: {cov}")
        if cov < 0:
            raise ValueError(f"Inventory coverage ratio cannot be negative, got: {cov}")
    elif inventory_units is not None and demand_units is not None:
        inv = float(inventory_units)
        dem = float(demand_units)
        if inv < 0 or dem < 0:
            raise ValueError(f"Inventory and demand units must be non-negative: inv={inv}, dem={dem}")
        if dem == 0:
            # Zero demand means no shortage exposure
            cov = 999.0
        else:
            cov = inv / dem
    else:
        raise ValueError(
            "Must provide either coverage_ratio or both (inventory_units, demand_units)."
        )

    # Continuous piece-wise risk function
    if cov <= 0.0:
        risk = 1.00
    elif cov < crit_thresh:
        # Scale [0, crit_thresh) to [1.00, 0.80]
        fraction = cov / crit_thresh
        risk = 0.80 + 0.20 * (1.0 - fraction)
    elif cov < low_thresh:
        # Scale [crit_thresh, low_thresh) to [0.80, 0.40]
        fraction = (cov - crit_thresh) / (low_thresh - crit_thresh)
        risk = 0.40 + 0.40 * (1.0 - fraction)
    else:
        # Healthy inventory: 1.0 <= cov. Diminishes from 0.40 down to 0.00 at cov >= 2.0
        buffer_fraction = min(1.0, (cov - low_thresh) / 1.0)
        risk = 0.40 * (1.0 - buffer_fraction)

    return round(float(np.clip(risk, 0.0, 1.0)), 4)


def calculate_capacity_risk(
    demand_units: float,
    capacity_units: float,
    thresholds: Optional[CapacityRiskConfig] = None,
) -> float:
    """Calculate normalized capacity risk score in [0.0, 1.0].

    Utilization = demand_units / capacity_units

    Threshold Bands:
    - utilization >= 1.00 (critical): critical capacity overload [0.85, 1.00]
    - 0.85 <= utilization < 1.00 (high): high pressure [0.60, 0.85)
    - 0.70 <= utilization < 0.85 (moderate): moderate pressure [0.30, 0.60)
    - utilization < 0.70 (low): low pressure [0.00, 0.30)

    Zero-Capacity Handling:
    - If capacity_units == 0 and demand_units > 0 -> returns 1.00 (total outage/inability to supply).
    - If capacity_units == 0 and demand_units == 0 -> returns 0.00.

    Args:
        demand_units: Required shipment units.
        capacity_units: Total available supplier capacity.
        thresholds: Configurable capacity thresholds.

    Returns:
        float: Normalized score in [0.0, 1.0].

    Raises:
        ValueError: If units are negative or infinite.
    """
    if not isinstance(demand_units, (int, float)) or not isinstance(capacity_units, (int, float)):
        raise TypeError("demand_units and capacity_units must be numeric.")

    dem = float(demand_units)
    cap = float(capacity_units)

    if math.isnan(dem) or math.isinf(dem) or math.isnan(cap) or math.isinf(cap):
        raise ValueError(f"Demand and capacity units must be finite, got dem={dem}, cap={cap}")

    if dem < 0 or cap < 0:
        raise ValueError(f"Demand and capacity units cannot be negative, got dem={dem}, cap={cap}")

    # Explicit Zero-Capacity Handling
    if cap == 0.0:
        return 1.00 if dem > 0.0 else 0.00

    utilization = dem / cap

    mod_thresh = thresholds.moderate_utilization_threshold if thresholds else 0.70
    high_thresh = thresholds.high_utilization_threshold if thresholds else 0.85
    crit_thresh = thresholds.critical_utilization_threshold if thresholds else 1.00

    if utilization >= crit_thresh:
        # Scale >= 1.0 up to 1.5 -> [0.85, 1.00]
        overload = min(1.0, (utilization - crit_thresh) / 0.50)
        risk = 0.85 + 0.15 * overload
    elif utilization >= high_thresh:
        # Scale [0.85, 1.00) -> [0.60, 0.85)
        span = crit_thresh - high_thresh
        frac = (utilization - high_thresh) / span
        risk = 0.60 + 0.25 * frac
    elif utilization >= mod_thresh:
        # Scale [0.70, 0.85) -> [0.30, 0.60)
        span = high_thresh - mod_thresh
        frac = (utilization - mod_thresh) / span
        risk = 0.30 + 0.30 * frac
    else:
        # Scale [0.00, 0.70) -> [0.00, 0.30)
        frac = utilization / mod_thresh
        risk = 0.30 * frac

    return round(float(np.clip(risk, 0.0, 1.0)), 4)


def calculate_service_risk(
    product_criticality: Optional[str] = None,
    coverage_ratio: Optional[float] = None,
    shortage_flag: Optional[int] = None,
    thresholds: Optional[ServiceRiskConfig] = None,
) -> Optional[float]:
    """Calculate normalized service risk score in [0.0, 1.0].

    Design & Leakage Prevention:
    - Never uses post-event disruption outcomes.
    - If product_criticality is missing/None/empty, marks dimension as unavailable (returns None)
      rather than fabricating an assumption.
    - If product_criticality is present:
        Base risk = mapped from tier (CRITICAL: 1.0, HIGH: 0.75, MEDIUM: 0.50, LOW: 0.25)
        Modulated by stock shortage exposure.

    Args:
        product_criticality: Optional tier string ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW').
        coverage_ratio: Optional inventory coverage ratio.
        shortage_flag: Optional shortage indicator (1 = shortage, 0 = no shortage).
        thresholds: Configurable service tier weights.

    Returns:
        Optional[float]: Normalized score in [0.0, 1.0], or None if dimension is unavailable.
    """
    if product_criticality is None:
        return None

    crit_str = str(product_criticality).strip().upper()
    if not crit_str or crit_str in {"NONE", "NAN", "NULL"}:
        return None

    weights = thresholds.criticality_weights if thresholds else {
        "CRITICAL": 1.00,
        "HIGH": 0.75,
        "MEDIUM": 0.50,
        "LOW": 0.25,
    }

    if crit_str not in weights:
        # Unrecognized criticality tier
        return None

    base_score = weights[crit_str]

    # Modulate by stock coverage / shortage exposure
    # If inventory coverage is below 0.5, maximum shortage exposure
    if coverage_ratio is not None:
        if coverage_ratio < 0.50 or (shortage_flag == 1 and coverage_ratio < 1.0):
            exposure_multiplier = 1.00
        elif coverage_ratio < 1.00:
            exposure_multiplier = 0.80
        else:
            # Healthy inventory cushions service risk
            exposure_multiplier = 0.40
    elif shortage_flag is not None:
        exposure_multiplier = 1.00 if shortage_flag == 1 else 0.40
    else:
        exposure_multiplier = 0.70

    score = base_score * exposure_multiplier
    return round(float(np.clip(score, 0.0, 1.0)), 4)
