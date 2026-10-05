"""
Supply Prescript — Feature Engineering Module (Day 3)
=====================================================
Transforms cleaned supply-chain operational records into domain-rich,
leakage-free feature representations for predictive modeling (Day 6) and
prescriptive optimization (Day 10).

Features engineered include:
1. Inventory Coverage & Shortage Indicators
2. Capacity Utilization & Demand Pressure
3. Sourcing & Logistics Cost Differentials (Secondary Premium, Air Freight Premium)
4. Supplier Operational Risk Indicators
5. Strictly Past-Only Historical & Rolling Performance Metrics (No Target Leakage)
6. Lead Time Variances & Operational Outcome Metrics (Post-Event Isolation)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes inventory coverage, gap, and shortage indicators.
    """
    df = df.copy()
    
    # 1. Inventory Coverage Ratio
    df["inventory_coverage_ratio"] = (df["inventory_units"] / df["demand_units"]).round(4)
    
    # 2. Inventory Gap Units (Positive = Shortage, Zero = Balanced, Negative = Surplus)
    df["inventory_gap_units"] = df["demand_units"] - df["inventory_units"]
    
    # 3. Shortage Flag (1 if inventory cannot cover demand, 0 otherwise)
    df["shortage_flag"] = (df["inventory_units"] < df["demand_units"]).astype(int)
    
    return df


def compute_capacity_and_supplier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes capacity utilization, demand pressure, and supplier risk scores.
    """
    df = df.copy()
    
    # Capacity Utilization & Demand Pressure
    df["capacity_utilization"] = (df["demand_units"] / df["supplier_capacity_units"]).round(4)
    df["demand_pressure"] = df["capacity_utilization"]  # Standardized alias for multi-echelon modeling
    
    # Supplier Risk Score (Inverted reliability: higher = riskier)
    df["supplier_risk_score"] = (1.0 - df["supplier_reliability"]).round(4)
    
    return df


def compute_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes primary cost per unit, secondary sourcing differentials, and air freight premiums.
    """
    df = df.copy()
    
    # Cost per demand unit under primary sourcing
    df["cost_per_demand_unit"] = (df["estimated_primary_cost_usd"] / df["demand_units"]).round(2)
    
    # Secondary Supplier Cost Difference (Premium cost above primary)
    df["secondary_cost_difference"] = (
        df["estimated_secondary_cost_usd"] - df["estimated_primary_cost_usd"]
    ).round(2)
    
    # Air Freight Premium (Expedited surcharge over standard logistics)
    df["air_freight_premium"] = (
        df["air_freight_cost_usd"] - df["transport_cost_usd"]
    ).round(2)
    
    return df


def compute_lead_time_and_outcome_features(
    df: pd.DataFrame,
    low_max_days: int = 0,
    medium_max_days: int = 3,
) -> pd.DataFrame:
    """
    Computes post-event lead time variance and lead time risk tier.
    NOTE: These are post-event outcome features and must not be used as inputs
    for predictive delay classification models.
    """
    df = df.copy()
    
    # Lead Time Variance Days (Actual - Planned)
    df["lead_time_variance_days"] = df["actual_lead_time_days"] - df["planned_lead_time_days"]
    
    # Lead Time Risk Classification
    def classify_lead_time_risk(variance: int) -> str:
        if variance <= low_max_days:
            return "Low"
        elif variance <= medium_max_days:
            return "Medium"
        else:
            return "High"

    df["lead_time_risk"] = df["lead_time_variance_days"].apply(classify_lead_time_risk)
    
    return df


def compute_historical_rolling_features(
    df: pd.DataFrame,
    window_size: int = 4,
) -> pd.DataFrame:
    """
    Generates historical lag and rolling aggregate features for each supplier-product pair.
    
    CRITICAL LEAKAGE PREVENTION RULE:
    Only past observations strictly prior to the current record are used (shift(1)).
    Current and future target/operational outcomes are never leaked.
    """
    df = df.copy()
    
    # Ensure chronological ordering
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["supplier_id", "product_id", "date"]).reset_index(drop=True)
    
    # Group by supplier and product to compute supplier-product specific trajectory
    grouped = df.groupby(["supplier_id", "product_id"])
    
    # 1. Previous Delay Days (Lag 1)
    df["previous_delay_days"] = grouped["delay_days"].shift(1).fillna(0).astype(int)
    
    # 2. Previous Delay Flag (Lag 1)
    df["previous_delay_flag"] = grouped["delay_occurred"].shift(1).fillna(0).astype(int)
    
    # 3. Rolling Average Delay Days (Past window_size cycles)
    df["rolling_average_delay"] = (
        grouped["delay_days"]
        .apply(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
        .round(2)
    )
    
    # 4. Rolling Average Demand (Past window_size cycles)
    df["rolling_average_demand"] = (
        grouped["demand_units"]
        .apply(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    # Fill first observation of each group with current demand to prevent cold-start nulls
    df["rolling_average_demand"] = df["rolling_average_demand"].fillna(df["demand_units"]).round(1)
    
    # 5. Rolling Average Inventory (Past window_size cycles)
    df["rolling_average_inventory"] = (
        grouped["inventory_units"]
        .apply(lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    # Fill first observation of each group with current inventory
    df["rolling_average_inventory"] = df["rolling_average_inventory"].fillna(df["inventory_units"]).round(1)
    
    # Re-sort chronologically by date and product
    df = df.sort_values(by=["date", "product_id"]).reset_index(drop=True)
    return df


def engineer_all_features(
    df: pd.DataFrame,
    thresholds: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Orchestrates end-to-end feature engineering pipeline across all domain dimensions.
    """
    if thresholds is None:
        thresholds = {
            "low_max_days": 0,
            "medium_max_days": 3,
            "window_size": 4,
        }
        
    df = compute_inventory_features(df)
    df = compute_capacity_and_supplier_features(df)
    df = compute_cost_features(df)
    df = compute_lead_time_and_outcome_features(
        df,
        low_max_days=thresholds.get("low_max_days", 0),
        medium_max_days=thresholds.get("medium_max_days", 3),
    )
    df = compute_historical_rolling_features(
        df,
        window_size=thresholds.get("window_size", 4),
    )
    
    return df


def verify_no_data_leakage(
    feature_names: List[str],
    forbidden_post_event_cols: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates that a given set of predictive features does not contain
    any post-event outcome variables (e.g. actual_lead_time_days, delay_days).
    """
    if forbidden_post_event_cols is None:
        forbidden_post_event_cols = [
            "actual_lead_time_days",
            "delay_occurred",
            "delay_days",
            "lead_time_variance_days",
            "lead_time_risk",
        ]
        
    violations = [col for col in feature_names if col in forbidden_post_event_cols]
    is_leakage_free = len(violations) == 0
    return is_leakage_free, violations
