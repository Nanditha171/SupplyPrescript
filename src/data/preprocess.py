"""
Supply Prescript — Data Engineering & Preprocessing Pipeline (Day 3)
====================================================================
Production data-engineering pipeline that cleans, validates, and engineers
features from raw supply-chain operational records.

Pipeline Steps:
1. Load Raw Dataset (Source of Truth - Read Only)
2. Schema & Business Key Validation
3. Missing Value Imputation (Median for numerical, Mode for categorical)
4. Duplicate Detection & Idempotent Removal
5. Data Type Standardization & Chronological Sorting
6. Operational Domain Integrity Validation
7. Systematic IQR Outlier Detection & Outlier Flagging
8. Feature Engineering (Domain features, Risk indices, Lag/Rolling aggregates)
9. Data Leakage Prevention Check (Pre-event vs Post-event separation)
10. Data Quality Report Generation (`reports/data_quality_report.csv`)
11. Save Processed Dataset (`data/processed/supply_prescript_processed_dataset.csv`)
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

# Import feature engineering module
try:
    from src.data.feature_engineering import engineer_all_features, verify_no_data_leakage
except ImportError:
    from feature_engineering import engineer_all_features, verify_no_data_leakage

RAW_COLUMNS = [
    "date",
    "supplier_id",
    "supplier_name",
    "supplier_country",
    "supplier_reliability",
    "product_id",
    "product_name",
    "category",
    "demand_units",
    "inventory_units",
    "supplier_capacity_units",
    "planned_lead_time_days",
    "actual_lead_time_days",
    "delay_occurred",
    "delay_days",
    "unit_cost_usd",
    "transport_cost_usd",
    "secondary_supplier_premium_pct",
    "inventory_demand_ratio",
    "estimated_primary_cost_usd",
    "estimated_secondary_cost_usd",
    "air_freight_cost_usd",
    "service_risk",
]

NUMERICAL_COLS = [
    "supplier_reliability",
    "demand_units",
    "inventory_units",
    "supplier_capacity_units",
    "planned_lead_time_days",
    "actual_lead_time_days",
    "delay_days",
    "unit_cost_usd",
    "transport_cost_usd",
    "secondary_supplier_premium_pct",
    "inventory_demand_ratio",
    "estimated_primary_cost_usd",
    "estimated_secondary_cost_usd",
    "air_freight_cost_usd",
]

CATEGORICAL_COLS = [
    "supplier_id",
    "supplier_name",
    "supplier_country",
    "product_id",
    "product_name",
    "category",
    "service_risk",
]


def load_raw_dataset(raw_path: Path) -> pd.DataFrame:
    """
    Loads the raw dataset as a read-only immutable source of truth.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at expected path: {raw_path}")
    df = pd.read_csv(raw_path)
    return df


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Identifies and imputes missing values.
    Uses median for numerical variables and mode for categorical variables.
    """
    df = df.copy()
    missing_report = df.isnull().sum().to_dict()
    
    # Impute numerical features with median
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # Impute categorical features with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            
    return df, missing_report


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Identifies and removes duplicate records using complete rows and business keys.
    """
    initial_count = len(df)
    
    # Check exact row duplicates
    exact_duplicates = int(df.duplicated().sum())
    df_clean = df.drop_duplicates(keep="first").copy()
    
    # Check business key duplicates (date, supplier_id, product_id)
    if set(["date", "supplier_id", "product_id"]).issubset(df_clean.columns):
        key_duplicates = int(df_clean.duplicated(subset=["date", "supplier_id", "product_id"]).sum())
        df_clean = df_clean.drop_duplicates(subset=["date", "supplier_id", "product_id"], keep="first")
    else:
        key_duplicates = 0
        
    final_count = len(df_clean)
    removed_count = initial_count - final_count
    
    dup_report = {
        "initial_records": initial_count,
        "exact_duplicates": exact_duplicates,
        "key_duplicates": key_duplicates,
        "records_removed": removed_count,
        "final_records": final_count,
    }
    return df_clean, dup_report


def standardize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes schema datatypes and ensures chronological sorting.
    """
    df = df.copy()
    
    # Date parsing & sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "supplier_id", "product_id"]).reset_index(drop=True)
    
    # Integer type conversions
    int_cols = [
        "demand_units",
        "inventory_units",
        "supplier_capacity_units",
        "planned_lead_time_days",
        "actual_lead_time_days",
        "delay_occurred",
        "delay_days",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].round().astype(int)
            
    # Float type conversions
    float_cols = [
        "supplier_reliability",
        "unit_cost_usd",
        "transport_cost_usd",
        "secondary_supplier_premium_pct",
        "inventory_demand_ratio",
        "estimated_primary_cost_usd",
        "estimated_secondary_cost_usd",
        "air_freight_cost_usd",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    # String type conversions
    str_cols = ["supplier_id", "supplier_name", "supplier_country", "product_id", "product_name", "category", "service_risk"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df


def validate_operational_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validates domain and business logic constraints on operational records.
    """
    df = df.copy()
    
    # 1. Supplier reliability in [0, 1]
    invalid_reliability = ((df["supplier_reliability"] < 0.0) | (df["supplier_reliability"] > 1.0)).sum()
    
    # 2. Demand > 0 and Capacity > 0 and Inventory >= 0
    invalid_demand_inv = ((df["demand_units"] <= 0) | (df["inventory_units"] < 0) | (df["supplier_capacity_units"] <= 0)).sum()
    
    # 3. Lead time constraints
    invalid_lead_times = (
        (df["planned_lead_time_days"] <= 0)
        | (df["actual_lead_time_days"] < df["planned_lead_time_days"])
        | (df["actual_lead_time_days"] != (df["planned_lead_time_days"] + df["delay_days"]))
    ).sum()
    
    # 4. Delay flag consistency
    invalid_delays = (
        ((df["delay_occurred"] == 0) & (df["delay_days"] != 0))
        | ((df["delay_occurred"] == 1) & (df["delay_days"] <= 0))
    ).sum()
    
    # 5. Cost constraints
    invalid_costs = (
        (df["unit_cost_usd"] <= 0)
        | (df["transport_cost_usd"] <= 0)
        | (df["air_freight_cost_usd"] <= 0)
        | (df["secondary_supplier_premium_pct"] < 0.05)
        | (df["secondary_supplier_premium_pct"] > 0.20)
    ).sum()
    
    rule_report = {
        "invalid_reliability": int(invalid_reliability),
        "invalid_demand_inv": int(invalid_demand_inv),
        "invalid_lead_times": int(invalid_lead_times),
        "invalid_delays": int(invalid_delays),
        "invalid_costs": int(invalid_costs),
    }
    
    return df, rule_report


def detect_and_flag_outliers(
    df: pd.DataFrame,
    multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Applies the Interquartile Range (IQR) method to detect statistical outliers
    and creates domain-specific outlier flags while preserving legitimate disruption events.
    """
    df = df.copy()
    outlier_counts = {}
    
    cols_to_check = [
        "demand_units",
        "inventory_units",
        "supplier_capacity_units",
        "planned_lead_time_days",
        "actual_lead_time_days",
        "delay_days",
        "unit_cost_usd",
        "transport_cost_usd",
        "air_freight_cost_usd",
    ]
    
    for col in cols_to_check:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (multiplier * iqr)
            upper_bound = q3 + (multiplier * iqr)
            
            is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_counts[col] = int(is_outlier.sum())
            
    # Domain-specific binary outlier indicators (0 = Normal, 1 = Potential Outlier)
    df["demand_outlier_flag"] = (
        (df["demand_units"] < df["demand_units"].quantile(0.01))
        | (df["demand_units"] > df["demand_units"].quantile(0.99))
    ).astype(int)
    
    df["inventory_outlier_flag"] = (
        (df["inventory_units"] < df["inventory_units"].quantile(0.01))
        | (df["inventory_units"] > df["inventory_units"].quantile(0.99))
    ).astype(int)
    
    df["cost_outlier_flag"] = (
        (df["unit_cost_usd"] > df["unit_cost_usd"].quantile(0.98))
        | (df["transport_cost_usd"] > df["transport_cost_usd"].quantile(0.98))
    ).astype(int)
    
    df["lead_time_outlier_flag"] = (
        (df["actual_lead_time_days"] > df["actual_lead_time_days"].quantile(0.98))
    ).astype(int)
    
    return df, outlier_counts


def generate_data_quality_report(
    df: pd.DataFrame,
    outlier_counts: Dict[str, int],
    report_output_path: Path,
) -> pd.DataFrame:
    """
    Generates a granular data quality report across all columns.
    """
    report_rows = []
    total_records = len(df)
    
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_pct = round((missing_count / total_records) * 100, 2)
        unique_vals = int(df[col].nunique())
        dtype_str = str(df[col].dtype)
        
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = round(float(df[col].min()), 2)
            max_val = round(float(df[col].max()), 2)
            mean_val = round(float(df[col].mean()), 2)
            median_val = round(float(df[col].median()), 2)
            outliers = outlier_counts.get(col, 0)
        else:
            min_val = np.nan
            max_val = np.nan
            mean_val = np.nan
            median_val = np.nan
            outliers = 0
            
        report_rows.append({
            "column": col,
            "data_type": dtype_str,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "unique_values": unique_vals,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "median": median_val,
            "outlier_count": outliers,
        })
        
    report_df = pd.DataFrame(report_rows)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_output_path, index=False)
    return report_df


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the complete Data Engineering and Feature Engineering pipeline.
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_path = base_dir / "data" / "raw" / "supply_prescript_raw_dataset.csv"
    processed_path = base_dir / "data" / "processed" / "supply_prescript_processed_dataset.csv"
    report_path = base_dir / "reports" / "data_quality_report.csv"
    config_path = base_dir / "config" / "features.yaml"
    
    print("=" * 65)
    print("  SUPPLY PRESCRIPT -- DATA & FEATURE ENGINEERING PIPELINE (DAY 3)")
    print("=" * 65)
    
    # 1. Load Raw Dataset
    print(f"\n[1/9] Loading raw dataset from: {raw_path}")
    df_raw = load_raw_dataset(raw_path)
    initial_records = len(df_raw)
    print(f"      Loaded {initial_records} records with {len(df_raw.columns)} raw columns.")
    
    # 2. Impute Missing Values
    print("\n[2/9] Evaluating and handling missing values...")
    df_imputed, missing_report = handle_missing_values(df_raw)
    total_missing = sum(missing_report.values())
    print(f"      Total missing values handled: {total_missing}")
    
    # 3. Duplicate Detection & Removal
    print("\n[3/9] Checking and removing duplicate observations...")
    df_dedup, dup_report = remove_duplicates(df_imputed)
    print(f"      Duplicates removed: {dup_report['records_removed']} (Final records: {dup_report['final_records']})")
    
    # 4. Standardize Data Types
    print("\n[4/9] Standardizing data types and sorting chronologically...")
    df_typed = standardize_data_types(df_dedup)
    
    # 5. Domain Rules Validation
    print("\n[5/9] Validating operational domain rules...")
    df_validated, rule_report = validate_operational_rules(df_typed)
    total_invalid_rules = sum(rule_report.values())
    print(f"      Rule violations detected: {total_invalid_rules}")
    
    # 6. Outlier Detection
    print("\n[6/9] Performing IQR outlier detection and generating risk flags...")
    df_outliers, outlier_counts = detect_and_flag_outliers(df_validated, multiplier=1.5)
    total_outliers = sum(outlier_counts.values())
    print(f"      Statistical outliers flagged: {total_outliers}")
    
    # 7. Feature Engineering
    print("\n[7/9] Engineering domain features and strictly past-only rolling metrics...")
    df_features = engineer_all_features(df_outliers)
    print(f"      Engineered dataset contains {len(df_features.columns)} total columns.")
    
    # 8. Data Leakage Check
    print("\n[8/9] Verifying strict pre-event vs post-event feature isolation...")
    pre_event_cols = [
        "demand_units", "inventory_units", "supplier_capacity_units",
        "supplier_reliability", "planned_lead_time_days", "unit_cost_usd",
        "transport_cost_usd", "secondary_supplier_premium_pct",
        "inventory_coverage_ratio", "inventory_gap_units", "capacity_utilization",
        "supplier_risk_score", "demand_pressure", "shortage_flag",
        "previous_delay_days", "previous_delay_flag", "rolling_average_delay",
        "rolling_average_demand", "rolling_average_inventory",
    ]
    is_leakage_free, violations = verify_no_data_leakage(pre_event_cols)
    if is_leakage_free:
        print("      PASS: No post-event leakage detected in pre-event feature set.")
    else:
        print(f"      FAIL: Target leakage detected in features: {violations}")
        
    # 9. Quality Report & Export Processed Dataset
    print(f"\n[9/9] Generating data quality report at: {report_path}")
    quality_df = generate_data_quality_report(df_features, outlier_counts, report_path)
    
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"      Saving processed dataset to: {processed_path}")
    df_features.to_csv(processed_path, index=False)
    
    # Final Pipeline Summary
    delay_count = int((df_features["delay_occurred"] == 1).sum())
    delay_pct = (delay_count / len(df_features)) * 100
    
    print("\n" + "=" * 65)
    print("                    DATA PIPELINE EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Initial Records          : {initial_records}")
    print(f"Final Processed Records  : {len(df_features)}")
    print(f"Duplicates Removed       : {dup_report['records_removed']}")
    print(f"Missing Values Handled   : {total_missing}")
    print(f"Outliers Flagged         : {total_outliers}")
    print(f"Invalid Records          : {total_invalid_rules}")
    print(f"Final Feature Count      : {len(df_features.columns)}")
    print(f"Target Distribution      : {delay_count} delayed ({delay_pct:.1f}%), {len(df_features) - delay_count} on-time ({100 - delay_pct:.1f}%)")
    print(f"Data Leakage Verification: {'PASSED' if is_leakage_free else 'FAILED'}")
    print(f"Processed Dataset Path   : {processed_path}")
    print(f"Quality Report Path      : {report_path}")
    print("=" * 65 + "\n")
    
    return df_features, quality_df


if __name__ == "__main__":
    run_pipeline()
