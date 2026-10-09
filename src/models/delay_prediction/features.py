"""Feature engineering, validation, and preprocessing pipeline for delay prediction.

Guarantees:
- Strict pre-event feature isolation to prevent target leakage
- Safe imputation and encoding for numerical and categorical features
- Chronological time-aware data splitting
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.delay_prediction.config import DelayPredictionConfig, default_config

logger = logging.getLogger(__name__)


# Column alias mapping to support varied schemas
COLUMN_ALIASES: Dict[str, List[str]] = {
    "unit_cost_usd": ["unit_cost", "unit_cost_usd"],
    "transport_cost_usd": ["transport_cost", "transport_cost_usd"],
    "secondary_supplier_premium_pct": [
        "secondary_supplier_premium",
        "secondary_supplier_premium_pct",
    ],
    "category": ["category", "product_category"],
    "supplier_reliability": ["supplier_reliability", "reliability_score"],
    "air_freight_cost_usd": ["air_freight_cost", "air_freight_cost_usd", "air_freight_premium"],
}


def validate_and_normalize_target(
    df: pd.DataFrame, target_column: str = "delay_occurred"
) -> pd.Series:
    """Validate and normalize the binary target column.

    Args:
        df: Input DataFrame.
        target_column: Target column name.

    Returns:
        Clean binary integer pd.Series (0 or 1).

    Raises:
        ValueError: If target column is missing or values cannot be normalized to binary {0, 1}.
    """
    if df is None or df.empty:
        raise ValueError("Input dataset is empty or None.")

    if target_column not in df.columns:
        raise ValueError(
            f"Required target column '{target_column}' is missing from the dataset. "
            f"Available columns: {list(df.columns)}"
        )

    target_series = df[target_column]

    if target_series.isnull().all():
        raise ValueError(f"Target column '{target_column}' contains only NULL/NaN values.")

    # Convert boolean or strings to integer 0/1
    if target_series.dtype == bool:
        normalized = target_series.astype(int)
    elif pd.api.types.is_numeric_dtype(target_series):
        # Check unique non-null values
        unique_vals = set(target_series.dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Target column '{target_column}' contains non-binary values: {unique_vals}. "
                "Target must be binary (0 = on-time, 1 = delayed)."
            )
        normalized = target_series.astype(int)
    else:
        # String representations
        str_map = {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0, "delayed": 1, "on_time": 0}
        normalized = (
            target_series.astype(str)
            .str.strip()
            .str.lower()
            .map(str_map)
        )
        if normalized.isnull().any():
            invalid_vals = target_series[normalized.isnull()].unique()
            raise ValueError(
                f"Target column '{target_column}' contains unmapped values: {invalid_vals}."
            )
        normalized = normalized.astype(int)

    # Verify both classes or class distribution
    counts = normalized.value_counts().to_dict()
    logger.info(f"Normalized target '{target_column}' class distribution: {counts}")
    return normalized


def resolve_feature_column_names(
    df_columns: List[str], requested_features: List[str]
) -> List[str]:
    """Resolve requested features against DataFrame columns considering aliases.

    Args:
        df_columns: List of columns present in DataFrame.
        requested_features: List of feature names to find.

    Returns:
        List of actually matching column names present in the DataFrame.
    """
    available_cols = set(df_columns)
    matched_features: List[str] = []

    for feat in requested_features:
        if feat in available_cols:
            matched_features.append(feat)
            continue
        # Check aliases
        found = False
        if feat in COLUMN_ALIASES:
            for alias in COLUMN_ALIASES[feat]:
                if alias in available_cols:
                    matched_features.append(alias)
                    found = True
                    break
        if not found:
            # Check reverse alias mapping
            for canonical, aliases in COLUMN_ALIASES.items():
                if feat in aliases:
                    for alias in aliases:
                        if alias in available_cols and alias not in matched_features:
                            matched_features.append(alias)
                            found = True
                            break
                    if found:
                        break

    return sorted(list(set(matched_features)))


def select_pre_event_features(
    df: pd.DataFrame, config: DelayPredictionConfig = default_config
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Select pre-event features while strictly excluding post-event leakage features.

    Args:
        df: Input DataFrame.
        config: Configuration containing approved feature lists and leakage blacklist.

    Returns:
        Tuple of (X DataFrame, list of numerical features, list of categorical features).

    Raises:
        ValueError: If no valid approved features are found.
    """
    df_cols = list(df.columns)

    # 1. Resolve numerical features
    num_cols = resolve_feature_column_names(df_cols, config.approved_numerical_features)

    # 2. Resolve categorical features
    cat_cols = resolve_feature_column_names(df_cols, config.approved_categorical_features)

    # 3. Strict leakage check: remove any column in leakage list
    leakage_set = set(config.leakage_columns)
    # Also add variations
    leakage_set.update([col.lower() for col in config.leakage_columns])

    num_cols = [c for c in num_cols if c.lower() not in leakage_set]
    cat_cols = [c for c in cat_cols if c.lower() not in leakage_set]

    all_features = num_cols + cat_cols
    if not all_features:
        raise ValueError(
            "No valid pre-event features found in DataFrame after filtering leakage columns. "
            f"Available DataFrame columns: {df_cols}"
        )

    logger.info(
        f"Selected {len(num_cols)} numerical features and {len(cat_cols)} categorical features. "
        f"Total pre-event features: {len(all_features)}"
    )

    X = df[all_features].copy()
    return X, num_cols, cat_cols


def build_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Build a Scikit-learn ColumnTransformer for preprocessing.

    - Numerical: Median imputation + StandardScaler
    - Categorical: Constant missing imputation + OneHotEncoder(handle_unknown='ignore')

    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.

    Returns:
        Configured ColumnTransformer instance.
    """
    transformers = []

    if numerical_features:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipeline, numerical_features))

    if categorical_features:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    numerical_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    """Extract transformed feature names from fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        names = list(numerical_features)
        if categorical_features and "cat" in preprocessor.named_transformers_:
            cat_step = preprocessor.named_transformers_["cat"]
            if hasattr(cat_step, "named_steps") and "onehot" in cat_step.named_steps:
                ohe = cat_step.named_steps["onehot"]
                if hasattr(ohe, "get_feature_names_out"):
                    names.extend(list(ohe.get_feature_names_out(categorical_features)))
        return names


def split_data_chronologically(
    df: pd.DataFrame,
    config: DelayPredictionConfig = default_config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split dataset chronologically into Train (64%), Validation (16%), and Test (20%).

    Requirements:
    1. Sort records chronologically using the available event or shipment date.
    2. Reserve the latest 20% of eligible records as the test set.
    3. Use the earlier records for training and validation.
    4. If no reliable timestamp exists, fallback to stratified train-test split.

    Args:
        df: Input DataFrame.
        config: Configuration parameters.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # 1. Normalize target
    y_full = validate_and_normalize_target(df, config.target_column)

    # 2. Select pre-event features
    X_full, num_features, cat_features = select_pre_event_features(df, config)

    # 3. Check for reliable date column
    has_date = (
        config.date_column in df.columns
        and pd.to_datetime(df[config.date_column], errors="coerce").notnull().all()
    )

    if has_date:
        logger.info(f"Performing chronological train/val/test split based on '{config.date_column}'.")
        sorted_indices = (
            pd.to_datetime(df[config.date_column]).sort_values(kind="mergesort").index
        )
        n_total = len(df)
        n_test = max(1, int(n_total * config.test_size))
        n_train_val = n_total - n_test
        n_val = max(1, int(n_train_val * config.val_size))
        n_train = n_train_val - n_val

        train_idx = sorted_indices[:n_train]
        val_idx = sorted_indices[n_train:n_train_val]
        test_idx = sorted_indices[n_train_val:]

        X_train, y_train = X_full.loc[train_idx].copy(), y_full.loc[train_idx].copy()
        X_val, y_val = X_full.loc[val_idx].copy(), y_full.loc[val_idx].copy()
        X_test, y_test = X_full.loc[test_idx].copy(), y_full.loc[test_idx].copy()

        logger.info(
            f"Chronological split sizes -> Train: {len(X_train)} (dates: {df.loc[train_idx, config.date_column].min()} to {df.loc[train_idx, config.date_column].max()}), "
            f"Val: {len(X_val)} (dates: {df.loc[val_idx, config.date_column].min()} to {df.loc[val_idx, config.date_column].max()}), "
            f"Test: {len(X_test)} (dates: {df.loc[test_idx, config.date_column].min()} to {df.loc[test_idx, config.date_column].max()})"
        )
    else:
        logger.warning(
            f"Date column '{config.date_column}' not found or invalid. Falling back to stratified split."
        )
        # Stratified train/test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_full,
            y_full,
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=y_full if y_full.nunique() > 1 else None,
        )
        val_relative_size = config.val_size / (1.0 - config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_relative_size,
            random_state=config.random_seed,
            stratify=y_train_val if y_train_val.nunique() > 1 else None,
        )

    return X_train, X_val, X_test, y_train, y_val, y_test
