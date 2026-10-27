"""
Supply Prescript - Data Preparation & Feature Engineering Pipeline
Author: Prasanna (ML & Optimization Lead)
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Data directory path
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "supply_chain_data.csv")

NUMERICAL_FEATURES = [
    "lead_time_days",
    "shipping_cost",
    "inventory_level",
    "order_quantity",
    "supplier_delay_rate",
    "route_distance_km",
    "weather_risk_index",
    "congestion_index",
    "lead_time_to_distance_ratio",
    "inventory_coverage_ratio",
    "risk_multiplier",
]

CATEGORICAL_FEATURES = [
    "shipping_mode",
    "product_category",
    "supplier_tier",
]

ALL_MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def generate_synthetic_supply_chain_data(
    n_samples: int = 3500, random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic, statistically grounded supply-chain dataset
    mirroring actual logistics dynamics (lead times, costs, disruptions, delay durations).
    """
    np.random.seed(random_state)

    suppliers = [
        ("GlobalTech Micro", "Tier-1", 0.12),
        ("Shenzhen Silicon Ltd", "Tier-2", 0.28),
        ("Apex Auto Components", "Tier-1", 0.08),
        ("Bavaria Logistics Parts", "Tier-1", 0.05),
        ("TransPacific Freight", "Tier-3", 0.42),
        ("Delta Industrial Supplies", "Tier-2", 0.22),
        ("Nordic Raw Materials", "Tier-1", 0.09),
        ("FastTrack Assembly Co", "Tier-2", 0.31),
    ]

    product_categories = [
        "Semiconductors & Electronics",
        "Automotive Assemblies",
        "Industrial Machinery",
        "Pharmaceuticals",
        "Consumer Electronics",
    ]

    shipping_modes = ["Standard Ground", "Air Freight", "Ocean Intermodal", "Express Carrier"]
    origins = ["Shenzhen, CN", "Munich, DE", "Tokyo, JP", "Chicago, US", "Taipei, TW", "Busan, KR"]
    destinations = ["Los Angeles, US", "Rotterdam, NL", "Dallas, US", "Frankfurt, DE", "Singapore, SG"]

    data = []

    for i in range(1, n_samples + 1):
        sup_name, sup_tier, base_delay_rate = suppliers[np.random.choice(len(suppliers))]
        prod_cat = np.random.choice(product_categories)
        mode = np.random.choice(shipping_modes, p=[0.40, 0.25, 0.25, 0.10])
        origin = np.random.choice(origins)
        dest = np.random.choice(destinations)

        # Baseline distances (km)
        if "Ocean" in mode:
            dist = np.random.uniform(7000, 14000)
            base_lead = np.random.uniform(18, 35)
            cost_per_unit = np.random.uniform(4, 12)
        elif "Air" in mode:
            dist = np.random.uniform(4000, 12000)
            base_lead = np.random.uniform(3, 8)
            cost_per_unit = np.random.uniform(25, 60)
        elif "Express" in mode:
            dist = np.random.uniform(2000, 8000)
            base_lead = np.random.uniform(2, 5)
            cost_per_unit = np.random.uniform(35, 75)
        else:  # Standard Ground
            dist = np.random.uniform(500, 3500)
            base_lead = np.random.uniform(5, 14)
            cost_per_unit = np.random.uniform(8, 20)

        order_qty = int(np.random.choice([50, 100, 250, 500, 1000, 2000], p=[0.1, 0.25, 0.3, 0.2, 0.1, 0.05]))
        shipping_cost = round(order_qty * cost_per_unit + np.random.normal(200, 50), 2)
        inventory_level = int(np.random.uniform(50, 1500))

        # Supplier historical delay rate (jittered)
        sup_delay_jitter = np.clip(base_delay_rate + np.random.normal(0, 0.04), 0.01, 0.75)
        
        # Environmental / route risk factors
        weather_risk = round(float(np.random.beta(2, 4)), 3)  # 0 to 1
        congestion_index = round(float(np.random.beta(2, 3)), 3)  # 0 to 1

        # Real-world physics-based logistics risk formula
        route_stress = (dist / 14000.0) * 0.20
        supplier_risk = sup_delay_jitter * 0.40
        environmental_risk = (weather_risk * 0.22) + (congestion_index * 0.28)
        inventory_buffer_effect = max(0.0, (1.0 - inventory_level / (order_qty * 3.0 + 1e-5))) * 0.10

        composite_risk = supplier_risk + environmental_risk + route_stress + inventory_buffer_effect

        # Ground truth delay probability & label
        is_delayed = 1 if composite_risk > 0.42 else 0

        # Delay duration calculation (days)
        if is_delayed:
            base_delay = (composite_risk - 0.42) * 22.0 + 3.0
            noise = np.random.normal(0, 1.2)
            delay_days = round(float(max(1.0, base_delay + noise)), 1)
        else:
            delay_days = 0.0

        actual_lead_time = round(base_lead + delay_days, 1)

        # Feature engineering directly
        lead_time_to_dist = round(base_lead / (dist / 1000 + 1e-5), 4)
        inv_coverage = round(inventory_level / (order_qty + 1e-5), 4)
        risk_mult = round(weather_risk * congestion_index * (1 + sup_delay_jitter), 4)

        data.append({
            "shipment_id": f"SHP-{10000 + i}",
            "supplier_name": sup_name,
            "supplier_tier": sup_tier,
            "product_category": prod_cat,
            "origin": origin,
            "destination": dest,
            "shipping_mode": mode,
            "lead_time_days": round(base_lead, 1),
            "shipping_cost": shipping_cost,
            "inventory_level": inventory_level,
            "order_quantity": order_qty,
            "supplier_delay_rate": round(sup_delay_jitter, 3),
            "route_distance_km": round(dist, 1),
            "weather_risk_index": weather_risk,
            "congestion_index": congestion_index,
            "lead_time_to_distance_ratio": lead_time_to_dist,
            "inventory_coverage_ratio": inv_coverage,
            "risk_multiplier": risk_mult,
            "actual_lead_time_days": actual_lead_time,
            "delay_days": delay_days,
            "is_delayed": is_delayed,
        })

    df = pd.DataFrame(data)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"Generated and saved {len(df)} supply chain records to {DATA_FILE}")
    return df


def load_data(data_path: str = DATA_FILE) -> pd.DataFrame:
    """Loads existing dataset or generates fresh realistic dataset if missing."""
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Generating synthetic dataset...")
        return generate_synthetic_supply_chain_data()
    return pd.read_csv(data_path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes engineered domain features for any given dataframe."""
    df = df.copy()
    if "lead_time_to_distance_ratio" not in df.columns:
        df["lead_time_to_distance_ratio"] = df["lead_time_days"] / (df["route_distance_km"] / 1000 + 1e-5)
    if "inventory_coverage_ratio" not in df.columns:
        df["inventory_coverage_ratio"] = df["inventory_level"] / (df["order_quantity"] + 1e-5)
    if "risk_multiplier" not in df.columns:
        df["risk_multiplier"] = df["weather_risk_index"] * df["congestion_index"] * (1 + df["supplier_delay_rate"])
    if "supplier_tier" not in df.columns and "supplier_name" in df.columns:
        tier_map = {
            "GlobalTech Micro": "Tier-1",
            "Shenzhen Silicon Ltd": "Tier-2",
            "Apex Auto Components": "Tier-1",
            "Bavaria Logistics Parts": "Tier-1",
            "TransPacific Freight": "Tier-3",
            "Delta Industrial Supplies": "Tier-2",
            "Nordic Raw Materials": "Tier-1",
            "FastTrack Assembly Co": "Tier-2",
        }
        df["supplier_tier"] = df["supplier_name"].map(lambda s: tier_map.get(s, "Tier-2"))
    return df


def create_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for scaling and one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def prepare_datasets(
    test_size: float = 0.2, random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepares train and test splits for both classification (is_delayed)
    and regression (delay_days).
    """
    df = load_data()
    df = engineer_features(df)

    X = df[ALL_MODEL_FEATURES]
    y_cls = df["is_delayed"]
    y_reg = df["delay_days"]

    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        X, y_cls, y_reg, test_size=test_size, random_state=random_state, stratify=y_cls
    )

    preprocessor = create_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERICAL_FEATURES + encoded_cat_names

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train": X_train_trans,
        "X_test": X_test_trans,
        "y_train_cls": y_train_cls,
        "y_test_cls": y_test_cls,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    generate_synthetic_supply_chain_data()
    data = prepare_datasets()
    print(f"Data prepared successfully: {data['X_train'].shape[0]} train rows, {data['X_test'].shape[0]} test rows.")
    print(f"Total features after encoding: {len(data['feature_names'])}")
