"""Batch Risk Engine Execution and Reporting Workflow.

Closed-Loop Prescriptive Analytics (Day 8):
Shipment Data -> Predictive Probability -> Operational Risk Assessment ->
Risk Classification -> Risk Alerts -> Optimization Input.

Pipeline:
1. Load & validate configuration (config/risk_config.yaml)
2. Load processed dataset (data/processed/supply_prescript_processed_dataset.csv)
3. Load Day 6 trained model pipeline (zero retraining)
4. Generate delay probabilities & predicted delays (Day 7 threshold: 0.65)
5. Calculate past-only historical supplier delay rates (leakage-free)
6. Calculate operational risk dimensions (supplier, inventory, capacity, service)
7. Calculate composite risk scores & assign risk levels
8. Generate machine & human-readable risk reasons
9. Generate structured alerts for HIGH and CRITICAL shipments
10. Export reports/risk_engine/risk_assessments.csv and high_risk_shipments.csv
11. Generate professional visualization charts
12. Generate comprehensive summary report (risk_assessment_summary.md)
13. Optional database persistence with transaction rollback
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.delay_prediction.config import default_config as delay_model_config
from src.models.delay_prediction.features import select_pre_event_features
from src.risk_engine.alerts import RiskAlertEngine
from src.risk_engine.config import RiskEngineConfig, load_risk_config
from src.risk_engine.operational_risk import compute_past_supplier_delay_rates
from src.risk_engine.persistence import persist_risk_assessments
from src.risk_engine.risk_scoring import assess_single_shipment
from src.risk_engine.schemas import (
    RiskAlert,
    RiskAssessmentInput,
    RiskAssessmentResult,
    RiskEngineSummary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("risk_engine")


def load_model_pipeline(
    model_path: Path, metadata_path: Path
) -> Tuple[Any, Dict[str, Any]]:
    """Load trained Day 6 scikit-learn pipeline and metadata with zero retraining.

    Args:
        model_path: Path to delay_classifier.joblib
        metadata_path: Path to model_metadata.json

    Returns:
        Tuple of (pipeline_object, metadata_dict)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model artifact not found at: {model_path}")

    logger.info(f"Loading trained delay prediction pipeline from {model_path}...")
    pipeline = joblib.load(model_path)

    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        logger.warning(f"Model metadata not found at {metadata_path}; using fallback defaults.")
        metadata = {
            "model_name": "supply_delay_classifier",
            "model_version": "1.0.0",
            "selected_algorithm": "LogisticRegression",
        }

    return pipeline, metadata


def generate_model_probabilities(
    df: pd.DataFrame,
    pipeline: Any,
) -> np.ndarray:
    """Generate pre-event delay probabilities with fitted pipeline (no retraining).

    Args:
        df: Input DataFrame containing shipment features.
        pipeline: Serialized Day 6 pipeline.

    Returns:
        np.ndarray of delay probabilities in [0.0, 1.0].
    """
    X, _, _ = select_pre_event_features(df, delay_model_config)

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)[:, 1]
    else:
        probs = pipeline.predict(X).astype(float)

    return np.clip(probs, 0.0, 1.0)


def evaluate_batch_risk(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    config: RiskEngineConfig,
    model_version: str = "1.0.0",
) -> Tuple[List[RiskAssessmentResult], int, int]:
    """Execute risk assessment for all records in the batch.

    Args:
        df: Shipment DataFrame.
        probabilities: Corresponding delay probabilities.
        config: RiskEngineConfig.
        model_version: Serialized model version string.

    Returns:
        Tuple of (assessments_list, skipped_count, failed_count).
    """
    logger.info("Computing past-only historical supplier delay rates (leakage-free)...")
    past_delay_rates = compute_past_supplier_delay_rates(df)

    results: List[RiskAssessmentResult] = []
    skipped = 0
    failed = 0

    logger.info(f"Evaluating multi-dimensional risk scores for {len(df)} shipment events...")
    for idx, row in df.iterrows():
        try:
            event_id = str(row.get("event_id", ""))
            if not event_id:
                # Synthesize standard event_id if not in column
                d_str = str(row.get("date", "2025-01-01"))
                sup_id = str(row.get("supplier_id", "SUP"))
                prod_id = str(row.get("product_id", "P"))
                event_id = f"EVT_{d_str}_{sup_id}_{prod_id}"

            prob = float(probabilities[idx])
            reliability = float(row.get("supplier_reliability", 0.85))
            hist_delay = float(past_delay_rates.iloc[idx]) if idx in past_delay_rates.index else 0.0

            cov_ratio = row.get("inventory_coverage_ratio")
            cov_ratio_val = float(cov_ratio) if pd.notnull(cov_ratio) else None

            inv_units = row.get("inventory_units")
            inv_units_val = float(inv_units) if pd.notnull(inv_units) else None

            dem_units = row.get("demand_units")
            dem_units_val = float(dem_units) if pd.notnull(dem_units) else None

            shortage_flag = row.get("shortage_flag")
            shortage_flag_val = int(shortage_flag) if pd.notnull(shortage_flag) else None

            cap_units = row.get("supplier_capacity_units")
            cap_units_val = float(cap_units) if pd.notnull(cap_units) else None

            cap_util = row.get("capacity_utilization")
            cap_util_val = float(cap_util) if pd.notnull(cap_util) else None

            # Product criticality: check for product_criticality or criticality_level
            # (If missing, calculate_service_risk cleanly marks service dimension as unavailable)
            prod_crit = row.get("product_criticality", row.get("criticality_level", None))
            prod_crit_val = str(prod_crit) if pd.notnull(prod_crit) else None

            inp = RiskAssessmentInput(
                event_id=event_id,
                delay_probability=prob,
                supplier_reliability=reliability,
                historical_delay_rate=hist_delay,
                inventory_coverage_ratio=cov_ratio_val,
                inventory_units=inv_units_val,
                demand_units=dem_units_val,
                shortage_flag=shortage_flag_val,
                supplier_capacity_units=cap_units_val,
                capacity_utilization=cap_util_val,
                product_criticality=prod_crit_val,
                category=str(row.get("category", "")),
                supplier_id=str(row.get("supplier_id", "")),
                supplier_name=str(row.get("supplier_name", "")),
                product_id=str(row.get("product_id", "")),
                product_name=str(row.get("product_name", "")),
                date=str(row.get("date", "")),
            )

            res = assess_single_shipment(inp, config=config, model_version=model_version)
            results.append(res)
        except Exception as exc:
            logger.warning(f"Error evaluating risk for record index {idx}: {exc}")
            failed += 1

    return results, skipped, failed


def export_risk_reports(
    assessments: List[RiskAssessmentResult],
    original_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Export risk_assessments.csv and high_risk_shipments.csv.

    Args:
        assessments: List of RiskAssessmentResult models.
        original_df: Original DataFrame for operational context.
        output_dir: Target output directory.

    Returns:
        Tuple of (assessments_csv_path, high_risk_csv_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for a in assessments:
        records.append({
            "event_id": a.event_id,
            "delay_probability": a.delay_probability,
            "predicted_delay": a.predicted_delay,
            "delay_risk_score": a.delay_risk_score,
            "supplier_risk_score": a.supplier_risk_score,
            "inventory_risk_score": a.inventory_risk_score,
            "capacity_risk_score": a.capacity_risk_score,
            "service_risk_score": a.service_risk_score,
            "composite_risk_score": a.composite_risk_score,
            "risk_level": a.risk_level,
            "risk_reasons": "; ".join(a.risk_reasons),
            "risk_reason_descriptions": " | ".join(a.risk_reason_descriptions),
            "model_version": a.model_version,
            "configuration_version": a.configuration_version,
            "assessment_timestamp": a.assessment_timestamp,
        })

    assessments_df = pd.DataFrame(records)
    assessments_csv = output_dir / "risk_assessments.csv"
    assessments_df.to_csv(assessments_csv, index=False)
    logger.info(f"Exported {len(assessments_df)} risk assessments to {assessments_csv}")

    # Merge operational context for high risk shipments
    high_risk_mask = assessments_df["risk_level"].isin(["HIGH", "CRITICAL"])
    high_risk_df = assessments_df[high_risk_mask].copy()

    # Add operational features from original_df if aligned
    if len(original_df) == len(assessments_df):
        context_cols = [
            "date", "supplier_id", "supplier_name", "product_id", "product_name",
            "category", "demand_units", "inventory_units", "supplier_capacity_units",
            "capacity_utilization", "inventory_coverage_ratio", "shortage_flag"
        ]
        available_context = [c for c in context_cols if c in original_df.columns]
        for col in available_context:
            high_risk_df[col] = original_df.loc[high_risk_df.index, col].values

    high_risk_csv = output_dir / "high_risk_shipments.csv"
    high_risk_df.to_csv(high_risk_csv, index=False)
    logger.info(f"Exported {len(high_risk_df)} HIGH/CRITICAL shipments to {high_risk_csv}")

    return assessments_csv, high_risk_csv


def generate_visualizations(
    assessments_df: pd.DataFrame,
    original_df: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Generate high-quality visualizations for risk distribution and supplier exposure.

    1. Risk-level distribution (risk_distribution.png)
    2. Supplier risk comparison (supplier_risk_comparison.png)
    3. Product category high-risk counts (product_risk_distribution.png)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_charts: List[Path] = []

    sns.set_theme(style="whitegrid")
    palette = {"LOW": "#2a9d8f", "MEDIUM": "#e9c46a", "HIGH": "#f4a261", "CRITICAL": "#e76f51"}

    # Chart 1: Risk Level Distribution & Score Spread
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Counts by risk level
    level_counts = assessments_df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
    total_records = len(assessments_df)
    bar_colors = [palette.get(lvl, "#457b9d") for lvl in level_counts.index]
    
    bars = axes[0].bar(level_counts.index, level_counts.values, color=bar_colors, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_records) * 100
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + max(1, total_records * 0.01),
            f"{int(h)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axes[0].set_title(f"Shipment Risk-Level Distribution (N = {total_records})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Operational Risk Level", fontsize=10)
    axes[0].set_ylabel("Shipment Event Count", fontsize=10)
    axes[0].set_ylim(0, max(level_counts.values) * 1.15)

    # Right: Composite Risk Score Histogram / KDE
    sns.histplot(
        assessments_df["composite_risk_score"],
        bins=20,
        kde=True,
        ax=axes[1],
        color="#264653",
        edgecolor="black",
    )
    axes[1].axvline(0.30, color="#2a9d8f", linestyle="--", label="Low/Medium (0.30)")
    axes[1].axvline(0.60, color="#f4a261", linestyle="--", label="Medium/High (0.60)")
    axes[1].axvline(0.80, color="#e76f51", linestyle="--", label="High/Critical (0.80)")
    axes[1].set_title("Composite Risk Score Distribution & Thresholds", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Composite Risk Score", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    chart1_path = output_dir / "risk_distribution.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    generated_charts.append(chart1_path)

    # Chart 2: Supplier Risk Comparison
    # Group by supplier
    merged_df = assessments_df.copy()
    if "supplier_id" in original_df.columns:
        merged_df["supplier_id"] = original_df["supplier_id"].values
        merged_df["supplier_name"] = original_df["supplier_name"].values if "supplier_name" in original_df.columns else merged_df["supplier_id"]
    
    if "supplier_name" in merged_df.columns:
        sup_summary = merged_df.groupby("supplier_name").agg(
            avg_composite=("composite_risk_score", "mean"),
            avg_supplier_risk=("supplier_risk_score", "mean"),
            avg_delay_risk=("delay_risk_score", "mean"),
            high_risk_count=("risk_level", lambda x: (x.isin(["HIGH", "CRITICAL"])).sum()),
            total_shipments=("composite_risk_score", "count"),
        ).reset_index().sort_values(by="avg_composite", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(sup_summary))
        width = 0.35

        ax.bar(x - width/2, sup_summary["avg_composite"], width, label="Mean Composite Risk", color="#e76f51", edgecolor="black")
        ax.bar(x + width/2, sup_summary["avg_supplier_risk"], width, label="Mean Supplier Risk (1-Rel)", color="#457b9d", edgecolor="black")

        ax.set_ylabel("Normalized Risk Score", fontsize=11)
        ax.set_title(f"Supplier Risk Comparison Across Assessed Cohort (N = {len(sup_summary)} Suppliers)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sup_summary["supplier_name"], rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)

        # Annotate high-risk shipment count
        for i, row in sup_summary.iterrows():
            ax.text(
                i,
                row["avg_composite"] + 0.03,
                f"HR: {int(row['high_risk_count'])}/{int(row['total_shipments'])}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1d3557",
                fontweight="bold",
            )

        plt.tight_layout()
        chart2_path = output_dir / "supplier_risk_comparison.png"
        plt.savefig(chart2_path, dpi=300)
        plt.close()
        generated_charts.append(chart2_path)

    # Chart 3: High-Risk Shipments by Product Category
    if "category" in original_df.columns:
        merged_df["category"] = original_df["category"].values
        cat_high_risk = merged_df[merged_df["risk_level"].isin(["HIGH", "CRITICAL"])]["category"].value_counts()
        cat_total = merged_df["category"].value_counts()

        fig, ax = plt.subplots(figsize=(10, 5))
        cats = cat_total.index
        cat_hr_vals = [cat_high_risk.get(c, 0) for c in cats]
        cat_tot_vals = [cat_total[c] for c in cats]

        x = np.arange(len(cats))
        ax.bar(x, cat_tot_vals, label="Total Shipments", color="#a8dadc", edgecolor="black")
        bars_hr = ax.bar(x, cat_hr_vals, label="High / Critical Risk", color="#e63946", edgecolor="black")

        for idx_b, bar in enumerate(bars_hr):
            h = bar.get_height()
            tot = cat_tot_vals[idx_b]
            pct = (h / tot * 100) if tot > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1,
                f"{h} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_ylabel("Shipment Volume", fontsize=11)
        ax.set_title("High and Critical Risk Disruption Exposure by Product Category", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(cat_tot_vals) * 1.15)

        plt.tight_layout()
        chart3_path = output_dir / "product_risk_distribution.png"
        plt.savefig(chart3_path, dpi=300)
        plt.close()
        generated_charts.append(chart3_path)

    return generated_charts


def generate_markdown_summary(
    summary: RiskEngineSummary,
    assessments_df: pd.DataFrame,
    high_risk_df: pd.DataFrame,
    alerts: List[RiskAlert],
    config: RiskEngineConfig,
    output_path: Path,
) -> Path:
    """Generate comprehensive, auditable markdown summary report."""
    total = summary.total_shipments_assessed
    level_counts = summary.risk_level_counts
    crit_count = level_counts.get("CRITICAL", 0)
    high_count = level_counts.get("HIGH", 0)
    med_count = level_counts.get("MEDIUM", 0)
    low_count = level_counts.get("LOW", 0)

    # Top triggered reasons
    all_reasons = []
    for r in assessments_df["risk_reasons"]:
        if pd.notnull(r) and isinstance(r, str):
            all_reasons.extend([x.strip() for x in r.split(";") if x.strip()])
    reason_series = pd.Series(all_reasons).value_counts()

    # Top high risk suppliers
    top_suppliers_md = ""
    if "supplier_name" in high_risk_df.columns:
        sup_counts = high_risk_df["supplier_name"].value_counts().head(5)
        top_suppliers_md = "\n".join(
            [f"- **{sup}**: {cnt} high/critical risk shipments" for sup, cnt in sup_counts.items()]
        )
    else:
        top_suppliers_md = "- Supplier details available in `high_risk_shipments.csv`"

    content = f"""# Day 8 Risk Assessment Summary Report
## Supply Prescript: Closed-Loop Prescriptive Analytics

---

### 1. Executive Summary
The **Supply Prescript Risk Engine** combines calibrated probabilistic delay predictions from Day 6 with deterministic operational domain rules across suppliers, warehouse inventory, manufacturing capacity, and customer service exposure. 

Across **{total} shipment events** evaluated:
- **Critical Risk**: **{crit_count} shipments** ({crit_count/total*100:.1f}%) requiring priority mitigation
- **High Risk**: **{high_count} shipments** ({high_count/total*100:.1f}%) flagged with warning disruption alerts
- **Medium Risk**: **{med_count} shipments** ({med_count/total*100:.1f}%) monitored under standard protocols
- **Low Risk**: **{low_count} shipments** ({low_count/total*100:.1f}%) clear for normal logistics execution
- **Total Disruptions Prioritized for Day 9 Optimization**: **{crit_count + high_count} shipments**

---

### 2. Dataset and Record Counts
- **Total Records Assessed**: {total}
- **Successfully Processed**: {summary.processed_count}
- **Skipped Records**: {summary.skipped_count}
- **Failed Records**: {summary.failed_count}
- **Target Variable (Pre-Event Separation)**: Pre-event operational indicators strictly isolated from post-event outcomes (zero target leakage).

---

### 3. Predictive Model Version
- **Model Name**: `supply_delay_classifier`
- **Model Version**: `{summary.model_version}`
- **Algorithm**: `LogisticRegression (class_weight='balanced')`
- **Model Retraining Policy**: **Strict Zero-Retraining**. Day 6 serialized pipeline loaded directly.
- **Classification Decision Threshold**: `{summary.classification_threshold:.2f}` (validated cost-optimal operating point from Day 7 validation).

---

### 4. Risk Configuration Version
- **Configuration Version**: `{summary.configuration_version}`
- **Approved Release Date**: `{config.approved_date}`
- **Dimension Weights**:
  - Delay Risk Weight: **{config.scoring.delay_weight:.2f}**
  - Supplier Risk Weight: **{config.scoring.supplier_weight:.2f}**
  - Inventory Risk Weight: **{config.scoring.inventory_weight:.2f}**
  - Capacity Risk Weight: **{config.scoring.capacity_weight:.2f}**
  - Service Risk Weight: **{config.scoring.service_weight:.2f}**
- **Missing Dimension Policy**: `{config.scoring.missing_dimension_policy}` (renormalizes weights dynamically when unobserved dimensions occur).

---

### 5. Risk Scoring Methodology
```text
Composite Risk Score = 
    (delay_weight × delay_risk)
  + (supplier_weight × supplier_risk)
  + (inventory_weight × inventory_risk)
  + (capacity_weight × capacity_risk)
  + (service_weight × service_risk)
```
All dimension scores are mathematically bounded in `[0.0, 1.0]`. Individual dimensions represent distinct failure modes:
1. **Delay Risk**: Model estimated likelihood of lead time variance.
2. **Supplier Risk**: Inverted supplier reliability supplemented by past-only historical delay frequency (`shift(1)` expanding mean).
3. **Inventory Risk**: Deficit relative to required shipment demand based on `inventory_coverage_ratio`.
4. **Capacity Risk**: Bottleneck pressure measured by `demand / supplier_capacity`, with strict zero-capacity failure handling.
5. **Service Risk**: Product criticality modulated by stockout exposure.

---

### 6. Risk-Level Distribution
| Operational Risk Level | Score Range | Shipment Count | Cohort Percentage | Priority Action |
| :--- | :---: | :---: | :---: | :--- |
| **CRITICAL** | `[0.80, 1.00]` | **{crit_count}** | **{crit_count/total*100:.1f}%** | Immediate prescriptive solver optimization |
| **HIGH** | `[0.60, 0.80)` | **{high_count}** | **{high_count/total*100:.1f}%** | Sourcing review and buffer allocation |
| **MEDIUM** | `[0.30, 0.60)` | **{med_count}** | **{med_count/total*100:.1f}%** | Scheduled tracking and standard monitoring |
| **LOW** | `[0.00, 0.30)` | **{low_count}** | **{low_count/total*100:.1f}%** | Normal execution without intervention |

---

### 7. High-Risk Suppliers
Top suppliers with concentrated disruption exposure:
{top_suppliers_md}

---

### 8. Inventory Exposure
- Shipments facing **stockout deficit (`inventory_coverage_ratio < 1.0`)**: **{(assessments_df['inventory_risk_score'] > 0.40).sum()} events**
- Critical inventory shortage (`coverage < 0.50`): **{(assessments_df['inventory_risk_score'] >= 0.80).sum()} events**

---

### 9. Capacity Pressure
- Shipments at or above 85% supplier capacity utilization: **{(assessments_df['capacity_risk_score'] >= 0.60).sum()} events**
- Shipments at capacity overload (>= 100% capacity): **{(assessments_df['capacity_risk_score'] >= 0.85).sum()} events**

---

### 10. Critical Product Exposure
- Exposure evaluation isolates high-criticality components experiencing concurrent inventory strain.
- When product criticality metadata is omitted, the engine activates the configured `{config.scoring.missing_dimension_policy}` policy without inventing unverified ratings.

---

### 11. Generated Alerts
- **Total Structured Alerts Generated**: **{len(alerts)}**
- **Priority Disruption Alerts (CRITICAL)**: **{sum(1 for a in alerts if a.severity == 'CRITICAL')}**
- **Warning Disruption Alerts (HIGH)**: **{sum(1 for a in alerts if a.severity == 'HIGH')}**
- **Deduplication Mechanism**: Fully enforced via `(event_id, alert_type)` compound keys.

Top Triggered Reason Codes:
| Reason Code | Trigger Count | Description |
| :--- | :---: | :--- |
{chr(10).join([f"| `{code}` | {cnt} | Primary operational risk driver |" for code, cnt in reason_series.head(6).items()])}

---

### 12. Missing Data and Limitations
1. **Product Criticality Granularity**: Product criticality levels are maintained in catalog definitions; when missing from transaction lines, the service dimension is omitted and remaining weights are renormalized.
2. **Probabilistic Calibration**: Model predictions provide an empirical probability of delay; risk scores represent composite severity indices rather than calibrated likelihood of complete stockout.

---

### 13. Recommended Operational Actions
1. **Priority Optimization Hand-Off**: Immediately feed the **{crit_count + high_count} high-risk events** to the Day 9 Prescriptive Optimization Engine.
2. **Secondary Sourcing Evaluation**: For shipments exhibiting high supplier risk, evaluate alternate suppliers with lower utilization pressure.
3. **Expedited Logistics Pre-Check**: Review critical inventory shortage events for expedited air freight intervention subject to optimization trade-offs.

---

### 14. Integration Notes for the Optimization Engine (Day 9 Contract)
The optimization engine consumes the exported `risk_assessments.csv` records using the following interface schema:
```text
event_id
delay_probability
risk_score
risk_level
supplier_risk_score
inventory_risk_score
capacity_risk_score
service_risk_score
risk_reasons
model_version
assessment_timestamp
```
The Prescriptive Optimization Engine will evaluate trade-offs between expedited air freight, secondary supplier sourcing, inventory redistribution, and customer SLA penalties to prescribe the mathematically cost-optimal intervention.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Generated risk assessment summary markdown report at {output_path}")
    return output_path


def run_risk_engine(
    config_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    persist_db: bool = False,
) -> RiskEngineSummary:
    """Execute complete end-to-end Risk Engine batch workflow."""
    # 1. Load Configuration
    config = load_risk_config(config_path)
    logger.info(f"Loaded Risk Engine configuration (version: {config.version})")

    # 2. Resolve Paths
    data_file = dataset_path or config.data.processed_dataset_path
    reports_dir = output_dir or Path("reports/risk_engine")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {data_file}")

    # 3. Load Data
    logger.info(f"Reading processed dataset from {data_file}...")
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} shipment records for risk assessment.")

    # 4. Load Model
    model_path = config.model.model_path
    meta_path = config.model.metadata_path
    pipeline, metadata = load_model_pipeline(model_path, meta_path)
    model_version = metadata.get("model_version", "1.0.0")

    # 5. Generate Delay Probabilities
    logger.info("Generating delay probabilities using fitted Day 6 model pipeline...")
    probs = generate_model_probabilities(df, pipeline)

    # 6. Evaluate Multi-Dimensional Risk
    assessments, skipped, failed = evaluate_batch_risk(
        df=df,
        probabilities=probs,
        config=config,
        model_version=model_version,
    )

    # 7. Generate Alerts
    alert_engine = RiskAlertEngine(config=config)
    alerts = alert_engine.process_assessments(assessments, include_monitoring=False)
    logger.info(f"Generated {len(alerts)} high and critical risk alerts.")

    # 8. Export CSVs
    assessments_csv, high_risk_csv = export_risk_reports(assessments, df, reports_dir)
    assessments_df = pd.read_csv(assessments_csv)
    high_risk_df = pd.read_csv(high_risk_csv)

    # 9. Generate Visualizations
    logger.info("Generating visualization charts...")
    charts = generate_visualizations(assessments_df, df, reports_dir)
    logger.info(f"Generated {len(charts)} charts in {reports_dir}.")

    # 10. Summary Aggregation
    level_counts = assessments_df["risk_level"].value_counts().to_dict()
    alert_counts = alert_engine.alert_counts

    summary = RiskEngineSummary(
        total_shipments_assessed=len(assessments),
        processed_count=len(assessments),
        skipped_count=skipped,
        failed_count=failed,
        risk_level_counts=level_counts,
        alert_counts=alert_counts,
        average_composite_score=round(float(assessments_df["composite_risk_score"].mean()), 4),
        model_version=model_version,
        configuration_version=config.version,
        classification_threshold=config.delay_probability.classification_threshold,
    )

    # 11. Generate Markdown Report
    summary_report_path = reports_dir / "risk_assessment_summary.md"
    generate_markdown_summary(
        summary=summary,
        assessments_df=assessments_df,
        high_risk_df=high_risk_df,
        alerts=alerts,
        config=config,
        output_path=summary_report_path,
    )

    # 12. Optional Database Persistence
    if persist_db:
        logger.info("Attempting database persistence for risk assessments...")
        count, err = persist_risk_assessments(assessments)
        if err:
            logger.warning(f"Database persistence note: {err}")
        else:
            logger.info(f"Database persistence completed: {count} records saved.")

    logger.info("=" * 60)
    logger.info("Supply Prescript Risk Engine batch execution completed successfully.")
    logger.info(f"Assessed: {summary.total_shipments_assessed} shipments | High/Critical Risk: {len(high_risk_df)}")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Supply Prescript Risk Engine.")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to risk_config.yaml")
    parser.add_argument("--dataset", "-d", type=str, default=None, help="Path to processed dataset CSV")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--persist", "-p", action="store_true", help="Persist records to SQLAlchemy database")
    args = parser.parse_args()

    c_path = Path(args.config) if args.config else None
    d_path = Path(args.dataset) if args.dataset else None
    o_path = Path(args.output) if args.output else None

    run_risk_engine(
        config_path=c_path,
        dataset_path=d_path,
        output_dir=o_path,
        persist_db=args.persist,
    )
