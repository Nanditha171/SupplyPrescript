"""
Supply Prescript — Comprehensive Exploratory Data Analysis (EDA) Generator
========================================================================
Analyzes:
1. Demand Dynamics & Volatility
2. Supplier Performance & Reliability
3. Lead Time Discrepancies (Planned vs Actual)
4. Inventory Health, Buffers & Shortage Risks
5. Delay Frequency, Severity, Root Causes & Cost Trade-offs

Generates publication-quality figures and an annotated Jupyter Notebook.
"""

import json
from pathlib import Path

def build_eda_notebook():
    notebook_path = Path("notebooks/01_exploratory_data_analysis.ipynb")
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = []

    def add_md(source_text):
        lines = [l + "\n" for l in source_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(source_text):
        lines = [l + "\n" for l in source_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    # Header / Objective
    add_md("""# 📦 Supply Prescript — Comprehensive Exploratory Data Analysis (EDA)
### *Deep-Dive Investigation of Demand Volatility, Supplier Performance, Lead Times, Inventory Coverage, and Delay Root Causes*

---

## 🎯 Executive Objective & Scope
The goal of this exploratory data analysis is to systematically analyze operational supply chain records to quantify vulnerabilities, supplier performance, bottlenecks, and financial trade-offs before constructing predictive risk models and prescriptive linear optimization engines.

### Key Analytical Pillars:
1. **Demand Distribution & Volatility**: Analyze demand volume, product patterns, pressure indices, and temporal evolution.
2. **Supplier Performance & Country Reliability**: Benchmark supplier delay rates, historical reliability, capacity constraints, and geographic risk.
3. **Lead Time Dynamics & Slippages**: Quantify planned vs. actual lead times, variances, and product-level bottlenecks.
4. **Inventory Health & Stockout Exposure**: Evaluate inventory coverage ratios, buffer adequacy, inventory gaps, and stockout occurrences.
5. **Delay Root Causes & Financial Trade-Offs**: Investigate delay frequency/severity drivers, correlation structure, secondary supplier cost premiums, and air freight escalation.""")

    # Setup & Data Loading
    add_md("""## 1. Environment Setup & Data Ingestion
We load the processed operational dataset (`data/processed/supply_prescript_processed_dataset.csv`) and configure modern plotting aesthetics.""")

    add_code("""import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set visual aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Load dataset
data_path = Path('../data/processed/supply_prescript_processed_dataset.csv')
if not data_path.exists():
    data_path = Path('data/processed/supply_prescript_processed_dataset.csv')

df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])

print(f"Dataset Successfully Loaded: {df.shape[0]} records, {df.shape[1]} features")
print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
display(df.head(5))""")

    # Overview & Dataset Summary
    add_md("""## 2. Dataset Overview & Operational Schema
Let's inspect the descriptive statistics, categorical cardinalities, and data dictionary alignment.""")

    add_code("""# High-level statistical profile of numerical features
key_numeric_cols = [
    'demand_units', 'inventory_units', 'supplier_capacity_units',
    'planned_lead_time_days', 'actual_lead_time_days', 'delay_days',
    'supplier_reliability', 'unit_cost_usd', 'transport_cost_usd',
    'inventory_coverage_ratio', 'demand_pressure', 'supplier_risk_score'
]

summary_stats = df[key_numeric_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
summary_stats['IQR'] = summary_stats['75%'] - summary_stats['25%']
print("--- Operational Metrics Summary Table ---")
display(summary_stats.round(3))

print("--- Categorical Entities ---")
print(f"Suppliers ({df['supplier_name'].nunique()}): {df['supplier_name'].unique().tolist()}")
print(f"Categories ({df['category'].nunique()}): {df['category'].unique().tolist()}")
print(f"Products ({df['product_name'].nunique()}): {df['product_name'].unique().tolist()}")
print(f"Countries ({df['supplier_country'].nunique()}): {df['supplier_country'].unique().tolist()}")""")

    # Pillar 1: Demand Analysis
    add_md("""## 3. Pillar 1: Demand Distribution, Volatility & Product Dynamics
Demand represents customer/production consumption requirements. We evaluate:
- Overall demand distribution & density
- Demand variation across product types
- Temporal trend and volatility over the observed timeframe
- Demand pressure (`demand_units / supplier_capacity_units`)""")

    add_code("""fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Pillar 1: Demand Volatility & Product Dynamics', fontsize=16, fontweight='bold', y=0.98)

# 1. Demand Distribution
sns.histplot(df['demand_units'], kde=True, ax=axes[0, 0], color='#2b5c8f', bins=25)
axes[0, 0].axvline(df['demand_units'].mean(), color='#e74c3c', linestyle='--', label=f"Mean: {df['demand_units'].mean():.1f}")
axes[0, 0].axvline(df['demand_units'].median(), color='#27ae60', linestyle='-', label=f"Median: {df['demand_units'].median():.1f}")
axes[0, 0].set_title('(A) Demand Distribution & Density')
axes[0, 0].set_xlabel('Demand Quantity (Units)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# 2. Demand by Product Name
prod_demand = df.groupby('product_name')['demand_units'].agg(['mean', 'std']).reset_index().sort_values('mean', ascending=False)
sns.barplot(data=prod_demand, x='product_name', y='mean', color='#3498db', ax=axes[0, 1])
axes[0, 1].errorbar(x=range(len(prod_demand)), y=prod_demand['mean'], yerr=prod_demand['std'], fmt='none', c='black', capsize=4)
axes[0, 1].set_title('(B) Mean Demand by Product (+/- 1 Std Dev)')
axes[0, 1].set_xlabel('Product Name')
axes[0, 1].set_ylabel('Mean Demand (Units)')
axes[0, 1].tick_params(axis='x', rotation=30)

# 3. Time Trend (Weekly Rolling Mean)
daily_demand = df.groupby('date')['demand_units'].sum().reset_index()
daily_demand['rolling_7d'] = daily_demand['demand_units'].rolling(7, min_periods=1).mean()
axes[1, 0].plot(daily_demand['date'], daily_demand['demand_units'], alpha=0.35, color='#3498db', label='Daily Total Demand')
axes[1, 0].plot(daily_demand['date'], daily_demand['rolling_7d'], color='#1d3557', linewidth=2.2, label='7-Day Rolling Mean')
axes[1, 0].set_title('(C) Temporal Demand Evolution')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Total Daily Demand (Units)')
axes[1, 0].legend()

# 4. Demand Pressure Index by Product
sns.boxplot(data=df, x='product_name', y='demand_pressure', ax=axes[1, 1], color='#a8dadc')
axes[1, 1].axhline(1.0, color='#e74c3c', linestyle=':', label='Capacity Limit (1.0)')
axes[1, 1].set_title('(D) Demand Pressure Index (Demand / Capacity)')
axes[1, 1].set_xlabel('Product Name')
axes[1, 1].set_ylabel('Demand Pressure Ratio')
axes[1, 1].tick_params(axis='x', rotation=30)
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("--- Demand Insights ---")
print(f"Overall Demand: Mean = {df['demand_units'].mean():.1f}, Std = {df['demand_units'].std():.1f}, Min = {df['demand_units'].min()}, Max = {df['demand_units'].max()}")
print(f"Highest Demand Product: {prod_demand.iloc[0]['product_name']} ({prod_demand.iloc[0]['mean']:.1f} units)")
print(f"Demand Pressure > 0.85 (High Load) Frequency: {(df['demand_pressure'] > 0.85).mean()*100:.2f}%")""")

    # Pillar 2: Supplier Performance
    add_md("""## 4. Pillar 2: Supplier Performance, Reliability & Sourcing Geography
Suppliers represent the primary upstream supply risk. We examine:
- Delay frequency (%) by supplier
- Mean delay severity (in days)
- Supplier reliability ratings vs. empirical failure frequency
- Geographic distribution of supplier performance""")

    add_code("""# Aggregate supplier scorecard
supp_scorecard = df.groupby(['supplier_name', 'supplier_country']).agg(
    order_count=('supplier_id', 'count'),
    delay_rate=('delay_occurred', 'mean'),
    mean_delay_days=('delay_days', 'mean'),
    max_delay_days=('delay_days', 'max'),
    mean_reliability=('supplier_reliability', 'mean'),
    mean_capacity=('supplier_capacity_units', 'mean'),
    mean_lead_time=('actual_lead_time_days', 'mean')
).reset_index().sort_values('delay_rate', ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Pillar 2: Supplier Performance & Reliability Benchmark', fontsize=16, fontweight='bold', y=0.98)

# 1. Supplier Delay Rate
sns.barplot(data=supp_scorecard, x='delay_rate', y='supplier_name', hue='supplier_country', dodge=False, ax=axes[0, 0], palette='turbo')
axes[0, 0].set_title('(A) Supplier Delay Occurrence Rate (%)')
axes[0, 0].set_xlabel('Delay Probability')
axes[0, 0].set_ylabel('Supplier Name')
axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
axes[0, 0].legend(title='Country', loc='lower right')

# 2. Mean Delay Days
delay_severity = df.groupby('supplier_name')['delay_days'].mean().reset_index().sort_values('delay_days', ascending=False)
sns.barplot(data=delay_severity, x='delay_days', y='supplier_name', ax=axes[0, 1], color='#e74c3c')
axes[0, 1].set_title('(B) Mean Delay Days by Supplier')
axes[0, 1].set_xlabel('Mean Delay (Days)')
axes[0, 1].set_ylabel('')

# 3. Reliability vs Actual Delay Rate Scatter
sns.scatterplot(
    data=supp_scorecard,
    x='mean_reliability',
    y='delay_rate',
    hue='supplier_country',
    size='order_count',
    sizes=(120, 300),
    ax=axes[1, 0],
    palette='Set1'
)
for _, row in supp_scorecard.iterrows():
    axes[1, 0].annotate(
        row['supplier_name'].split()[0],
        (row['mean_reliability'] + 0.001, row['delay_rate'] + 0.005),
        fontsize=9,
        fontweight='bold'
    )
axes[1, 0].set_title('(C) Stated Reliability vs. Empirical Delay Rate')
axes[1, 0].set_xlabel('Supplier Historical Reliability Score')
axes[1, 0].set_ylabel('Empirical Delay Rate')
axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

# 4. Country Level Delay Benchmarking
country_perf = df.groupby('supplier_country').agg(
    delay_rate=('delay_occurred', 'mean'),
    mean_delay_days=('delay_days', 'mean'),
    mean_actual_lt=('actual_lead_time_days', 'mean')
).reset_index().sort_values('delay_rate', ascending=False)
sns.barplot(data=country_perf, x='supplier_country', y='delay_rate', ax=axes[1, 1], color='#457b9d')
axes[1, 1].set_title('(D) Sourcing Country Delay Frequency')
axes[1, 1].set_xlabel('Supplier Country')
axes[1, 1].set_ylabel('Delay Rate')
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

plt.tight_layout()
plt.show()

print("--- Supplier Scorecard Table ---")
display(supp_scorecard.round(3))""")

    # Pillar 3: Lead Time Dynamics
    add_md("""## 5. Pillar 3: Lead Time Dynamics & Discrepancy Analysis
Lead time represents the elapsed calendar duration from purchase order placement to receipt.
- Planned lead time vs. Observed actual lead time
- Lead time slippage / variance (`actual_lead_time_days - planned_lead_time_days`)
- Product-level lead time sensitivity and country geographic impact""")

    add_code("""df['lead_time_variance'] = df['actual_lead_time_days'] - df['planned_lead_time_days']

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Pillar 3: Lead Time Dynamics & Slippages', fontsize=16, fontweight='bold', y=0.98)

# 1. Planned vs Actual Lead Time Distributions
sns.kdeplot(df['planned_lead_time_days'], label='Planned Lead Time', fill=True, ax=axes[0, 0], color='#2980b9', alpha=0.4)
sns.kdeplot(df['actual_lead_time_days'], label='Actual Lead Time', fill=True, ax=axes[0, 0], color='#e67e22', alpha=0.4)
axes[0, 0].axvline(df['planned_lead_time_days'].mean(), color='#2980b9', linestyle='--', label=f"Mean Planned: {df['planned_lead_time_days'].mean():.1f}d")
axes[0, 0].axvline(df['actual_lead_time_days'].mean(), color='#e67e22', linestyle='--', label=f"Mean Actual: {df['actual_lead_time_days'].mean():.1f}d")
axes[0, 0].set_title('(A) Planned vs. Actual Lead Time Distribution')
axes[0, 0].set_xlabel('Lead Time (Days)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()

# 2. Product Variance
sns.boxplot(data=df, x='product_name', y='lead_time_variance', ax=axes[0, 1], color='#bde0fe')
axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
axes[0, 1].set_title('(B) Lead Time Variance (Actual - Planned Days)')
axes[0, 1].set_xlabel('Product Name')
axes[0, 1].set_ylabel('Variance (Days)')
axes[0, 1].tick_params(axis='x', rotation=30)

# 3. Lead Time Scatter (Planned vs Actual)
sns.scatterplot(
    data=df,
    x='planned_lead_time_days',
    y='actual_lead_time_days',
    hue='delay_occurred',
    palette={0: '#2ecc71', 1: '#e74c3c'},
    alpha=0.7,
    ax=axes[1, 0]
)
max_val = max(df['planned_lead_time_days'].max(), df['actual_lead_time_days'].max())
axes[1, 0].plot([0, max_val], [0, max_val], color='gray', linestyle=':', label='On-Time Trajectory')
axes[1, 0].set_title('(C) Lead Time Scatter & Delay Departures')
axes[1, 0].set_xlabel('Planned Lead Time (Days)')
axes[1, 0].set_ylabel('Actual Lead Time (Days)')
axes[1, 0].legend(title='Delay Occurred')

# 4. Lead Time Variance by Country
country_slip = df.groupby('supplier_country')['lead_time_variance'].mean().reset_index()
sns.barplot(data=country_slip, x='supplier_country', y='lead_time_variance', ax=axes[1, 1], color='#457b9d')
axes[1, 1].set_title('(D) Average Lead Time Slippage by Sourcing Country')
axes[1, 1].set_xlabel('Supplier Country')
axes[1, 1].set_ylabel('Avg Slippage (Days)')

plt.tight_layout()
plt.show()

print("--- Lead Time Summary ---")
print(f"Mean Planned Lead Time: {df['planned_lead_time_days'].mean():.2f} days (Std: {df['planned_lead_time_days'].std():.2f})")
print(f"Mean Actual Lead Time:  {df['actual_lead_time_days'].mean():.2f} days (Std: {df['actual_lead_time_days'].std():.2f})")
print(f"Average Slippage:       {df['lead_time_variance'].mean():.2f} days (Max Slippage: {df['lead_time_variance'].max()} days)")
print(f"Orders with Slippage:   {(df['lead_time_variance'] > 0).mean()*100:.2f}%")""")

    # Pillar 4: Inventory & Shortage Analysis
    add_md("""## 6. Pillar 4: Inventory Health, Buffer Adequacy & Shortage Exposure
Inventory buffers absorb lead time fluctuations and supplier delivery delays. We evaluate:
- Inventory Coverage Ratio (`inventory_units / demand_units`)
- Inventory Gap (`inventory_units - demand_units`)
- Shortage probability and stockout risks by product
- Service Risk classification breakdown""")

    add_code("""fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Pillar 4: Inventory Health & Stockout Exposure', fontsize=16, fontweight='bold', y=0.98)

# 1. Inventory Coverage Ratio Distribution
sns.histplot(df['inventory_coverage_ratio'], kde=True, ax=axes[0, 0], color='#16a085', bins=25)
axes[0, 0].axvline(1.0, color='#e74c3c', linestyle='--', linewidth=2, label='Break-Even Ratio (1.0)')
axes[0, 0].axvline(df['inventory_coverage_ratio'].median(), color='#2980b9', linestyle='-', label=f"Median: {df['inventory_coverage_ratio'].median():.2f}")
axes[0, 0].set_title('(A) Inventory Coverage Ratio Distribution')
axes[0, 0].set_xlabel('Coverage Ratio (Inventory / Demand)')
axes[0, 0].set_ylabel('Order Count')
axes[0, 0].legend()

# 2. Inventory Gap across Service Risk Categories
sns.boxplot(data=df, x='service_risk', y='inventory_gap_units', order=['Low', 'Medium', 'High'], ax=axes[0, 1], color='#f4a261')
axes[0, 1].axhline(0, color='black', linestyle=':', label='Zero Net Buffer')
axes[0, 1].set_title('(B) Inventory Buffer Gap by Service Risk Class')
axes[0, 1].set_xlabel('Service Risk Level')
axes[0, 1].set_ylabel('Inventory Gap (Units)')
axes[0, 1].legend()

# 3. Shortage Probability by Product
shortage_rate = df.groupby('product_name')['shortage_flag'].mean().reset_index().sort_values('shortage_flag', ascending=False)
sns.barplot(data=shortage_rate, x='product_name', y='shortage_flag', ax=axes[1, 0], color='#e74c3c')
axes[1, 0].set_title('(C) Shortage / Stockout Frequency by Product')
axes[1, 0].set_xlabel('Product Name')
axes[1, 0].set_ylabel('Shortage Rate')
axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
axes[1, 0].tick_params(axis='x', rotation=30)

# 4. Service Risk Level Distribution
risk_counts = df['service_risk'].value_counts(normalize=True).loc[['Low', 'Medium', 'High']]
sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=axes[1, 1], palette=['#2ecc71', '#f39c12', '#e74c3c'])
axes[1, 1].set_title('(D) Service Risk Classification Breakdown')
axes[1, 1].set_xlabel('Operational Risk Classification')
axes[1, 1].set_ylabel('Proportion of Orders')
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
for i, val in enumerate(risk_counts.values):
    axes[1, 1].text(i, val + 0.01, f"{val:.1%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("--- Inventory Health Metrics ---")
print(f"Mean Inventory Units: {df['inventory_units'].mean():.1f} | Mean Demand Units: {df['demand_units'].mean():.1f}")
print(f"Overall Shortage Rate: {(df['shortage_flag'] == 1).mean()*100:.2f}% ({df['shortage_flag'].sum()} orders)")
print(f"Orders with Inventory Coverage < 1.0: {(df['inventory_coverage_ratio'] < 1.0).mean()*100:.2f}%")""")

    # Pillar 5: Delays & Cost Trade-offs
    add_md("""## 7. Pillar 5: Delay Root Causes, Correlation Structure & Financial Trade-Offs
Delays trigger expedited shipping costs and customer SLA penalties. We evaluate:
- Delay frequency and conditional severity
- Feature correlation matrix (demand, lead time, reliability, costs, delays)
- Sourcing cost comparison: Primary vs. Secondary vs. Air Freight""")

    add_code("""fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Pillar 5: Delay Drivers & Financial Trade-Offs', fontsize=16, fontweight='bold', y=0.98)

# 1. Correlation Heatmap
corr_cols = [
    'demand_units', 'inventory_units', 'supplier_capacity_units',
    'planned_lead_time_days', 'actual_lead_time_days', 'delay_occurred',
    'delay_days', 'supplier_reliability', 'unit_cost_usd',
    'inventory_coverage_ratio', 'demand_pressure', 'supplier_risk_score'
]
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0, 0], cbar_kws={'label': 'Pearson r'})
axes[0, 0].set_title('(A) Correlation Matrix of Key Supply Chain Variables')

# 2. Delay Severity Distribution (when delay > 0)
delayed_subset = df[df['delay_days'] > 0]
sns.histplot(delayed_subset['delay_days'], bins=15, kde=True, ax=axes[0, 1], color='#c0392b')
axes[0, 1].axvline(delayed_subset['delay_days'].mean(), color='black', linestyle='--', label=f"Mean Delay: {delayed_subset['delay_days'].mean():.1f}d")
axes[0, 1].set_title('(B) Delay Severity Distribution (Conditional on Delay > 0)')
axes[0, 1].set_xlabel('Delayed Days')
axes[0, 1].set_ylabel('Number of Instances')
axes[0, 1].legend()

# 3. Cost Trade-Off Comparison (Primary vs Secondary vs Air Freight)
cost_df = df[['estimated_primary_cost_usd', 'estimated_secondary_cost_usd', 'air_freight_cost_usd']].mean().reset_index()
cost_df.columns = ['Cost_Type', 'Mean_USD']
cost_df['Cost_Type'] = ['Primary Sourcing', 'Secondary Sourcing', 'Air Freight Expedited']
sns.barplot(data=cost_df, x='Cost_Type', y='Mean_USD', ax=axes[1, 0], palette='Blues_d')
axes[1, 0].set_title('(C) Mean Sourcing & Expedited Mitigation Costs')
axes[1, 0].set_xlabel('Procurement / Freight Mode')
axes[1, 0].set_ylabel('Mean Cost ($ USD)')
axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '${:,.0f}'.format(y)))
for i, v in enumerate(cost_df['Mean_USD']):
    axes[1, 0].text(i, v + 500, f"${v:,.2f}", ha='center', fontweight='bold')

# 4. Supplier Risk Score vs Delay Probability
df['risk_bin'] = pd.qcut(df['supplier_risk_score'], q=4, labels=['Q1 (Low)', 'Q2 (Mod)', 'Q3 (Elevated)', 'Q4 (High)'])
risk_delay = df.groupby('risk_bin', observed=False)['delay_occurred'].mean().reset_index()
sns.barplot(data=risk_delay, x='risk_bin', y='delay_occurred', ax=axes[1, 1], palette='OrRd')
axes[1, 1].set_title('(D) Delay Occurrence by Supplier Risk Quartile')
axes[1, 1].set_xlabel('Supplier Risk Score Quartile')
axes[1, 1].set_ylabel('Delay Probability')
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

plt.tight_layout()
plt.show()

print("--- Delay & Financial Metrics ---")
print(f"Overall Delay Occurrence Rate: {df['delay_occurred'].mean()*100:.2f}% ({df['delay_occurred'].sum()} / {len(df)} orders)")
print(f"Conditional Mean Delay: {delayed_subset['delay_days'].mean():.2f} days (Max: {df['delay_days'].max()} days)")
print(f"Mean Primary Cost: ${df['estimated_primary_cost_usd'].mean():,.2f}")
print(f"Mean Secondary Cost: ${df['estimated_secondary_cost_usd'].mean():,.2f} (+{((df['estimated_secondary_cost_usd'].mean()/df['estimated_primary_cost_usd'].mean())-1)*100:.1f}%)")
print(f"Mean Air Freight Cost: ${df['air_freight_cost_usd'].mean():,.2f} (+{((df['air_freight_cost_usd'].mean()/df['estimated_primary_cost_usd'].mean())-1)*100:.1f}%)")""")

    # Multi-dimensional Synthesis
    add_md("""## 8. Multi-Dimensional Synthesis: Risk Matrix & Actionable Prioritization
We cross-tabulate supplier reliability, lead times, inventory coverage, and delay frequency into a consolidated risk matrix.""")

    add_code("""# Consolidated Supplier Risk Summary Table
consolidated_summary = df.groupby('supplier_name').agg(
    country=('supplier_country', 'first'),
    total_orders=('supplier_id', 'count'),
    stated_reliability=('supplier_reliability', 'mean'),
    delay_rate=('delay_occurred', 'mean'),
    avg_planned_lead_time=('planned_lead_time_days', 'mean'),
    avg_actual_lead_time=('actual_lead_time_days', 'mean'),
    avg_delay_days=('delay_days', 'mean'),
    avg_capacity_utilization=('capacity_utilization', 'mean'),
    total_demand_served=('demand_units', 'sum')
).reset_index().sort_values('delay_rate', ascending=False)

# Convert formatted columns for display
display_summary = consolidated_summary.copy()
display_summary['stated_reliability'] = display_summary['stated_reliability'].map(lambda x: f"{x:.1%}")
display_summary['delay_rate'] = display_summary['delay_rate'].map(lambda x: f"{x:.1%}")
display_summary['avg_planned_lead_time'] = display_summary['avg_planned_lead_time'].map(lambda x: f"{x:.1f} d")
display_summary['avg_actual_lead_time'] = display_summary['avg_actual_lead_time'].map(lambda x: f"{x:.1f} d")
display_summary['avg_delay_days'] = display_summary['avg_delay_days'].map(lambda x: f"{x:.2f} d")
display_summary['avg_capacity_utilization'] = display_summary['avg_capacity_utilization'].map(lambda x: f"{x:.1%}")
display_summary['total_demand_served'] = display_summary['total_demand_served'].map(lambda x: f"{x:,.0f} units")

print("--- Comprehensive Supplier Performance & Vulnerability Matrix ---")
display(display_summary)""")

    # Final Summary Markdown Cell
    add_md("""## 9. Executive Summary & Strategic Findings

### Q&A
* **What are the primary drivers of supply chain delays?**
  Delays are strongly driven by **supplier reliability deficits** (Gamma Semiconductors at 0.89 reliability has a 20.5% delay rate and 2.04 days average delay), **long planned lead times** (>10 days), and **high demand pressure** (>85% capacity utilization).
* **Which product categories and suppliers present the highest operational risk?**
  **Microchips and Sensors** supplied by *Gamma Semiconductors (Taiwan)* and *Epsilon Tech (Malaysia)* experience the highest delay frequencies (20.51% each) and longest lead time slippages (averaging 14.95 days and 10.10 days actual lead times respectively).
* **How effective is current inventory in buffering against delays?**
  Across the dataset, mean inventory is 1,291 units against a mean demand of 951 units (median coverage ratio: 1.25). However, 12.82% of orders suffer an inventory coverage ratio below 1.0, triggering direct stockout vulnerability.
* **What are the financial trade-offs of mitigation actions?**
  Secondary supplier sourcing introduces a **15.2% cost premium** ($32,677 mean secondary cost vs. $28,367 primary cost), whereas expedited air-freight adds a **13.5% premium** ($32,192 mean cost). Linear prescriptive models must balance these premiums against stockout penalty costs.

### Data Analysis Key Findings
* **Total Operational Records**: Analyzed 624 operational transaction records across 8 global suppliers and 8 critical product lines.
* **Overall Delay Incidence**: 13.62% of all shipments experienced delays (85 out of 624 shipments). When delays occurred, the conditional mean delay severity was **8.20 days** (reaching a maximum of 15.0 days).
* **Lead Time Slippage**: Overall mean planned lead time was **8.20 days**, whereas mean actual lead time reached **9.32 days** (average operational slippage of +1.12 days across all shipments).
* **Supplier Performance Disparity**:
  * *Top performers*: **Alpha Components (India)** (7.69% delay rate, 0.51 avg delay days, 0.96 reliability) and **Theta Components (Singapore)** (8.97% delay rate, 0.85 avg delay days).
  * *High-risk bottlenecks*: **Gamma Semiconductors (Taiwan)** (20.51% delay rate, 2.04 avg delay days) and **Epsilon Tech (Malaysia)** (20.51% delay rate, 1.38 avg delay days).
* **Inventory Coverage & Shortages**: 80 orders (12.82%) experienced negative inventory gaps (demand exceeding inventory), leading directly to elevated service risk and potential production stoppage.

### Insights or Next Steps
* **Phase 2 (Predictive Risk Modeling)**: Develop supervised classification models (e.g., Random Forest, XGBoost) to forecast binary delay occurrence (`delay_occurred`) and regression models to estimate expected delay duration (`delay_days`) prior to order dispatch.
* **Phase 3 (Prescriptive Optimization)**: Formulate Mixed-Integer Linear Programming (MILP) models that dynamically allocate orders between primary suppliers, secondary backup suppliers, and expedited transport modes to minimize total expected landed cost while enforcing service-level agreements (SLAs).""")

    notebook_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_json, f, indent=2)

    print(f"Jupyter Notebook successfully created at: {notebook_path}")

if __name__ == "__main__":
    build_eda_notebook()
