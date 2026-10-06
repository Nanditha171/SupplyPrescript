"""
Execute Exploratory Data Analysis & Save Visualizations
======================================================
Generates all 5 publication-ready analytical figures in reports/figures/
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def generate_visualizations():
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Set aesthetics
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

    # Load dataset
    data_path = Path("data/processed/supply_prescript_processed_dataset.csv")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded {len(df)} records from {data_path}")

    # ==========================================
    # 1. Demand Analysis
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Pillar 1: Demand Volatility & Product Dynamics', fontsize=16, fontweight='bold', y=0.98)

    # 1A. Distribution
    sns.histplot(df['demand_units'], kde=True, ax=axes[0, 0], color='#2b5c8f', bins=25)
    axes[0, 0].axvline(df['demand_units'].mean(), color='#e74c3c', linestyle='--', label=f"Mean: {df['demand_units'].mean():.1f}")
    axes[0, 0].axvline(df['demand_units'].median(), color='#27ae60', linestyle='-', label=f"Median: {df['demand_units'].median():.1f}")
    axes[0, 0].set_title('(A) Demand Distribution & Density')
    axes[0, 0].set_xlabel('Demand Quantity (Units)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()

    # 1B. Product Demand
    prod_demand = df.groupby('product_name')['demand_units'].agg(['mean', 'std']).reset_index().sort_values('mean', ascending=False)
    sns.barplot(data=prod_demand, x='product_name', y='mean', color='#3498db', ax=axes[0, 1])
    axes[0, 1].errorbar(x=range(len(prod_demand)), y=prod_demand['mean'], yerr=prod_demand['std'], fmt='none', c='black', capsize=4)
    axes[0, 1].set_title('(B) Mean Demand by Product (+/- 1 Std Dev)')
    axes[0, 1].set_xlabel('Product Name')
    axes[0, 1].set_ylabel('Mean Demand (Units)')
    axes[0, 1].tick_params(axis='x', rotation=30)

    # 1C. Time Trend
    daily_demand = df.groupby('date')['demand_units'].sum().reset_index()
    daily_demand['rolling_7d'] = daily_demand['demand_units'].rolling(7, min_periods=1).mean()
    axes[1, 0].plot(daily_demand['date'], daily_demand['demand_units'], alpha=0.35, color='#3498db', label='Daily Total Demand')
    axes[1, 0].plot(daily_demand['date'], daily_demand['rolling_7d'], color='#1d3557', linewidth=2.2, label='7-Day Rolling Mean')
    axes[1, 0].set_title('(C) Temporal Demand Evolution')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Total Daily Demand (Units)')
    axes[1, 0].legend()

    # 1D. Demand Pressure
    sns.boxplot(data=df, x='product_name', y='demand_pressure', ax=axes[1, 1], color='#a8dadc')
    axes[1, 1].axhline(1.0, color='#e74c3c', linestyle=':', label='Capacity Limit (1.0)')
    axes[1, 1].set_title('(D) Demand Pressure Index (Demand / Capacity)')
    axes[1, 1].set_xlabel('Product Name')
    axes[1, 1].set_ylabel('Demand Pressure Ratio')
    axes[1, 1].tick_params(axis='x', rotation=30)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(figures_dir / '01_demand_analysis.png', bbox_inches='tight')
    plt.close()
    print("Saved 01_demand_analysis.png")

    # ==========================================
    # 2. Supplier Performance
    # ==========================================
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

    # 2A. Delay Rate
    sns.barplot(data=supp_scorecard, x='delay_rate', y='supplier_name', hue='supplier_country', dodge=False, ax=axes[0, 0], palette='turbo')
    axes[0, 0].set_title('(A) Supplier Delay Occurrence Rate (%)')
    axes[0, 0].set_xlabel('Delay Probability')
    axes[0, 0].set_ylabel('Supplier Name')
    axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    axes[0, 0].legend(title='Country', loc='lower right')

    # 2B. Delay Severity
    delay_severity = df.groupby('supplier_name')['delay_days'].mean().reset_index().sort_values('delay_days', ascending=False)
    sns.barplot(data=delay_severity, x='delay_days', y='supplier_name', ax=axes[0, 1], color='#e74c3c')
    axes[0, 1].set_title('(B) Mean Delay Days by Supplier')
    axes[0, 1].set_xlabel('Mean Delay (Days)')
    axes[0, 1].set_ylabel('')

    # 2C. Reliability vs Delay Rate
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

    # 2D. Country Breakdown
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
    plt.savefig(figures_dir / '02_supplier_performance.png', bbox_inches='tight')
    plt.close()
    print("Saved 02_supplier_performance.png")

    # ==========================================
    # 3. Lead Time Dynamics
    # ==========================================
    df['lead_time_variance'] = df['actual_lead_time_days'] - df['planned_lead_time_days']

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Pillar 3: Lead Time Dynamics & Slippages', fontsize=16, fontweight='bold', y=0.98)

    # 3A. Distribution
    sns.kdeplot(df['planned_lead_time_days'], label='Planned Lead Time', fill=True, ax=axes[0, 0], color='#2980b9', alpha=0.4)
    sns.kdeplot(df['actual_lead_time_days'], label='Actual Lead Time', fill=True, ax=axes[0, 0], color='#e67e22', alpha=0.4)
    axes[0, 0].axvline(df['planned_lead_time_days'].mean(), color='#2980b9', linestyle='--', label=f"Mean Planned: {df['planned_lead_time_days'].mean():.1f}d")
    axes[0, 0].axvline(df['actual_lead_time_days'].mean(), color='#e67e22', linestyle='--', label=f"Mean Actual: {df['actual_lead_time_days'].mean():.1f}d")
    axes[0, 0].set_title('(A) Planned vs. Actual Lead Time Distribution')
    axes[0, 0].set_xlabel('Lead Time (Days)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()

    # 3B. Product Variance
    sns.boxplot(data=df, x='product_name', y='lead_time_variance', ax=axes[0, 1], color='#bde0fe')
    axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('(B) Lead Time Variance (Actual - Planned Days)')
    axes[0, 1].set_xlabel('Product Name')
    axes[0, 1].set_ylabel('Variance (Days)')
    axes[0, 1].tick_params(axis='x', rotation=30)

    # 3C. Scatter Planned vs Actual
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

    # 3D. Slippage by Country
    country_slip = df.groupby('supplier_country')['lead_time_variance'].mean().reset_index()
    sns.barplot(data=country_slip, x='supplier_country', y='lead_time_variance', ax=axes[1, 1], color='#457b9d')
    axes[1, 1].set_title('(D) Average Lead Time Slippage by Sourcing Country')
    axes[1, 1].set_xlabel('Supplier Country')
    axes[1, 1].set_ylabel('Avg Slippage (Days)')

    plt.tight_layout()
    plt.savefig(figures_dir / '03_lead_time_variance.png', bbox_inches='tight')
    plt.close()
    print("Saved 03_lead_time_variance.png")

    # ==========================================
    # 4. Inventory Health
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Pillar 4: Inventory Health & Stockout Exposure', fontsize=16, fontweight='bold', y=0.98)

    # 4A. Coverage Ratio
    sns.histplot(df['inventory_coverage_ratio'], kde=True, ax=axes[0, 0], color='#16a085', bins=25)
    axes[0, 0].axvline(1.0, color='#e74c3c', linestyle='--', linewidth=2, label='Break-Even Ratio (1.0)')
    axes[0, 0].axvline(df['inventory_coverage_ratio'].median(), color='#2980b9', linestyle='-', label=f"Median: {df['inventory_coverage_ratio'].median():.2f}")
    axes[0, 0].set_title('(A) Inventory Coverage Ratio Distribution')
    axes[0, 0].set_xlabel('Coverage Ratio (Inventory / Demand)')
    axes[0, 0].set_ylabel('Order Count')
    axes[0, 0].legend()

    # 4B. Gap across Risk Classes
    sns.boxplot(data=df, x='service_risk', y='inventory_gap_units', order=['Low', 'Medium', 'High'], ax=axes[0, 1], color='#f4a261')
    axes[0, 1].axhline(0, color='black', linestyle=':', label='Zero Net Buffer')
    axes[0, 1].set_title('(B) Inventory Buffer Gap by Service Risk Class')
    axes[0, 1].set_xlabel('Service Risk Level')
    axes[0, 1].set_ylabel('Inventory Gap (Units)')
    axes[0, 1].legend()

    # 4C. Shortage Probability by Product
    shortage_rate = df.groupby('product_name')['shortage_flag'].mean().reset_index().sort_values('shortage_flag', ascending=False)
    sns.barplot(data=shortage_rate, x='product_name', y='shortage_flag', ax=axes[1, 0], color='#e74c3c')
    axes[1, 0].set_title('(C) Shortage / Stockout Frequency by Product')
    axes[1, 0].set_xlabel('Product Name')
    axes[1, 0].set_ylabel('Shortage Rate')
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    axes[1, 0].tick_params(axis='x', rotation=30)

    # 4D. Risk Distribution
    risk_counts = df['service_risk'].value_counts(normalize=True).loc[['Low', 'Medium', 'High']]
    sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=axes[1, 1], palette=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[1, 1].set_title('(D) Service Risk Classification Breakdown')
    axes[1, 1].set_xlabel('Operational Risk Classification')
    axes[1, 1].set_ylabel('Proportion of Orders')
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    for i, val in enumerate(risk_counts.values):
        axes[1, 1].text(i, val + 0.01, f"{val:.1%}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(figures_dir / '04_inventory_health.png', bbox_inches='tight')
    plt.close()
    print("Saved 04_inventory_health.png")

    # ==========================================
    # 5. Delay Drivers & Cost Trade-Offs
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pillar 5: Delay Drivers & Financial Trade-Offs', fontsize=16, fontweight='bold', y=0.98)

    # 5A. Correlation Matrix
    corr_cols = [
        'demand_units', 'inventory_units', 'supplier_capacity_units',
        'planned_lead_time_days', 'actual_lead_time_days', 'delay_occurred',
        'delay_days', 'supplier_reliability', 'unit_cost_usd',
        'inventory_coverage_ratio', 'demand_pressure', 'supplier_risk_score'
    ]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0, 0], cbar_kws={'label': 'Pearson r'})
    axes[0, 0].set_title('(A) Correlation Matrix of Key Supply Chain Variables')

    # 5B. Delay Severity
    delayed_subset = df[df['delay_days'] > 0]
    sns.histplot(delayed_subset['delay_days'], bins=15, kde=True, ax=axes[0, 1], color='#c0392b')
    axes[0, 1].axvline(delayed_subset['delay_days'].mean(), color='black', linestyle='--', label=f"Mean Delay: {delayed_subset['delay_days'].mean():.1f}d")
    axes[0, 1].set_title('(B) Delay Severity Distribution (Conditional on Delay > 0)')
    axes[0, 1].set_xlabel('Delayed Days')
    axes[0, 1].set_ylabel('Number of Instances')
    axes[0, 1].legend()

    # 5C. Cost Comparison
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

    # 5D. Delay by Risk Quartile
    df['risk_bin'] = pd.qcut(df['supplier_risk_score'], q=4, labels=['Q1 (Low)', 'Q2 (Mod)', 'Q3 (Elevated)', 'Q4 (High)'])
    risk_delay = df.groupby('risk_bin', observed=False)['delay_occurred'].mean().reset_index()
    sns.barplot(data=risk_delay, x='risk_bin', y='delay_occurred', ax=axes[1, 1], palette='OrRd')
    axes[1, 1].set_title('(D) Delay Occurrence by Supplier Risk Quartile')
    axes[1, 1].set_xlabel('Supplier Risk Score Quartile')
    axes[1, 1].set_ylabel('Delay Probability')
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    plt.tight_layout()
    plt.savefig(figures_dir / '05_delay_drivers_and_costs.png', bbox_inches='tight')
    plt.close()
    print("Saved 05_delay_drivers_and_costs.png")
    print("All 5 visualization figures generated successfully!")

if __name__ == "__main__":
    generate_visualizations()
