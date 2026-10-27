"""
Supply Prescript - Interactive Demonstration Dashboard
Closed-Loop Prescriptive Analytics for Supply-Chain Operations
Author: Prasanna (ML & Optimization Lead)
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.interfaces import (
    predict_delay,
    optimize_actions,
    validate_recommendation,
    evaluate_decision,
    log_and_evaluate_cycle,
    get_model_metrics,
    get_history,
    get_feedback_analytics,
    retrain_model_with_feedback,
)
from optimization.constraints import OptimizationConstraints

# Page Configuration
st.set_page_config(
    page_title="Supply Prescript — Closed-Loop Prescriptive Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .recommendation-card {
        background: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .recommendation-card-optimal {
        border: 2px solid #10B981 !important;
        background: #F0FDF4 !important;
    }
    .recommendation-card-infeasible {
        border: 2px solid #EF4444 !important;
        background: #FEF2F2 !important;
        opacity: 0.85;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-optimal { background: #D1FAE5; color: #065F46; }
    .badge-feasible { background: #DBEAFE; color: #1E40AF; }
    .badge-infeasible { background: #FEE2E2; color: #991B1B; }
    .badge-critical { background: #FEE2E2; color: #991B1B; }
    .badge-high { background: #FFEDD5; color: #9A3412; }
    .badge-medium { background: #FEF3C7; color: #92400E; }
    .badge-low { background: #D1FAE5; color: #065F46; }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None
if "last_shipment_data" not in st.session_state:
    st.session_state.last_shipment_data = None


# -------------------------------------------------------------
# SIDEBAR: SCENARIOS & SYSTEM STATUS
# -------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Demo Controls & Scenarios")
st.sidebar.info("💡 **Quick Scenario Presets** (Click to load preset values)")

preset_scenario = st.sidebar.selectbox(
    "Choose Shipment Scenario:",
    [
        "🚨 Scenario 1: Critical Semiconductor Delay (Shenzhen → LA)",
        "⚠️ Scenario 2: Moderate Automotive Disruption (Munich → Dallas)",
        "✅ Scenario 3: Normal Low-Risk Warehouse Restock (Chicago)",
        "🛠️ Custom User Input",
    ],
    index=0,
)

# Preset default values
if "Scenario 1" in preset_scenario:
    def_sup = "Shenzhen Silicon Ltd"
    def_cat = "Semiconductors & Electronics"
    def_mode = "Ocean Intermodal"
    def_lead = 24.0
    def_cost = 8500.0
    def_qty = 500
    def_inv = 80
    def_rate = 0.48
    def_dist = 11500.0
    def_weather = 0.75
    def_cong = 0.70
    def_budget = 24000.0
    def_deadline = 10.0
elif "Scenario 2" in preset_scenario:
    def_sup = "Apex Auto Components"
    def_cat = "Automotive Assemblies"
    def_mode = "Standard Ground"
    def_lead = 10.0
    def_cost = 4500.0
    def_qty = 250
    def_inv = 200
    def_rate = 0.22
    def_dist = 4200.0
    def_weather = 0.38
    def_cong = 0.45
    def_budget = 12000.0
    def_deadline = 8.0
elif "Scenario 3" in preset_scenario:
    def_sup = "Bavaria Logistics Parts"
    def_cat = "Industrial Machinery"
    def_mode = "Standard Ground"
    def_lead = 5.0
    def_cost = 1800.0
    def_qty = 100
    def_inv = 800
    def_rate = 0.04
    def_dist = 800.0
    def_weather = 0.08
    def_cong = 0.10
    def_budget = 8000.0
    def_deadline = 6.0
else:
    def_sup = "GlobalTech Micro"
    def_cat = "Semiconductors & Electronics"
    def_mode = "Air Freight"
    def_lead = 7.0
    def_cost = 5000.0
    def_qty = 300
    def_inv = 150
    def_rate = 0.20
    def_dist = 5000.0
    def_weather = 0.25
    def_cong = 0.30
    def_budget = 18000.0
    def_deadline = 6.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Three-Member Ownership")
st.sidebar.markdown("""
- **Prasanna (You)**: ML (XGBoost) + Prescriptive Optimization (PuLP) + Demo UI
- **Trisha**: Backend API + PostgreSQL + Write-Back
- **Nanditha**: Operational UI & Team Dashboard
""")

metrics_summary = get_model_metrics()
cls_acc = metrics_summary.get("classification", {}).get("accuracy", 0.961) * 100
reg_mae = metrics_summary.get("regression", {}).get("mae", 0.67)

st.sidebar.markdown("---")
st.sidebar.metric("ML Model Accuracy", f"{cls_acc:.1f}%", "+XGBoost v2")
st.sidebar.metric("Regression MAE", f"{reg_mae:.2f} Days", "High Precision")

# -------------------------------------------------------------
# MAIN APP HEADER & TABS
# -------------------------------------------------------------
st.markdown('<div class="main-header">📦 Supply Prescript</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Closed-Loop Prescriptive Analytics: <b>Predict Disruptions ➔ Optimize Actions ➔ Execute Decision ➔ Evaluate Closed-Loop Outcome</b></div>',
    unsafe_allow_html=True,
)

tab_demo, tab_model_metrics, tab_audit_stress, tab_history = st.tabs([
    "🎯 Live Demo Flow (Steps 1–11)",
    "📈 Model Performance & Metrics",
    "🔬 Optimization Hard-Constraints Audit",
    "🔄 Closed-Loop History & Retraining",
])


# =============================================================
# TAB 1: LIVE DEMO FLOW
# =============================================================
with tab_demo:
    # ---------------------------------------------------------
    # SECTION 1: SHIPMENT INFORMATION & PARAMETERS
    # ---------------------------------------------------------
    st.markdown("### 1. Shipment Information & Operational Context")
    with st.expander("📋 Configure Shipment & Constraints", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🏢 Supplier & Cargo")
            supplier_name = st.selectbox(
                "Supplier Name",
                [
                    "Shenzhen Silicon Ltd",
                    "GlobalTech Micro",
                    "Apex Auto Components",
                    "Bavaria Logistics Parts",
                    "TransPacific Freight",
                    "Delta Industrial Supplies",
                    "Nordic Raw Materials",
                    "FastTrack Assembly Co",
                ],
                index=[
                    "Shenzhen Silicon Ltd",
                    "GlobalTech Micro",
                    "Apex Auto Components",
                    "Bavaria Logistics Parts",
                    "TransPacific Freight",
                    "Delta Industrial Supplies",
                    "Nordic Raw Materials",
                    "FastTrack Assembly Co",
                ].index(def_sup) if def_sup in [
                    "Shenzhen Silicon Ltd",
                    "GlobalTech Micro",
                    "Apex Auto Components",
                    "Bavaria Logistics Parts",
                    "TransPacific Freight",
                    "Delta Industrial Supplies",
                    "Nordic Raw Materials",
                    "FastTrack Assembly Co",
                ] else 0,
            )
            product_category = st.selectbox(
                "Product Category",
                [
                    "Semiconductors & Electronics",
                    "Automotive Assemblies",
                    "Industrial Machinery",
                    "Pharmaceuticals",
                    "Consumer Electronics",
                ],
                index=["Semiconductors & Electronics", "Automotive Assemblies", "Industrial Machinery", "Pharmaceuticals", "Consumer Electronics"].index(def_cat) if def_cat in ["Semiconductors & Electronics", "Automotive Assemblies", "Industrial Machinery", "Pharmaceuticals", "Consumer Electronics"] else 0,
            )
            shipping_mode = st.selectbox(
                "Planned Shipping Mode",
                ["Standard Ground", "Air Freight", "Ocean Intermodal", "Express Carrier"],
                index=["Standard Ground", "Air Freight", "Ocean Intermodal", "Express Carrier"].index(def_mode) if def_mode in ["Standard Ground", "Air Freight", "Ocean Intermodal", "Express Carrier"] else 0,
            )

        with col2:
            st.markdown("##### ⏱️ Logistics & Cost")
            lead_time_days = st.number_input("Promised Lead Time (Days)", min_value=1.0, max_value=60.0, value=float(def_lead), step=1.0)
            shipping_cost = st.number_input("Base Shipping Cost ($)", min_value=100.0, max_value=100000.0, value=float(def_cost), step=500.0)
            order_quantity = st.number_input("Order Quantity (Units)", min_value=10, max_value=5000, value=int(def_qty), step=50)
            inventory_level = st.number_input("Warehouse Safety Stock (Units)", min_value=0, max_value=5000, value=int(def_inv), step=50)

        with col3:
            st.markdown("##### ⚠️ Environmental Risk & Constraints")
            supplier_delay_rate = st.slider("Historical Supplier Delay Rate", min_value=0.0, max_value=1.0, value=float(def_rate), step=0.01, format="%.2f")
            weather_risk_index = st.slider("Weather Disruption Index (0-1)", min_value=0.0, max_value=1.0, value=float(def_weather), step=0.01)
            congestion_index = st.slider("Port / Corridor Congestion (0-1)", min_value=0.0, max_value=1.0, value=float(def_cong), step=0.01)
            route_distance_km = st.number_input("Route Distance (km)", min_value=100.0, max_value=25000.0, value=float(def_dist), step=500.0)

        st.markdown("---")
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            max_budget = st.number_input("Maximum Disruption Budget ($)", min_value=1000.0, max_value=150000.0, value=float(def_budget), step=1000.0)
        with c_col2:
            max_delivery_days = st.number_input("Required Delivery SLA Deadline (Days)", min_value=1.0, max_value=45.0, value=float(def_deadline), step=1.0)
        with c_col3:
            penalty_per_day = st.number_input("SLA Penalty Rate ($/Day Late)", min_value=100.0, max_value=5000.0, value=850.0, step=50.0)

    # Predict Button
    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        predict_clicked = st.button("🚀 PREDICT DELAY", type="primary", use_container_width=True)

    current_shipment = {
        "shipment_id": "SHP-10492",
        "supplier_name": supplier_name,
        "product_category": product_category,
        "shipping_mode": shipping_mode,
        "lead_time_days": lead_time_days,
        "shipping_cost": shipping_cost,
        "inventory_level": inventory_level,
        "order_quantity": order_quantity,
        "supplier_delay_rate": supplier_delay_rate,
        "route_distance_km": route_distance_km,
        "weather_risk_index": weather_risk_index,
        "congestion_index": congestion_index,
    }

    current_constraints = {
        "max_budget": max_budget,
        "max_delivery_days": max_delivery_days,
        "penalty_cost_per_day_late": penalty_per_day,
    }

    if predict_clicked:
        st.session_state.last_shipment_data = current_shipment
        # Run ML Prediction
        with st.spinner("Running XGBoost inference..."):
            pred = predict_delay(current_shipment)
            st.session_state.prediction_result = pred
            
            # Automatically run optimization if delay is predicted or risk is significant
            opt = optimize_actions(current_shipment, pred, current_constraints)
            st.session_state.optimization_result = opt
            st.session_state.selected_option = None
            st.session_state.evaluation_result = None

    # ---------------------------------------------------------
    # SECTION 2: ML PREDICTION RESULTS
    # ---------------------------------------------------------
    if st.session_state.prediction_result is not None:
        pred = st.session_state.prediction_result
        st.markdown("---")
        st.markdown("### 2. Machine Learning Shipment Risk Analysis")
        
        # Risk Badge Class
        risk_lvl = pred["risk_level"]
        badge_cls = {
            "CRITICAL": "badge-critical",
            "HIGH": "badge-high",
            "MEDIUM": "badge-medium",
            "LOW": "badge-low",
        }.get(risk_lvl, "badge-low")

        p_col1, p_col2, p_col3, p_col4 = st.columns([1.2, 1, 1, 1.5])
        
        with p_col1:
            st.markdown(f"#### Risk Level: <span class='status-badge {badge_cls}'>{risk_lvl}</span>", unsafe_allow_html=True)
            st.metric("Delay Probability", f"{pred['delay_probability_pct']}%", f"{risk_lvl} RISK")

        with p_col2:
            st.metric("Predicted Delay", f"{pred['predicted_delay_days']:.1f} Days", "Extra Transit Time")

        with p_col3:
            st.metric("Expected Total Lead Time", f"{pred['expected_total_days']:.1f} Days", f"Base: {pred['input_summary']['lead_time_days']:.1f}d")

        with p_col4:
            st.metric("Model Confidence", f"{pred['confidence_pct']}%", "XGBoost Probability Density")

        # Visual Gauge / Risk Drivers
        g_col1, g_col2 = st.columns([1, 1])
        with g_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred["delay_probability_pct"],
                title={"text": "<b>Shipment Disruption Risk Probability</b>", "font": {"size": 16}},
                number={"suffix": "%", "font": {"size": 28, "color": "#1E293B"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#EF4444" if pred["delay_probability_pct"] >= 60 else ("#F59E0B" if pred["delay_probability_pct"] >= 30 else "#10B981")},
                    "steps": [
                        {"range": [0, 30], "color": "#D1FAE5"},
                        {"range": [30, 60], "color": "#FEF3C7"},
                        {"range": [60, 85], "color": "#FFEDD5"},
                        {"range": [85, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": pred["delay_probability_pct"],
                    },
                },
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g_col2:
            st.markdown("##### 🔍 Top Delay Risk Drivers")
            for driver in pred["top_risk_drivers"]:
                impact_color = "🔴" if driver["impact"] == "HIGH" else ("🟡" if driver["impact"] == "MODERATE" else "🟢")
                st.markdown(f"- {impact_color} **{driver['factor']}**: `{driver['value']}` — *{driver['direction']}*")

        # -----------------------------------------------------
        # SECTION 3: PRESCRIPTIVE OPTIMIZATION
        # -----------------------------------------------------
        st.markdown("---")
        st.markdown("### 3. Prescriptive Optimization Engine (PuLP MILP Solver)")
        st.caption("Flow: **PREDICTED DELAY ➔ OPTIMIZATION ENGINE ➔ 3 RECOMMENDED ACTIONS**")

        opt_res = st.session_state.optimization_result
        if opt_res:
            solver_st = opt_res["solver_status"]
            if solver_st == "Optimal":
                st.success(f"✅ Optimization Solver Status: **{solver_st}** (Mixed-Integer Linear Program Solved Successfully)")
            elif solver_st == "INFEASIBLE":
                st.error("🚨 Optimization Solver Status: **INFEASIBLE** — No candidate action satisfies all hard business constraints!")
            else:
                st.warning(f"⚠️ Optimization Solver Status: **{solver_st}**")

            # 3 Recommendation Cards
            c_cards = st.columns(3)
            options = opt_res["options"]

            for idx, opt in enumerate(options):
                with c_cards[idx]:
                    is_rec = opt.get("is_recommended", False)
                    is_feas = opt.get("is_feasible", True)

                    card_border = "recommendation-card-optimal" if is_rec else ("recommendation-card-infeasible" if not is_feas else "recommendation-card")
                    badge_label = "RECOMMENDED OPTION ✓" if is_rec else (opt["status"])
                    badge_class = "badge-optimal" if is_rec else ("badge-infeasible" if not is_feas else "badge-feasible")

                    st.markdown(f"""
                    <div class="{card_border}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-weight: bold; font-size: 1.15rem; color: #1E293B;">{opt['id']}</span>
                            <span class="status-badge {badge_class}">{badge_label}</span>
                        </div>
                        <h4 style="margin: 0 0 10px 0; color: #0F172A;">{opt['name']}</h4>
                        <p style="font-size: 0.85rem; color: #475569; min-height: 40px;">{opt['description']}</p>
                        <hr style="margin: 8px 0;"/>
                        <div style="font-size: 0.95rem; margin-bottom: 6px;"><b>Total Cost:</b> <span style="font-size: 1.1rem; color: #0284C7; font-weight: bold;">${opt['cost']:,.2f}</span> <span style="font-size: 0.8rem; color: #64748B;">(+${opt['additional_cost']:,.2f})</span></div>
                        <div style="font-size: 0.95rem; margin-bottom: 6px;"><b>Expected Delivery:</b> <span style="font-size: 1.1rem; color: #16A34A; font-weight: bold;">{opt['delivery_days']:.1f} Days</span> <span style="font-size: 0.8rem; color: #64748B;">({opt['days_saved']:.1f}d saved)</span></div>
                        <div style="font-size: 0.9rem; margin-bottom: 8px;"><b>Risk Level:</b> {opt['risk_level']} | <b>Residual Delay:</b> {opt['residual_delay_days']}d</div>
                        <div style="font-size: 0.85rem; background: #F1F5F9; padding: 6px; border-radius: 6px; margin-bottom: 12px;">
                            <b>Constraint Audit:</b><br/>
                            {'✅' if opt['constraint_checks']['budget_satisfied'] else '❌'} Budget (≤ ${current_constraints['max_budget']:,.0f})<br/>
                            {'✅' if opt['constraint_checks']['delivery_satisfied'] else '❌'} Delivery SLA (≤ {current_constraints['max_delivery_days']:.1f}d)<br/>
                            {'✅' if opt['constraint_checks']['capacity_satisfied'] else '❌'} Capacity ({opt['capacity']} units)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if not is_feas:
                        st.error(f"⚠️ {opt['violations'][0] if opt['violations'] else 'Violates constraints'}")

                    if st.button(f"👉 EXECUTE {opt['id']}", key=f"btn_sel_{opt['id']}", disabled=not is_feas, use_container_width=True):
                        st.session_state.selected_option = opt
                        st.session_state.evaluation_result = None
                        st.rerun()

            # "Why These Recommendations?" Section
            st.markdown("---")
            st.markdown("#### 💡 Why These Recommendations? (Transparent Optimization Audit)")
            why = opt_res["why_explanation"]
            
            w_col1, w_col2 = st.columns([1, 1])
            with w_col1:
                st.markdown(f"""
                - **Baseline Predicted Delay:** `{why['predicted_delay']}`
                - **Hard Budget Limit:** `{why['hard_budget_limit']}`
                - **Required Delivery Deadline:** `{why['required_delivery_deadline']}`
                - **Optimization Rationale:** {why['rationale']}
                """)
            with w_col2:
                st.markdown(f"""
                - **Mathematical Objective:** `{why['optimization_objective']}`
                - **Estimated Disruption Cost Avoided:** <span style="color: #16A34A; font-weight: bold; font-size: 1.1rem;">{why['estimated_net_savings']}</span>
                """, unsafe_allow_html=True)

            # Cost vs Speed Trade-off Visual Chart
            st.markdown("##### 📊 Cost vs. Speed Operational Trade-Off")
            tradeoff_df = pd.DataFrame(opt_res["tradeoff_comparison"])
            
            fig_tradeoff = px.scatter(
                tradeoff_df,
                x="delivery_days",
                y="cost",
                text="option_id",
                color="status",
                size=[30, 24, 20],
                hover_data=["name", "additional_cost", "residual_delay", "risk_level"],
                labels={"delivery_days": "Delivery Time (Days) — Lower is Faster", "cost": "Action Cost ($) — Lower is Cheaper"},
                color_discrete_map={"FEASIBLE": "#0284C7", "INFEASIBLE": "#EF4444"},
            )
            # Add deadline and budget reference lines
            fig_tradeoff.add_vline(x=current_constraints["max_delivery_days"], line_dash="dash", line_color="orange", annotation_text="Max Delivery SLA")
            fig_tradeoff.add_hline(y=current_constraints["max_budget"], line_dash="dash", line_color="red", annotation_text="Max Budget")
            fig_tradeoff.update_traces(textposition="top center", textfont=dict(size=14, color="black"))
            fig_tradeoff.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_tradeoff, use_container_width=True)

        # -----------------------------------------------------
        # SECTION 4: DECISION EXECUTION (WRITE-BACK)
        # -----------------------------------------------------
        if st.session_state.selected_option is not None:
            sel_opt = st.session_state.selected_option
            st.markdown("---")
            st.markdown("### 4. Decision Execution (Simulated Write-Back)")
            
            st.success(f"""
            ✅ **DECISION EXECUTED SUCCESSFULLY**
            - **Selected Action:** `{sel_opt['name']}` ({sel_opt['id']})
            - **Target Cost:** `${sel_opt['cost']:,.2f}`
            - **Target Delivery:** `{sel_opt['delivery_days']:.1f} Days`
            - **Database Write-Back Status:** `COMMITTED TO OPERATIONAL DB (IN_TRANSIT_MITIGATED)`
            """)

            # -------------------------------------------------
            # SECTION 5: CLOSED-LOOP OUTCOME & EVALUATION
            # -------------------------------------------------
            st.markdown("---")
            st.markdown("### 5. Closed-Loop Outcome Entry & Evaluation")
            st.caption("Record what actually happened in the real world to evaluate prediction error, cost variance, and decision ROI.")

            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                actual_cost_val = st.number_input("Actual Recorded Cost ($)", value=float(sel_opt['cost'] * 1.06), step=250.0)
            with e_col2:
                actual_delivery_val = st.number_input("Actual Delivery Time (Days)", value=float(sel_opt['delivery_days'] + 0.5), step=0.5)
            with e_col3:
                actual_notes_val = st.text_input("Operational Notes / Root Cause", value="Minor port customs delay; completed smoothly.")

            if st.button("📊 EVALUATE DECISION (CLOSE THE LOOP)", type="primary", use_container_width=True):
                actual_payload = {
                    "actual_cost": actual_cost_val,
                    "actual_delivery_days": actual_delivery_val,
                    "notes": actual_notes_val,
                }
                with st.spinner("Calculating closed-loop variances and decision ROI..."):
                    cycle_res = log_and_evaluate_cycle(
                        st.session_state.last_shipment_data,
                        st.session_state.prediction_result,
                        sel_opt,
                        actual_payload,
                    )
                    st.session_state.evaluation_result = cycle_res["evaluation"]

            if st.session_state.evaluation_result is not None:
                ev = st.session_state.evaluation_result
                st.markdown("#### 🎯 Decision Outcome & Variance Analysis")
                
                ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
                with ev_col1:
                    cv_color = "normal" if ev["cost_variance"] <= 0 else "inverse"
                    st.metric(
                        "Cost Variance",
                        f"${ev['actual_cost']:,.2f}",
                        f"{ev['cost_variance']:+,.2f} ({ev['cost_variance_pct']:+.1f}%)",
                        delta_color=cv_color,
                    )
                with ev_col2:
                    dv_color = "normal" if ev["delivery_variance_days"] <= 0 else "inverse"
                    st.metric(
                        "Delivery Variance",
                        f"{ev['actual_delivery_days']:.1f} Days",
                        f"{ev['delivery_variance_days']:+.1f} Days",
                        delta_color=dv_color,
                    )
                with ev_col3:
                    st.metric("Days Saved vs Disruption", f"{ev['actual_days_saved']:.1f} Days", f"Penalties: ${ev['penalties_avoided']:,.0f}")
                with ev_col4:
                    st.metric("Mitigation ROI", f"{ev['mitigation_roi_pct']:+.1f}%", ev["outcome_category"])

                st.info(f"📋 **Outcome Summary:** {ev['qualitative_outcome']}")

                # Before / After Chart
                chart_data = pd.DataFrame([
                    {"Category": "Direct Cost ($)", "Target (Optimized)": ev["predicted_cost"], "Actual Outcome": ev["actual_cost"]},
                    {"Category": "Delivery Time (Days x1000)", "Target (Optimized)": ev["predicted_delivery_days"] * 1000, "Actual Outcome": ev["actual_delivery_days"] * 1000},
                ])
                fig_comp = go.Figure(data=[
                    go.Bar(name='Target / Predicted', x=['Cost ($)', 'Delivery Time (Days)'], y=[ev["predicted_cost"], ev["predicted_delivery_days"]], marker_color='#0284C7'),
                    go.Bar(name='Actual Outcome', x=['Cost ($)', 'Delivery Time (Days)'], y=[ev["actual_cost"], ev["actual_delivery_days"]], marker_color='#10B981' if ev["cost_variance"] <= 0 else '#F59E0B'),
                ])
                fig_comp.update_layout(barmode='group', height=280, margin=dict(l=20, r=20, t=30, b=20), title="Target vs Actual Realized Performance")
                st.plotly_chart(fig_comp, use_container_width=True)


# =============================================================
# TAB 2: MODEL PERFORMANCE & METRICS
# =============================================================
with tab_model_metrics:
    st.markdown("### 📈 Machine Learning Model Performance & Diagnostics")
    metrics = get_model_metrics()
    if metrics:
        cls_m = metrics.get("classification", {})
        reg_m = metrics.get("regression", {})

        st.markdown("#### 1. Classification Metrics (Delay Risk Detection)")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Accuracy", f"{cls_m.get('accuracy', 0) * 100:.2f}%")
        m_col2.metric("Precision", f"{cls_m.get('precision', 0) * 100:.2f}%")
        m_col3.metric("Recall", f"{cls_m.get('recall', 0) * 100:.2f}%")
        m_col4.metric("F1 Score", f"{cls_m.get('f1_score', 0) * 100:.2f}%")
        m_col5.metric("ROC-AUC", f"{cls_m.get('roc_auc', 0):.4f}")

        st.markdown("#### 2. Regression Metrics (Delay Duration Estimation)")
        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("Mean Absolute Error (MAE)", f"{reg_m.get('mae', 0):.2f} Days")
        r_col2.metric("Root Mean Squared Error (RMSE)", f"{reg_m.get('rmse', 0):.2f} Days")
        r_col3.metric("R² Score", f"{reg_m.get('r2_score', 0):.4f}")

        st.markdown("#### 3. Feature Importance Breakdown (XGBoost)")
        fi_df = pd.DataFrame(metrics.get("feature_importances", [])[:10])
        if not fi_df.empty:
            fig_fi = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
                title="Top 10 Feature Importances in XGBoost Disruption Model",
            )
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=380)
            st.plotly_chart(fig_fi, use_container_width=True)


# =============================================================
# TAB 3: OPTIMIZATION STRESS TEST & AUDIT
# =============================================================
with tab_audit_stress:
    st.markdown("### 🔬 Optimization Engine Stress Testing & Hard Constraint Audit")
    st.markdown("Test how the PuLP optimization solver responds to extreme or impossible business constraints.")

    st_col1, st_col2 = st.columns(2)
    with st_col1:
        test_case = st.selectbox(
            "Select Constraint Stress Scenario:",
            [
                "Strict Budget Limit (Max $10,000) — Tests Budget Feasibility",
                "Impossible Fast SLA (Max 2.0 Days) — Tests Air Freight Feasibility",
                "Completely Impossible Constraints ($2,000 Budget & 1 Day SLA)",
                "Relaxed High-Budget Constraints ($50,000 & 20 Days)",
            ],
        )

    # Base test shipment
    test_shipment = {
        "shipping_cost": 7500.0,
        "lead_time_days": 15.0,
        "order_quantity": 400,
        "route_distance_km": 9000.0,
    }
    test_prediction = {"predicted_delay_days": 10.0, "delay_probability": 0.90, "risk_level": "CRITICAL"}

    if "Strict Budget" in test_case:
        test_c = OptimizationConstraints(max_budget=10000.0, max_delivery_days=25.0)
    elif "Impossible Fast" in test_case:
        test_c = OptimizationConstraints(max_budget=35000.0, max_delivery_days=2.0)
    elif "Completely Impossible" in test_case:
        test_c = OptimizationConstraints(max_budget=2000.0, max_delivery_days=1.0)
    else:
        test_c = OptimizationConstraints(max_budget=50000.0, max_delivery_days=20.0)

    if st.button("🧪 RUN OPTIMIZATION AUDIT TEST", type="primary"):
        test_res = optimize_actions(test_shipment, test_prediction, test_c)
        
        st.markdown(f"**Solver Status:** `{test_res['solver_status']}` | **Optimal Found:** `{test_res['is_optimal']}`")
        st.markdown(f"**Recommended Action:** `{test_res['why_explanation']['recommended_action']}`")
        
        for opt in test_res["options"]:
            st.markdown(f"- **{opt['id']} ({opt['name']})**: Status = `{opt['status']}` | Cost = `${opt['cost']:,.2f}` | Delivery = `{opt['delivery_days']:.1f}d` | Violations: *{opt['audit_summary']}*")


# =============================================================
# TAB 4: CLOSED-LOOP HISTORY & RETRAINING
# =============================================================
with tab_history:
    st.markdown("### 🔄 Closed-Loop Decision History & Model Feedback")
    history_data = get_history()
    analytics = get_feedback_analytics()

    a_col1, a_col2, a_col3, a_col4 = st.columns(4)
    a_col1.metric("Total Decisions Logged", analytics["total_decisions"])
    a_col2.metric("Average Cost Variance", f"${analytics['avg_cost_variance']:+,.2f}")
    a_col3.metric("Average Delivery Variance", f"{analytics['avg_delivery_variance']:+.1f} Days")
    a_col4.metric("Average Mitigation ROI", f"{analytics['avg_mitigation_roi_pct']:+.1f}%")

    if analytics["retraining_recommended"]:
        st.warning("⚠️ **Model Drift Detected / Sufficient History Accumulated:** Retraining is recommended to align predictions with actual realized lead times.")
    else:
        st.success("✅ **Model Performance Stable:** Predictions remain well-calibrated with operational outcomes.")

    if st.button("🔄 TRIGGER MODEL RETRAINING WITH FEEDBACK", type="secondary"):
        with st.spinner("Retraining XGBoost models on updated closed-loop weights..."):
            retrain_res = retrain_model_with_feedback()
            st.success(retrain_res["message"])
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📜 Recorded Operational Decision Logs")
    if history_data:
        flat_records = []
        for h in history_data:
            flat_records.append({
                "Event ID": h["event_id"],
                "Shipment ID": h["shipment_id"],
                "Supplier": h["supplier_name"],
                "Predicted Risk": f"{h['prediction']['risk_level']} ({h['prediction']['delay_probability']*100:.0f}%)",
                "Action Taken": h["decision"]["name"],
                "Target Cost": f"${h['decision']['cost']:,.2f}",
                "Actual Cost": f"${h['actual_outcome']['actual_cost']:,.2f}",
                "Cost Variance": f"${h['evaluation']['cost_variance']:+,.2f}",
                "Target Delivery": f"{h['decision']['delivery_days']:.1f}d",
                "Actual Delivery": f"{h['actual_outcome']['actual_delivery_days']:.1f}d",
                "Mitigation ROI": f"{h['evaluation']['mitigation_roi_pct']:+.1f}%",
                "Outcome": h["evaluation"]["outcome_category"],
            })
        st.dataframe(pd.DataFrame(flat_records), use_container_width=True)
    else:
        st.info("No decision records found yet. Execute and evaluate a decision in Tab 1 to populate history.")
