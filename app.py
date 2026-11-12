"""Supply Prescript — Closed-Loop Prescriptive Supply-Chain Analytics Control Tower.

Enterprise Pair-Programming Implementation (Days 10-20).
Workflow: Predict -> Assess Risk -> Optimize -> Recommend -> Decide -> Execute -> Measure -> Learn.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

from src.dashboard.api_client import DashboardApiClient

st.set_page_config(
    page_title="Supply Prescript | Prescriptive Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

api = DashboardApiClient()

# Sidebar Navigation
st.sidebar.title("🛡️ Supply Prescript")
st.sidebar.caption("Prescriptive Decision-Support Platform")

# System Health Pill
health = api.check_health()
h_color = "#10b981" if health.get("status") == "healthy" else "#f59e0b"
st.sidebar.markdown(f"**System Status:** <span style='color:{h_color}'>● {health.get('status', 'online').upper()}</span>", unsafe_allow_html=True)
st.sidebar.divider()

nav_page = st.sidebar.radio(
    "Navigation",
    [
        "1. Executive Overview",
        "2. Supply Chain Monitoring",
        "3. Predictive Analytics",
        "4. Risk Monitoring",
        "5. Optimization Workspace",
        "6. Recommendation Comparison",
        "7. Decision Management",
        "8. Outcome Tracking",
        "9. Feedback & Learning",
        "⚡ Core Scenario Demo",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "**Closed-Loop Workflow:**\n"
    "Predict → Risk → Optimize → Recommend → Decide → Execute → Measure → Learn"
)


# ==============================================================================
# 1. EXECUTIVE OVERVIEW
# ==============================================================================
if nav_page == "1. Executive Overview":
    st.markdown("<div class='main-header'>Executive Overview Control Tower</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>High-level operational metrics across all supply disruption events, decisions, and variances.</div>", unsafe_allow_html=True)

    shipments = api.get_shipments()
    predictions = api.get_predictions()
    risks = api.get_risk_assessments()
    decisions = api.get_decisions()
    outcomes = api.get_outcomes()

    # Top KPI Metrics Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Supply Events", len(shipments))
    high_risks = sum(1 for r in risks if r["risk_level"] in ("HIGH", "CRITICAL"))
    c2.metric("Critical / High Risks", high_risks)
    crit_risks = sum(1 for r in risks if r["risk_level"] == "CRITICAL")
    c3.metric("Critical Shipments", crit_risks)
    pending_decs = sum(1 for d in decisions if d["decision_status"] == "PENDING")
    c4.metric("Pending Decisions", pending_decs)
    completed_exec = sum(1 for d in decisions if d["execution_status"] == "COMPLETED")
    c5.metric("Completed Executions", completed_exec)
    avg_cost_var = (sum(float(o.get("cost_variance", 0.0) or 0.0) for o in outcomes) / len(outcomes)) if outcomes else 0.0
    c6.metric("Avg Cost Variance", f"${avg_cost_var:,.2f}", delta=f"${avg_cost_var:,.0f}", delta_color="inverse")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Operational Risk Distribution")
        if risks:
            df_risk = pd.DataFrame(risks)
            fig_risk = px.pie(
                df_risk,
                names="risk_level",
                color="risk_level",
                color_discrete_map={"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#f97316", "CRITICAL": "#ef4444"},
                hole=0.4,
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("No risk assessments calculated yet.")

    with col_right:
        st.subheader("Executed Strategies Performance")
        if outcomes:
            df_out = pd.DataFrame(outcomes)
            if "cost_variance" in df_out.columns:
                df_out["cost_variance"] = pd.to_numeric(df_out["cost_variance"], errors="coerce").fillna(0.0)
            if "strategy" not in df_out.columns:
                df_out["strategy"] = "AIR_FREIGHT"
            fig_strat = px.bar(
                df_out,
                x="strategy",
                y="cost_variance",
                color="outcome_status" if "outcome_status" in df_out.columns else None,
                title="Cost Variance by Strategy ($)",
                color_discrete_map={"SUCCESS": "#10b981", "PARTIAL_SUCCESS": "#f59e0b", "FAILED": "#ef4444"},
            )
            st.plotly_chart(fig_strat, use_container_width=True)
        else:
            st.info("No post-execution outcomes recorded yet.")


# ==============================================================================
# 2. SUPPLY CHAIN MONITORING
# ==============================================================================
elif nav_page == "2. Supply Chain Monitoring":
    st.markdown("<div class='main-header'>Supply Chain Monitoring</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Operational shipment visibility, supplier capacity utilization, and warehouse stock levels.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📦 Shipments & Orders", "🏭 Approved Suppliers", "🏢 Products & Inventory"])

    with tab1:
        shipments = api.get_shipments()
        if shipments:
            df_s = pd.DataFrame(shipments)
            # Filter bar
            search = st.text_input("Filter shipments by Event ID or Product Name", "")
            if search:
                df_s = df_s[df_s["event_id"].str.contains(search, case=False) | df_s["product_name"].str.contains(search, case=False)]
            st.dataframe(df_s, use_container_width=True, hide_index=True)
        else:
            st.warning("No shipment records found.")

    with tab2:
        suppliers = api.get_suppliers()
        if suppliers:
            df_sup = pd.DataFrame(suppliers)
            st.dataframe(df_sup, use_container_width=True, hide_index=True)
            fig_cap = px.bar(df_sup, x="supplier_name", y="supplier_capacity_units", color="reliability_score", title="Supplier Capacity vs Reliability Score")
            st.plotly_chart(fig_cap, use_container_width=True)
        else:
            st.warning("No suppliers found.")

    with tab3:
        prods = api.get_products()
        if prods:
            df_p = pd.DataFrame(prods)
            st.dataframe(df_p, use_container_width=True, hide_index=True)


# ==============================================================================
# 3. PREDICTIVE ANALYTICS
# ==============================================================================
elif nav_page == "3. Predictive Analytics":
    st.markdown("<div class='main-header'>Predictive Machine Learning Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Supervised delay probabilities, duration forecasts, and model evaluation metrics.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Run Delay Inference")
        event_choice = st.selectbox("Select Shipment Event", ["EVT-001", "EVT-002", "EVT-003", "EVT-004"])
        if st.button("Generate ML Prediction", type="primary"):
            with st.spinner("Running calibrated Random Forest/XGBoost inference..."):
                res = api.run_prediction(event_choice)
                st.success(f"Prediction generated: P(Delay) = {res['delay_probability']:.2f} ({res['risk_level']})")

    with c2:
        st.subheader("Prediction History")
        preds = api.get_predictions()
        if preds:
            df_p = pd.DataFrame(preds)
            st.dataframe(df_p, use_container_width=True, hide_index=True)
        else:
            st.info("No predictions recorded yet.")


# ==============================================================================
# 4. RISK MONITORING
# ==============================================================================
elif nav_page == "4. Risk Monitoring":
    st.markdown("<div class='main-header'>Operational Risk Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Multidimensional operational risk scoring: Delay, Supplier, Inventory, Capacity, Service.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Trigger Risk Assessment")
        evt = st.selectbox("Select Event for Risk Assessment", ["EVT-001", "EVT-002", "EVT-003"])
        if st.button("Evaluate Multidimensional Risk", type="primary"):
            with st.spinner("Evaluating operational risk dimensions..."):
                r = api.run_risk_assessment(evt)
                st.success(f"Assessment complete: Composite Risk = {r['composite_risk_score']:.2f} ({r['risk_level']})")

    with col2:
        risks = api.get_risk_assessments()
        if risks:
            df_r = pd.DataFrame(risks)
            st.dataframe(df_r[["event_id", "composite_risk_score", "risk_level", "risk_reasons"]], use_container_width=True, hide_index=True)
        else:
            st.info("No risk assessments available.")


# ==============================================================================
# 5. OPTIMIZATION WORKSPACE
# ==============================================================================
elif nav_page == "5. Optimization Workspace":
    st.markdown("<div class='main-header'>Prescriptive Optimization Workspace</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Formulate Mixed-Integer Linear Program (MILP) with PuLP & CBC Solver.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Configure & Solve Problem")
        sel_event = st.selectbox("Select Disrupted Shipment", ["EVT-001", "EVT-002"])
        obj_mode = st.selectbox("Optimization Objective", ["MIN_COST", "MIN_RISK", "MIN_DELAY", "BALANCED"])

        if st.button("Execute Prescriptive Solver", type="primary"):
            with st.spinner("Executing PuLP CBC solver..."):
                run_res = api.run_optimization(sel_event, objective_type=obj_mode)
                st.success(f"Solver Status: {run_res['solver_status']} | Run ID: {run_res['optimization_run_id']}")
                st.session_state["active_run_id"] = run_res["optimization_run_id"]

    with c2:
        st.subheader("Active Optimization Run")
        active_id = st.session_state.get("active_run_id", "OPT-RUN-001")
        st.text_input("Optimization Run ID", active_id, key="run_id_input")
        recs_data = api.get_recommendations(st.session_state.get("run_id_input", active_id))

        if recs_data.get("recommendations"):
            df_rec = pd.DataFrame(recs_data["recommendations"])
            st.dataframe(
                df_rec[["recommendation_code", "strategy_type", "estimated_cost", "estimated_delay_days", "estimated_risk", "rank", "is_recommended"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Run solver or enter valid Optimization Run ID to inspect solutions.")


# ==============================================================================
# 6. RECOMMENDATION COMPARISON
# ==============================================================================
elif nav_page == "6. Recommendation Comparison":
    st.markdown("<div class='main-header'>Prescriptive Recommendation Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Detailed trade-off analysis across Air Freight, Secondary Supplier, and Postponement.</div>", unsafe_allow_html=True)

    run_id = st.text_input("Enter Optimization Run ID", st.session_state.get("active_run_id", "OPT-RUN-001"))
    data = api.get_recommendations(run_id)

    if data.get("recommendations"):
        recs = data["recommendations"]

        st.subheader("Multi-Alternative Decision Matrix")
        df_comp = pd.DataFrame(data.get("comparison_table", []))
        if not df_comp.empty:
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.subheader("Strategy Trade-Off Cards")
        cols = st.columns(len(recs))
        for i, rec in enumerate(recs):
            with cols[i]:
                rec_badge = "⭐ RECOMMENDED" if rec["is_recommended"] else f"Rank #{rec['rank']}"
                st.markdown(f"### {rec['recommendation_code']} — {rec['strategy_type']}")
                st.caption(rec_badge)
                st.metric("Estimated Cost", f"${rec['estimated_cost']:,.2f}")
                st.metric("Delivery Lead Time", f"{rec['estimated_delay_days']:.1f} days")
                st.metric("Operational Risk", f"{rec['estimated_risk']:.2f}")

                with st.expander("Advantages"):
                    for a in rec.get("advantages", []):
                        st.write(f"- {a}")
                with st.expander("Disadvantages"):
                    for d in rec.get("disadvantages", []):
                        st.write(f"- {d}")
                with st.expander("Reasoning"):
                    st.write(rec.get("reasoning", ""))
    else:
        st.warning("No recommendations found for this Optimization Run ID.")


# ==============================================================================
# 7. DECISION MANAGEMENT
# ==============================================================================
elif nav_page == "7. Decision Management":
    st.markdown("<div class='main-header'>Managerial Decision Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Human-in-the-loop decision approval, override justification, and execution tracking.</div>", unsafe_allow_html=True)

    tab_review, tab_history = st.tabs(["✍️ Record / Approve Decision", "📜 Decision History & Lifecycle"])

    with tab_review:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Decision Form")
            run_id = st.text_input("Optimization Run ID", st.session_state.get("active_run_id", "OPT-RUN-001"), key="dec_run_id")
            recs_data = api.get_recommendations(run_id)
            recs = recs_data.get("recommendations", [])

            rec_options = {f"{r['recommendation_code']} - {r['strategy_type']} (${r['estimated_cost']:,.2f})": r for r in recs} if recs else {}
            selected_choice = st.selectbox("Select Recovery Strategy", list(rec_options.keys()) if rec_options else ["None"])

            maker = st.text_input("Decision Maker (Authorized Name)", "Global Operations Director")
            status_choice = st.selectbox("Decision Status", ["APPROVED", "OVERRIDDEN", "REJECTED"])
            reason = st.text_area("Decision / Override Reason", "Approved recommended air freight charter to eliminate 14-day microchip delay.")

            confirm = st.checkbox("I confirm this operational decision should be committed to PostgreSQL.")

            if st.button("Submit Decision", type="primary", disabled=not confirm):
                if rec_options and selected_choice in rec_options:
                    chosen_rec = rec_options[selected_choice]
                    with st.spinner("Recording decision..."):
                        dec_res = api.record_decision(
                            run_id=1,  # Resolves in API client
                            rec_id=1,
                            maker=maker,
                            status=status_choice,
                            reason=reason,
                        )
                        st.success(f"Decision recorded successfully! Code: {dec_res.get('decision_id', 'DEC-001')}")
                else:
                    st.error("Please load a valid optimization run first.")

        with c2:
            st.subheader("Update Execution Status")
            decs = api.get_decisions()
            if decs:
                dec_map = {f"{d.get('decision_id', 'DEC')} ({d.get('strategy', 'STRATEGY')}) - Status: {d.get('execution_status', 'STATUS')}": d for d in decs}
                sel_dec_label = st.selectbox("Select Decision to Update", list(dec_map.keys()))
                target_dec = dec_map[sel_dec_label]

                curr_status = target_dec.get("execution_status", "NOT_STARTED")
                st.write(f"**Current Status:** `{curr_status}`")

                next_status = st.selectbox("New Execution Status", ["IN_PROGRESS", "COMPLETED", "FAILED"])
                if st.button("Update Execution Status"):
                    api.update_decision_execution(target_dec["id"], next_status)
                    st.success(f"Updated {target_dec.get('decision_id', 'DEC')} to {next_status}!")
                    st.rerun()

    with tab_history:
        st.subheader("Decision Audit Trail")
        decs = api.get_decisions()
        if decs:
            st.dataframe(pd.DataFrame(decs), use_container_width=True, hide_index=True)
        else:
            st.info("No decisions recorded yet.")


# ==============================================================================
# 8. OUTCOME TRACKING
# ==============================================================================
elif nav_page == "8. Outcome Tracking":
    st.markdown("<div class='main-header'>Post-Execution Outcome Evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Record actual costs and lead times, calculate operational variances, and classify success.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Record Invoiced Actuals")
        decs = api.get_decisions()
        completed_decs = [d for d in decs if d.get("execution_status") == "COMPLETED"]

        if completed_decs:
            dec_map = {f"{d.get('decision_id', 'DEC')} - {d.get('strategy', 'STRATEGY')}": d for d in completed_decs}
            sel_label = st.selectbox("Select Completed Decision", list(dec_map.keys()))
            sel_dec = dec_map[sel_label]

            act_cost = st.number_input("Actual Invoiced Cost ($)", min_value=0.0, value=18000.0, step=500.0)
            act_delay = st.number_input("Actual Delivery Time (Days)", min_value=0.0, value=2.0, step=0.5)
            act_sla = st.slider("Actual Service Level Delivered", 0.0, 1.0, 1.0)
            notes = st.text_area("Operational Notes", "Air charter delivered microchips within 48h; incurred $3,000 spot surcharge.")

            if st.button("Record Actual Outcome", type="primary"):
                with st.spinner("Recording actual outcome and computing variances..."):
                    out_res = api.record_outcome(sel_dec["id"], act_cost, act_delay, act_sla, notes)
                    var_val = float(out_res.get("cost_variance", 0.0) or 0.0)
                    st.success(f"Outcome recorded! Status: {out_res.get('outcome_status', 'SUCCESS')} | Cost Var: +${var_val:,.2f}")
        else:
            st.warning("No decisions currently marked 'COMPLETED'. Advance execution status in Decision Management first.")

    with c2:
        st.subheader("Recorded Outcomes & Variances")
        outs = api.get_outcomes()
        if outs:
            st.dataframe(pd.DataFrame(outs), use_container_width=True, hide_index=True)
        else:
            st.info("No outcomes evaluated yet.")


# ==============================================================================
# 9. FEEDBACK & LEARNING
# ==============================================================================
elif nav_page == "9. Feedback & Learning":
    st.markdown("<div class='main-header'>Closed-Loop Feedback & Prescriptive Learning</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Calibrate future cost multipliers using Exponentially Weighted Moving Average (EWMA).</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Process Outcome into Feedback")
        outs = api.get_outcomes()
        if outs:
            out_map = {f"Outcome #{o.get('id', i)} ({o.get('strategy', 'STRATEGY')}) - Var: ${float(o.get('cost_variance', 0.0) or 0.0):,.2f}": o for i, o in enumerate(outs)}
            sel_out = st.selectbox("Select Outcome to Close Loop", list(out_map.keys()))
            target_out = out_map[sel_out]

            if st.button("Trigger Feedback Calibration", type="primary"):
                with st.spinner("Processing feedback signal & safeguards..."):
                    fb = api.trigger_feedback(target_out["id"])
                    st.success(f"Feedback Processed! Signal: {fb.get('learning_signal', 'OK')}")
        else:
            st.warning("No outcomes available. Record post-execution actuals first.")

    with c2:
        st.subheader("Historical Feedback Signals")
        fbs = api.get_feedback()
        if fbs:
            st.dataframe(pd.DataFrame(fbs), use_container_width=True, hide_index=True)
        else:
            st.info("No feedback records generated yet.")


# ==============================================================================
# 10. CORE SCENARIO DEMONSTRATION WALKTHROUGH
# ==============================================================================
elif nav_page == "⚡ Core Scenario Demo":
    st.markdown("<div class='main-header'>⚡ Core Microchip X1 Demonstration Walkthrough</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Full end-to-end simulation of the 8-stage closed-loop decision lifecycle.</div>", unsafe_allow_html=True)

    st.info(
        "**Scenario Specification:**\n"
        "- Critical microchip shipment (`EVT-001`) predicted with a 14-day disruption.\n"
        "- Risk Engine classifies shipment as CRITICAL risk.\n"
        "- Optimization evaluates Air Freight ($15,000, 2 days), Secondary Supplier (10% premium, 7 days), and Launch Delay ($21,800, 14 days).\n"
        "- Manager approves Air Freight.\n"
        "- Invoiced actual cost arrives at $18,000 (+$3,000 variance).\n"
        "- Prescriptive learning loop calculates variance and updates future optimization assumptions!"
    )

    if st.button("🚀 Run Complete Closed-Loop Demonstration", type="primary", use_container_width=True):
        with st.spinner("Executing end-to-end prescriptive analytics workflow..."):
            with api._get_db() as db:
                from src.workflows.orchestration import WorkflowOrchestrator
                orch = WorkflowOrchestrator()
                res = orch.execute_closed_loop_workflow(
                    db=db,
                    event_id="EVT-001",
                    decision_maker="VP of Global Supply Chain",
                    selected_strategy_code="ALT_A",
                    actual_cost=18000.0,
                    actual_delay_days=2.0,
                    actual_service_level=1.0,
                )

                if res.overall_status == "COMPLETED":
                    st.success(f"🎉 Closed-Loop Prescriptive Workflow Completed Successfully! (ID: {res.workflow_id})")

                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("1. Delay Probability", f"{res.prediction_summary['delay_probability']:.2f}", res.prediction_summary['risk_level'])
                    s2.metric("2. Composite Risk", f"{res.risk_summary['composite_risk_score']:.2f}", res.risk_summary['risk_level'])
                    s3.metric("3. Selected Recovery", res.recommended_strategy)
                    s4.metric("4. Cost Variance", f"+${res.outcome_summary['cost_variance']:,.2f}", "Overrun", delta_color="inverse")

                    st.subheader("Step-by-Step Execution Audit Trail")
                    for s in res.steps:
                        with st.expander(f"Stage {s.step_number}: {s.step_name} — {s.status} ({s.execution_time_ms:.1f}ms)", expanded=True):
                            st.json(s.output_summary)

                    st.subheader("Prescriptive Learning Signal")
                    st.info(f"**Closed-Loop Signal:** `{res.feedback_summary['learning_signal']}`")
                else:
                    st.error(f"Workflow encountered an error: {res.error_detail}")

