# 📦 Supply Prescript — Project Guide & Operational Walkthrough

**Closed-Loop Prescriptive Analytics for Resilient Supply Chain Operations**  
*Lead Author & Engineer: Prasanna (ML & Optimization Lead)*  
*Team Members: Prasanna (ML/PuLP/Demo), Trisha (Backend/DB/Write-Back), Nanditha (Operational UI)*

---

## 📌 Executive Summary

Modern supply chains are highly vulnerable to disruptions—such as port congestions, supplier delays, extreme weather, and unexpected demand spikes. Most enterprise analytics dashboards are **open-loop** and strictly **predictive**: they forecast a disruption (e.g., *"85% probability of a 14-day delay"*), but leave human operators guessing the optimal response without visibility into cost-time trade-offs or actual post-execution outcomes.

**Supply Prescript** closes this gap by transforming predictive signals into mathematically optimal prescriptive actions, recording operator execution, measuring real-world outcomes, and continuously learning from execution variance.

```
       Traditional Open-Loop Dashboards                   Supply Prescript (Closed-Loop)
┌──────────────────────────────────────────┐          ┌───────────────────────────────────┐
│  Data  ➔  Predict  ➔  Static Dashboard   │    VS    │  Data  ➔  Predict  ➔  Prescribe   │
│       (Human guesses mitigation action)  │          │    ➔ Decide ➔ Execute ➔ Measure   │
└──────────────────────────────────────────┘          │          ➔ Feed Forward ↺         │
                                                      └───────────────────────────────────┘
```

---

## 🔁 The 6-Stage Closed-Loop Architecture

The platform executes a continuous 6-stage operational cycle:

1. **Ingest Supply Chain Signals:** Real-time data from ERP, logistics routes, supplier lead times, weather indices, port congestion, and warehouse inventory.
2. **Predictive Risk Assessment (`ml/`):**
   - **XGBoost Classifier (`delay_classifier.joblib`):** Estimates delay likelihood and assigns risk tiers (*LOW*, *MEDIUM*, *HIGH*, *CRITICAL*). Accuracy: **96.1%**.
   - **XGBoost Regressor (`delay_regressor.joblib`):** Quantifies exact delay duration in days. Mean Absolute Error (MAE): **0.67 Days**.
   - Identifies top risk drivers (e.g., port congestion index, route distance, weather risk).
3. **Prescriptive Optimization (`optimization/`):**
   - Mixed-Integer Linear Programming (MILP) formulated with **PuLP**.
   - Formulates cost-minimization objectives subject to hard business constraints:
     - **Budget Cap:** Total mitigation cost cannot exceed maximum allowable budget.
     - **Delivery SLA:** Required arrival lead time must be within SLA deadlines.
     - **Safety Stock / Inventory:** Protects baseline warehouse buffer levels.
     - **Supplier Capacity:** Limits order allocation to verified vendor bandwidth.
   - Produces 3 vetted mitigation alternatives:
     - **Option A (Expedited Air Freight):** Maximum lead-time reduction, premium cost.
     - **Option B (Dual-Sourcing Split):** 60/40 volume distribution to balance cost and risk.
     - **Option C (Buffer Stock & Port Rerouting):** Minimum cost alternative utilizing safety stock.
4. **Human-in-the-Loop Decision & Execution (`api/` & `frontend/`):**
   - Visualizes multi-criteria trade-offs (Cost vs. Lead Time Pareto frontier).
   - Operator selects the best option or approves the MILP optimal recommendation.
   - Logs operational execution details to the database and audit trail.
5. **Outcome Evaluation (`evaluation/evaluator.py`):**
   - Post-transit arrival metrics are ingested.
   - Calculates **Cost Variance Delta** ($\text{Actual Cost} - \text{Predicted Cost}$), **SLA Recovery Rate**, and **Total Delay Avoided**.
6. **Continuous Feedback Learning (`evaluation/feedback.py`):**
   - Logs execution outcomes into `data/closed_loop_history.json`.
   - Empirically calibrates optimization safety margins and retraining weights to eliminate recurring systematic errors.

---

## 👥 Three-Member Ownership & Responsibilities

| Contributor | Role | Core Deliverables |
| :--- | :--- | :--- |
| **Prasanna** | **ML & Optimization Lead** | • Predictive Modeling (XGBoost Classifier & Regressor)<br>• Prescriptive Engine (PuLP MILP solver & constraint auditor)<br>• Interactive Streamlit Demonstration Dashboard (`frontend/app.py`)<br>• Closed-loop evaluation and parameter recalibration engine (`evaluation/`) |
| **Trisha** | **Backend & Database Engineer** | • FastAPI REST service architecture (`api/main.py`)<br>• PostgreSQL schema, migrations (Alembic), and connection pool<br>• Decision logging and write-back audit tables |
| **Nanditha** | **Operational UI & Product Lead** | • Operational Dashboard UI and scenario flows<br>• Data dictionary and schema documentation<br>• Comprehensive system validation |

---

## 🚀 Step-by-Step Guide: How to Run the Project

### Prerequisites
- Python 3.10+ or 3.11+
- Virtual environment with all dependencies satisfied (Streamlit, FastAPI, XGBoost, PuLP, Plotly, SQLAlchemy, Alembic, Psycopg)

### Step 1: Run the Automated Dependency & Test Check
Verify that all 28 automated tests across ML, optimization, database, and API contracts pass:
```powershell
.\venv\Scripts\pytest.exe
```

### Step 2: Launch the Interactive Demo Dashboard
Run the one-click demo launcher:
```powershell
.\venv\Scripts\python.exe run_demo.py
```
*Alternatively, start Streamlit directly:*
```powershell
.\venv\Scripts\streamlit.exe run frontend/app.py
```
* **Dashboard URL:** Open `http://localhost:8501` in your web browser.

### Step 3: (Optional) Launch the FastAPI REST Service
If showcasing the headless API backend to technical evaluators:
```powershell
.\venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
* **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
* **Root Status Endpoint:** `http://127.0.0.1:8000/`

---

## 🎯 Step-by-Step Showcase / Presentation Script

Follow this structured walk-through when presenting the project:

### 1. The Hook (30 Seconds)
> *"Welcome to Supply Prescript. Traditional supply chain platforms alert you that a shipment will be late, but don't tell you how to resolve it under tight budgets and deadlines. We built a closed-loop prescriptive engine that predicts disruptions, computes optimal mitigation actions via linear programming, logs operator decisions, and learns from post-execution results."*

### 2. Tab 1: Live Demo Flow (Steps 1 to 5)
1. **Select a Disruption Preset:**
   - In the sidebar, select **"🚨 Scenario 1: Critical Semiconductor Delay (Shenzhen → LA)"**.
   - Highlight the high-risk conditions: Weather Index = 0.75, Port Congestion = 0.70, ocean transit, budget limit = \$24,000, deadline = 10 days.
2. **Execute Step 2 — Run ML Delay Prediction:**
   - Click the prediction button.
   - Observe the **XGBoost** output: Delay probability >90% (**CRITICAL**), predicted delay of ~12–14 days.
   - Point out the **Risk Drivers Chart**: Congestion and weather dominate the risk score.
3. **Execute Step 3 — Prescriptive Optimization:**
   - Click **Generate Prescriptive Mitigation Options**.
   - Show how the **PuLP MILP solver** evaluates the solution space against budget and lead-time constraints.
   - Review the 3 mitigation cards (Air Freight, Dual-Sourcing, Buffer Stock & Reroute).
   - Point out the **Cost vs. Lead Time Trade-Off Plot** illustrating the Pareto-efficient frontier.
4. **Execute Step 4 — Operator Decision & Write-Back:**
   - Choose the recommended optimal option (Expedited Air Freight).
   - Click **Execute Decision**.
   - Explain how this writes back to the audit table and simulates dispatch.
5. **Execute Step 5 — Closed-Loop Outcome Evaluation:**
   - Click **Evaluate Closed-Loop Outcome**.
   - Review the post-execution scorecard: **Cost Variance Delta**, **SLA Recovery Rate**, and **Delay Avoided**.

### 3. Tab 2: Model Performance & Metrics
- Navigate to the **"📈 Model Performance & Metrics"** tab.
- Showcase the model validation metrics:
  - Classification Accuracy: **96.1%**
  - Regression MAE: **0.67 Days**
  - Feature Importance rankings and confusion matrix.

### 4. Tab 3: Optimization Hard-Constraints Audit
- Navigate to the **"🔬 Optimization Hard-Constraints Audit"** tab.
- Demonstrate constraint stress testing:
  - Tighten the budget below \$3,000 or set deadline to 2 days.
  - Show how the solver detects **INFEASIBLE** configurations and activates penalty-bounded fallback paths.

### 5. Tab 4: Closed-Loop History & Retraining
- Navigate to the **"🔄 Closed-Loop History & Retraining"** tab.
- Highlight the persistence log in `data/closed_loop_history.json`.
- Explain how empirical variances feed back into future optimization rounds to eliminate open-loop drift.

---

## 📁 Repository Structure

```text
SupplyPrescript/
├── api/                  # FastAPI REST endpoints and clean interface contracts
│   ├── interfaces.py     # Clean contract layer for ML, optimization, and evaluation
│   └── main.py           # FastAPI application entrypoint
├── data/                 # Raw and processed datasets, audit logs
│   ├── closed_loop_history.json
│   ├── supply_chain_data.csv
│   └── project_explanation.json
├── database/             # Relational schema, models, connection pool
├── evaluation/           # Evaluation metrics, feedback loop, and variance calculator
│   ├── evaluator.py      # Computes cost variance, SLA recovery, ROI
│   └── feedback.py       # Manages persistent history and calibration
├── frontend/             # Interactive Streamlit application
│   └── app.py            # Complete multi-tab decision dashboard
├── ml/                   # Machine learning models, training, and inference
│   ├── models/           # Serialized XGBoost joblib artifacts and metrics.json
│   ├── predictor.py      # Delay inference service
│   └── train_model.py    # Training and validation pipeline
├── optimization/         # Prescriptive MILP solver and constraints
│   ├── constraints.py    # Constraint definitions and validators
│   ├── optimizer.py      # PuLP formulation minimizing total mitigation cost
│   └── validator.py      # Audits and stress tests solutions
├── tests/                # 28 passing unit and integration tests
├── PROJECT_GUIDE.md      # Comprehensive walkthrough and showcase guide
├── requirements.txt      # Verified project dependencies
└── run_demo.py           # Single-command launcher script
```

---
*Created for the Supply Prescript project demonstration.*
