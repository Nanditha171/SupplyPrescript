# Supply Prescript: Closed-Loop Prescriptive Supply-Chain Decision Platform

[![Tests](https://img.shields.io/badge/tests-100%2F100%20passed-brightgreen.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg?logo=fastapi)]()
[![PuLP](https://img.shields.io/badge/PuLP-CBC%20Solver-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Control%20Tower-FF4B4B.svg?logo=streamlit)]()
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-red.svg)]()

> Intelligent supply-chain decision-support platform transforming predictive disruption signals into mathematically optimized operational recovery decisions with post-execution outcome learning.

---

## 1. Project Overview

**Supply Prescript** closes the gap between predictive analytics and operational decision-making. Conventional supply chain platforms stop at forecasting delays—leaving managers to guess the financial and schedule trade-offs of mitigation options. Supply Prescript implements a true closed-loop decision lifecycle:

$$\mathbf{Predict} \longrightarrow \mathbf{Assess\ Risk} \longrightarrow \mathbf{Optimize} \longrightarrow \mathbf{Recommend} \longrightarrow \mathbf{Decide} \longrightarrow \mathbf{Execute} \longrightarrow \mathbf{Measure} \longrightarrow \mathbf{Learn}$$

---

## 2. Business Problem

Global supply chains are chronically vulnerable to port congestion, supplier component stockouts, geopolitical delays, and logistic capacity stockouts. When a critical shipment (e.g., semiconductor microcontrollers) is delayed:
- Operating teams face high-stress, conflicting recovery options: charter expedited air freight, contract expensive backup suppliers, or delay customer deliveries.
- Without linear programming, recovery choices are made ad-hoc, leading to budget blowouts or stockouts.
- Crucially, when actual execution costs differ from initial quotes (e.g. air cargo spot surcharges), organizations fail to capture this variance—repeating the same faulty assumptions in future decisions.

---

## 3. Proposed Solution

Supply Prescript provides an end-to-end prescriptive decision control tower:
1. **Predict:** Supervised machine learning identifies disruption probability before shipments arrive.
2. **Assess Risk:** Multidimensional risk scoring analyzes supplier reliability, warehouse coverage, and product criticality.
3. **Optimize:** Mixed-Integer Linear Program (MILP) calculates cost-optimal fulfillment across primary, secondary, and air freight channels.
4. **Recommend:** Multi-criteria ranking generates explainable business alternatives (`ALT_A`, `ALT_B`, `ALT_C`).
5. **Decide:** Human-in-the-loop managerial interface captures selection, approval, or override justification.
6. **Execute:** Operational lifecycle progression (`NOT_STARTED` $\rightarrow$ `IN_PROGRESS` $\rightarrow$ `COMPLETED`).
7. **Measure:** Post-execution variance calculation between estimated and actual performance.
8. **Learn:** Controlled Exponentially Weighted Moving Average (EWMA) updates future optimization assumptions.

---

## 4. Main Features

- **MILP Optimization Engine:** Formulated with PuLP and solved via COIN-OR CBC for global optimality.
- **Explainable Recommendations:** Automatic generation of advantages, disadvantages, and trade-offs.
- **Human-in-the-Loop Governance:** Mandatory audit trail and reasoning for overrides.
- **Outcome Variance Tracking:** Automated directional favorability detection for cost, lead time, and service level.
- **Closed-Loop Feedback Safeguards:** Minimum sample size verification, $\pm 30\%$ multiplier clamping, and statistical outlier filtering.
- **9-View Control Tower Dashboard:** Interactive Streamlit enterprise UI connected to FastAPI and relational database.
- **Complete REST API:** Documented OpenAPI / Swagger specification at `/docs`.

---

## 5. System Architecture

```text
[ Operational Ingestion ] ──► [ Predictive Engine (ML) ] ──► [ Multidimensional Risk Engine ]
                                                                       │
[ Future Optimization Assumptions ] ◄── [ Feedback Engine ]            ▼
            ▲                                      ▲       [ PuLP Prescriptive Solver ]
            │                                      │                   │
[ Outcome Variance Engine ] ◄── [ Execution Tracking ] ◄── [ Manager Decision Interface ]
```

---

## 6. Technology Stack

- **Prescriptive Optimization:** Python 3.10+, PuLP, COIN-OR CBC Solver.
- **Machine Learning & Analytics:** scikit-learn, XGBoost, Pandas, NumPy, SciPy.
- **Backend REST API:** FastAPI, Pydantic V2, Uvicorn, HTTPX.
- **Web Dashboard:** Streamlit, Plotly Express, Altair.
- **Relational Database:** PostgreSQL 16 / SQLite, SQLAlchemy 2.0 ORM, Alembic.
- **Testing & Quality:** Pytest 9.x (100 passing unit and integration tests).

---

## 7. Project Structure

```text
SupplyPrescript/
├── app.py                          # Enterprise Streamlit Control Tower Dashboard
├── config/
│   ├── features.yaml               # Feature engineering definitions
│   ├── risk_config.yaml            # Operational risk thresholds & weights
│   └── optimization_config.yaml    # Solver parameters, cost models & learning safeguards
├── data/
│   ├── raw/                        # 1,000 raw supply event records
│   └── processed/                  # Preprocessed, leakage-free dataset
├── database/
│   ├── schema.sql                  # PostgreSQL relational DDL
│   ├── seed.py                     # Database seeder with SQLite fallback
│   └── seed.sql                    # Initial master data
├── docs/
│   ├── system_architecture.md      # Subsystems & interaction diagrams
│   ├── api_documentation.md        # Complete REST API specifications
│   ├── optimization_methodology.md # MILP mathematical formulation
│   └── closed_loop_workflow.md     # 8-stage decision lifecycle guide
├── migrations/                     # Alembic schema migrations (001, 002)
├── models/
│   └── delay_prediction/           # Saved joblib classifiers & metadata
├── reports/
│   ├── final_implementation_audit.md
│   ├── final_testing_report.md
│   ├── optimization/               # Day 10 execution summary
│   ├── recommendations/            # Day 11 recommendation matrix
│   ├── outcomes/                   # Day 14 variance summary
│   └── feedback/                   # Day 15 prescriptive learning summary
├── src/
│   ├── api/                        # FastAPI routers and dependency injection
│   ├── data/                       # Data engineering & feature transformers
│   ├── database/                   # SQLAlchemy models, CRUD operations & base
│   ├── decision_management/        # Decision approval, overrides & audit logging
│   ├── feedback_engine/            # EWMA multiplier calibration & safeguards
│   ├── models/delay_prediction/    # Supervised classification & inference
│   ├── optimization/               # PuLP MILP solver, feasibility & cost calculator
│   ├── outcome_engine/             # Post-execution actuals & variance calculations
│   ├── recommendation_engine/      # Multi-criteria ranking & business explanations
│   ├── risk_engine/                # Multidimensional operational risk scoring
│   └── workflows/                  # Complete end-to-end orchestration
├── tests/                          # 100 passing unit & integration tests
├── Dockerfile                      # Production container configuration
├── docker-compose.yml              # Multi-container orchestration (DB, API, UI)
├── requirements.txt                # Pinned production dependencies
└── .env.example                    # Environment variable template
```

---

## 8. Database Design

11 relational tables linked with cascading foreign keys and check constraints:
- `products`: SKU catalog, unit costs, categories, criticality tiers.
- `suppliers`: Reliability scores, capacity units, country, lead times.
- `inventory`: Warehouse stock snapshots, available units, reserved units.
- `supply_events`: Operational replenishment orders and shipment records.
- `predictions`: Machine learning inference probability and predicted delay.
- `risk_assessments`: 5-dimensional operational risk scores and reason codes.
- `optimization_runs`: Solver runs, objective weights, and baseline costs.
- `recommendations`: Actionable strategies (`AIR_FREIGHT`, `SECONDARY_SUPPLIER`, etc.).
- `decisions`: Manager selections, approval status, override justification.
- `outcomes`: Post-execution actuals, cost variances, and SLA performance.
- `feedback`: Error signals, weight adjustments, and versioned learning signals.

---

## 9. Core Demonstration Walkthrough (Microchip X1 — EVT-001)

1. **Supply Disruption:** Microchip X1 shipment `EVT-001` experiences a predicted 14-day delay ($P(\text{delay}) = 0.87$).
2. **Risk Classification:** Evaluated as `CRITICAL` risk (Composite Score: 0.82) due to high delay probability and zero warehouse buffer.
3. **Optimization:** PuLP solver evaluates 3 alternatives:
   - **Alternative A (Air Freight):** \$15,000.00 | 2.0 days delivery | Risk: 0.08
   - **Alternative B (Secondary Supplier):** \$138,050.00 | 7.0 days delivery | Risk: 0.15
   - **Alternative C (Delay Launch):** \$21,800.00 | 14.0 days delivery | Risk: 0.35
4. **Recommendation:** Alternative A is ranked #1 (`is_recommended = true`).
5. **Manager Decision:** Authorized Operations VP approves Air Freight.
6. **Execution:** Status advances: `NOT_STARTED` $\rightarrow$ `IN_PROGRESS` $\rightarrow$ `COMPLETED`.
7. **Outcome Evaluation:** Invoiced cost arrives at \$18,000.00:
   - Cost Variance: **+$3,000.00 (+20.0% Unfavorable)**
   - Delay Variance: **0.0 days (On-Time / Favorable)**
8. **Prescriptive Learning:** Feedback Engine validates safeguards and recalibrates the air freight multiplier via EWMA:
   $$\mu_{t} = 0.20 \times 1.20 + 0.80 \times 1.00 = 1.04$$
   Once evidence threshold ($N \ge 3$) is reached, future optimization runs calculate air freight cost as:
   $$\text{Adjusted Cost} = \$15,000 \times 1.10 = \mathbf{\$16,500.00}$$

---

## 10. Quickstart & Installation

### Option A: Local Development Setup (Quickest)

```bash
# 1. Clone repository
git clone https://github.com/organization/supply-prescript.git
cd supply-prescript

# 2. Activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt httpx

# 4. Initialize environment configuration
cp .env.example .env

# 5. Seed SQLite demonstration database
python -m database.seed

# 6. Run test suite (100/100 tests pass)
pytest

# 7. Start FastAPI Backend (Terminal 1)
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# 8. Start Streamlit Control Tower Dashboard (Terminal 2)
streamlit run app.py
```

### Option B: Docker Container Deployment

```bash
# Start PostgreSQL, FastAPI Backend, and Streamlit Dashboard
docker-compose up --build -d

# View running containers
docker-compose ps

# Access Services:
# - Streamlit Control Tower: http://localhost:8501
# - FastAPI Documentation:   http://localhost:8000/docs
# - PostgreSQL Database:     localhost:5432
```

---

## 11. Testing and Verification

Run the complete 100-test test suite:
```bash
pytest --verbose
```
Expected output:
```text
===================== 100 passed, 41 warnings in 24.74s ======================
```

---

## 12. Limitations & Future Roadmap

- **Multi-Echelon Network Expansion:** Scaling solver formulations to multi-tier supply networks with intermediate transfer hubs.
- **Seasonal Surcharge Modeling:** Integrating exogenous air cargo index data to predict rate surges during peak seasons.
- **Automated ERP Webhooks:** Directly subscribing to SAP / Oracle SCM webhook event streams.

