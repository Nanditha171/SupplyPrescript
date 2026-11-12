# Supply Prescript — System Architecture
**Version:** 1.0.0  
**Domain:** Closed-Loop Prescriptive Supply Chain Analytics  

---

## 1. High-Level Architectural Overview

Supply Prescript bridges predictive risk modeling with operational linear programming, human managerial oversight, and automated post-execution learning.

```
       [ Upstream Operational Events / ERP Ingestion ]
                            │
                            ▼
           ┌───────────────────────────────────┐
           │ 1. Delay Prediction Engine (ML)   │
           │    (Random Forest / XGBoost)      │
           └─────────────────┬─────────────────┘
                             │ Delay Probability P(delay)
                             ▼
           ┌───────────────────────────────────┐
           │ 2. Multidimensional Risk Engine   │
           │    (Supplier, Inventory, Capacity)│
           └─────────────────┬─────────────────┘
                             │ Composite Risk & Classification
                             ▼
           ┌───────────────────────────────────┐
           │ 3. Prescriptive Optimizer (PuLP)  │
           │    (COIN-OR CBC MILP Solver)      │
           └─────────────────┬─────────────────┘
                             │ Optimal Quantities & Alternatives
                             ▼
           ┌───────────────────────────────────┐
           │ 4. Recommendation Engine          │
           │    (Ranking, Explanations, Matrix)│
           └─────────────────┬─────────────────┘
                             │ Ranked Alternatives
                             ▼
           ┌───────────────────────────────────┐
           │ 5. Managerial Decision Interface  │
           │    (Selection, Override, Approval)│
           └─────────────────┬─────────────────┘
                             │ Decision Committed
                             ▼
           ┌───────────────────────────────────┐
           │ 6. Operational Execution Tracker  │
           │    (NOT_STARTED -> COMPLETED)     │
           └─────────────────┬─────────────────┘
                             │ Execution Finished
                             ▼
           ┌───────────────────────────────────┐
           │ 7. Outcome Evaluation Engine      │
           │    (Actual vs Expected Variances) │
           └─────────────────┬─────────────────┘
                             │ Variance Signal (+$3k)
                             ▼
           ┌───────────────────────────────────┐
           │ 8. Closed-Loop Feedback Engine    │
           │    (EWMA Multipliers & Safeguards)│
           └─────────────────┬─────────────────┘
                             │
                             └──────► Updates Future Optimization Runs
```

---

## 2. Subsystem Descriptions

### Subsystem 1: Data Engineering & Master Catalog
- **Relational Storage:** PostgreSQL 16 / SQLite with SQLAlchemy 2.x ORM models.
- **Entities:** Products, Suppliers, Warehouse Inventories, Supply Events.
- **Integrity:** Strict check constraints prevent negative unit costs, invalid SLA thresholds, or negative stock quantities.

### Subsystem 2: Predictive Machine Learning
- **Algorithm:** Calibrated ensemble model trained on pre-event features without future-leakage data.
- **Output:** Calibrated delay probability $P(\text{delay}) \in [0.0, 1.0]$ and expected delay days.

### Subsystem 3: Operational Risk Engine
- **Dimensions:** Delay Probability Risk (40%), Supplier Reliability Risk (20%), Inventory Coverage Risk (20%), Capacity Utilization Risk (10%), Service Criticality Risk (10%).
- **Bands:** LOW ($<0.30$), MEDIUM ($[0.30, 0.60)$), HIGH ($[0.60, 0.80)$), CRITICAL ($\ge 0.80$).

### Subsystem 4: Prescriptive Optimization Engine
- **Formulation:** Mixed-Integer Linear Program (MILP) solved using PuLP and COIN-OR CBC.
- **Variables:** Continuous/integer supplier allocations, air cargo charter quantities, binary strategy activation flags.
- **Constraints:** Demand fulfillment, supplier capacities, inventory non-double counting, SLA target requirements, mutual exclusivity.

### Subsystem 5: Recommendation Engine
- **Objectives:** `MIN_COST`, `MIN_RISK`, `MIN_DELAY`, `BALANCED`.
- **Explainability:** Generates natural language advantages, disadvantages, and business reasoning for each alternative.

### Subsystem 6: Decision Tracking & Auditing
- **Workflow:** Manager review $\rightarrow$ Selection $\rightarrow$ Override validation $\rightarrow$ Approval $\rightarrow$ Execution tracking.
- **Audit Trail:** Append-only history preserving original recommendations, manager identity, rationale, and execution timestamps.

### Subsystem 7: Outcome Evaluation & Closed-Loop Learning
- **Empirical Variance:** Compares actual invoiced cost and observed lead time against optimization estimates.
- **Calibration:** Exponentially Weighted Moving Average (EWMA) updates future strategy cost multipliers:
  $$\mu_t = \alpha \cdot \text{Ratio}_t + (1 - \alpha) \cdot \mu_{t-1}$$
- **Safeguards:** Minimum 3 observations required, multiplier clamped to $[0.70, 1.30]$, statistical outlier filtering.

