# Supply Prescript — Closed-Loop Prescriptive Workflow
**Cycle:** Predict $\rightarrow$ Assess Risk $\rightarrow$ Optimize $\rightarrow$ Recommend $\rightarrow$ Decide $\rightarrow$ Execute $\rightarrow$ Measure $\rightarrow$ Learn  

---

## 1. Step-by-Step Closed-Loop Walkthrough

```text
1. Disrupted Event Ingested (EVT-001: 1,000 Microchips, 14-day delay)
                           │
                           ▼
2. Predictive Machine Learning (P(delay) = 0.87, Duration = 14 days)
                           │
                           ▼
3. Operational Risk Engine (Composite Risk = 0.82 CRITICAL)
                           │
                           ▼
4. Prescriptive Optimization Solver (PuLP MILP Evaluates 3 Alternatives)
   - Alternative A: Air Freight ($15,000, 2 days)
   - Alternative B: Secondary Supplier ($138,050, 7 days)
   - Alternative C: Delay Launch ($21,800, 14 days)
                           │
                           ▼
5. Recommendation Engine (Ranks ALT_A #1 with Business Trade-offs)
                           │
                           ▼
6. Managerial Decision Review (VP Approves ALT_A Air Freight)
                           │
                           ▼
7. Execution Tracking (NOT_STARTED -> IN_PROGRESS -> COMPLETED)
                           │
                           ▼
8. Post-Execution Outcome Ingestion (Actual Invoiced Cost = $18,000)
   - Cost Variance: +$3,000 (+20% Unfavorable)
   - Delay Variance: 0.0 days (On-Time)
                           │
                           ▼
9. Prescriptive Learning Engine
   - Computes EWMA Multiplier: mu = alpha * Ratio + (1 - alpha) * mu_prev
   - Verifies Safeguards (Sample Count, Bounded [0.70, 1.30], Outliers)
   - Recalibrates Air Freight Cost Multiplier to 1.10
                           │
                           ▼
10. Future Optimization Runs use Updated Multiplier ($15,000 * 1.10 = $16,500)
```

---

## 2. Prescriptive Learning Safeguards

1. **Minimum Sample Size Requirement ($N \ge 3$):** An isolated decision outcome records the variance but does not prematurely update the operational cost multiplier until at least 3 validated observations are logged.
2. **Boundary Clamping ($\pm 30\%$):** Multipliers are strictly clamped within $[0.70, 1.30]$ to prevent runaway inflation or ungrounded deflation.
3. **Outlier Filtering ($0.20 \le \text{Ratio} \le 3.00$):** Ratios outside this range are flagged as data errors and ignored.
4. **Idempotency:** Unique keys prevent duplicate processing of the same outcome.
5. **Model Retraining Isolation:** Machine learning inference weights are not dynamically modified by decision feedback; ML retraining occurs in scheduled model validation cycles.

