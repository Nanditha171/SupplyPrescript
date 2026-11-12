# Supply Prescript — Closed-Loop Feedback & Learning Summary (Day 15)
**Date:** November 7, 2026  
**Module:** `src/feedback_engine/` (Prescriptive Learning & Parameter Recalibration)  
**Mechanism:** Exponentially Weighted Moving Average (EWMA) with Bounded Safeguards  

---

## 1. Closed-Loop Prescriptive Learning Architecture

Supply Prescript completes the full operational cycle by feeding actual post-execution performance back into future mathematical optimization models:

$$\text{Predict} \longrightarrow \text{Assess Risk} \longrightarrow \text{Optimize} \longrightarrow \text{Recommend} \longrightarrow \text{Decide} \longrightarrow \text{Execute} \longrightarrow \text{Measure} \longrightarrow \mathbf{\text{Learn}}$$

When an executed strategy experiences operational or financial variance (such as higher-than-expected air freight spot charges), the Feedback Engine calculates the error signal, validates it against statistical safeguards, and recalibrates the strategy's cost multiplier:

$$\mu_t = \alpha \cdot \left(\frac{\text{Actual Cost}}{\text{Estimated Cost}}\right) + (1 - \alpha) \cdot \mu_{t-1}$$

For future optimization runs:
$$\text{Adjusted Cost Estimate} = \text{Base Cost Estimate} \times \mu_t$$

---

## 2. Learning Safeguards & Governance

To ensure operational stability and prevent volatile oscillations, strict safeguards govern parameter calibration:

1. **Minimum Sample Size Requirement:** A minimum of 3 validated historical observations is required before updating a strategy's operational cost multiplier. Single isolated decisions do not trigger premature parameter changes.
2. **Adjustment Boundary Clamping:** Multipliers are strictly bounded between $[0.70, 1.30]$ ($\pm 30\%$ maximum deviation from baseline).
3. **Statistical Outlier Rejection:** Cost ratios $< 0.20$ or $> 3.00$ are flagged as data entry errors or anomalous black-swan disruptions and excluded from parameter recalibration.
4. **Idempotency Protection:** Prevents duplicate feedback signal generation for already processed outcome records.
5. **Model Retraining Isolation:** Machine learning inference models (Day 6) are not dynamically retrained on single decision outcomes; retraining is decoupled into a dedicated validation cycle.

---

## 3. Core Demonstration Scenario Calibration (Microchip X1 — EVT-001)

- **Strategy Evaluated:** Air Freight Expedited Charter (`ALT_A`)
- **Estimated Benchmark Cost:** $15,000.00
- **Actual Post-Execution Cost:** $18,000.00
- **Cost Error:** +$3,000.00 (Actual-to-Estimated Ratio: 1.20)
- **Observation 1 Behavior:**
  - With default safeguard ($N_{min} = 3$), observation count = 1.
  - Multiplier held at baseline: **1.00**
  - Learning Signal: `"AIR_FREIGHT: 1.00; Observation 1/3 recorded. Retaining baseline estimate (1.00) until minimum sample threshold (3) is satisfied."`
- **Trained Calibration Behavior (Threshold Met):**
  - Once sample count reaches threshold (e.g. 3 consecutive +20% variances):
  - Calibrated multiplier: **1.10** ($\alpha = 0.20$)
  - Future Air Freight estimate: $\$15,000 \times 1.10 = \mathbf{\$16,500.00}$
  - Subsequent solver runs incorporate the updated \$16,500 estimate, optimizing future recovery decisions with real-world accuracy.

---

## 4. Verification & Validation

The test suite in `tests/feedback_engine/test_feedback_engine.py` validates:
1. Strict multiplier clamping within $[0.70, 1.30]$.
2. Statistical outlier rejection.
3. EWMA calculation accuracy.
4. Retention of baseline multiplier when sample count $< 3$.
5. Controlled activation of updated multipliers when evidence requirements are satisfied.

