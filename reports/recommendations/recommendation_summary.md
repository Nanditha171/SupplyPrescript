# Supply Prescript — Recommendation Engine Summary (Day 11)
**Date:** November 7, 2026  
**Module:** `src/recommendation_engine/` (Prescriptive Alternatives & Business Ranking)  
**Ranking Modes:** `MIN_COST`, `MIN_RISK`, `MIN_DELAY`, `BALANCED`  

---

## 1. Objective & Architecture

The Recommendation Engine translates raw mathematical solutions from the Day 10 PuLP Optimization Solver into transparent, ranked, business-actionable recovery options. It bridges operational research with executive decision-making, providing managers with clear trade-offs, advantages, disadvantages, and auditable reasoning.

### Key Capabilities
- **Multi-Alternative Generation:** Solves and filters recovery options (`AIR_FREIGHT`, `SECONDARY_SUPPLIER`, `DELAY_LAUNCH`, `PRIMARY_SUPPLIER`, `PARTIAL_ALLOCATION`).
- **Normalized Multi-Criteria Ranking:** Supports `MIN_COST`, `MIN_RISK`, `MIN_DELAY`, and `BALANCED`. In `BALANCED` mode, cost, risk, and delivery days are normalized against reference boundaries, avoiding unweighted additions of dollars, days, and risk probabilities.
- **Explainability:** Generates human-readable descriptions, advantages, disadvantages, and rationale for every evaluated alternative.
- **Database Persistence:** Stores each alternative in the `recommendations` table with a foreign key to its corresponding `optimization_run_id`.

---

## 2. Core Demonstration Scenario Comparison (Microchip X1 — EVT-001)

| Recommendation Code | Strategy Type | Estimated Cost | Expected Delivery | Operational Risk | Service Level | Advantages | Disadvantages | Rank (`MIN_COST`) |
|---|---|---|---|---|---|---|---|---|
| **ALT_A** | Air Freight Expedited | **$15,000.00** | **2.0 days** | **0.08** | 100.0% | Fastest delivery; protects launch schedule; low delay risk. | Higher expediting freight cost. | **1 (Recommended)** |
| **ALT_C** | Delay Product Launch | $21,800.00 | 14.0 days | 0.35 | 95.0% | Avoids air shipping premium; uses standard transit. | Postponement commercial penalty; customer slip. | 2 |
| **ALT_B** | Secondary Supplier | $138,050.00 | 7.0 days | 0.15 | 100.0% | Independent supplier capacity; reliable 7-day turnaround. | 10% unit premium on high volume. | 3 |

---

## 3. Explanation Logic Example

```text
[RECOMMENDATION] ALT_A — AIR_FREIGHT
Estimated Cost: $15,000.00
Estimated Delivery: 2.0 days
Estimated Risk: 0.08
Service Level: 100.0%
Rank: 1 (RECOMMENDED)

Reasoning:
Air freight provides the fastest feasible recovery option (2.0 days)
and reduces operational delay risk to 0.08, securing a 100.0% service level.
```

---

## 4. Verification & Validation

The test suite in `tests/recommendation_engine/test_recommendation_engine.py` validates:
1. `MIN_COST` ranking ordering.
2. `MIN_RISK` ranking ordering.
3. `MIN_DELAY` turnaround ranking ordering.
4. `BALANCED` normalized multi-attribute scoring.
5. Automated generation of business advantages, disadvantages, and reasoning.

