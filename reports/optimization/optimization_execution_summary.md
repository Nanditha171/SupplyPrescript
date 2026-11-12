# Supply Prescript — Optimization Engine Execution Summary (Day 10)
**Date:** November 7, 2026  
**Module:** `src/optimization/` (Prescriptive Optimization & Linear Programming)  
**Solver:** PuLP with COIN-OR CBC (Mixed-Integer Linear Programming)  

---

## 1. Mathematical Formulation & Architecture

The Supply Prescript Optimization Engine models operational supply-chain recovery as a Mixed-Integer Linear Program (MILP). When a disruption or severe delay is predicted for a critical component shipment (e.g. Microchip X1 on event `EVT-001`), the engine determines the minimum-cost, risk-mitigated recovery allocation.

### Decision Variables
- $x_{prim} \in \mathbb{Z}_{\ge 0}$: Quantity allocated to primary supplier.
- $x_{sec} \in \mathbb{Z}_{\ge 0}$: Quantity allocated to secondary / backup supplier.
- $x_{air} \in \mathbb{Z}_{\ge 0}$: Quantity allocated to expedited air freight logistics.
- $s_{unmet} \in \mathbb{Z}_{\ge 0}$: Unmet demand units incurring shortage penalty (if allowed).
- $y_{air} \in \{0, 1\}$: Binary activation flag for air freight expedition.
- $y_{sec} \in \{0, 1\}$: Binary activation flag for secondary supplier contracting.
- $y_{delay} \in \{0, 1\}$: Binary activation flag for product launch postponement.

### Objective Functions
The objective function minimizes total operational cost or a normalized multi-criteria objective:

$$\min Z = C_{proc}(x_{prim}, x_{sec}) + C_{trans}(x_{prim}) + C_{air}(x_{air}, y_{air}) + C_{shortage}(s_{unmet}) + C_{delay}(y_{delay})$$

Where:
- $C_{proc} = c_{unit} \cdot x_{prim} + c_{unit} \cdot (1 + \rho_{sec}) \cdot M_{sec} \cdot x_{sec}$
- $C_{air} = (\text{FixedAirCost} \cdot y_{air} + c_{air, unit} \cdot x_{air}) \cdot M_{air}$
- $C_{shortage} = P_{shortage} \cdot s_{unmet}$
- $C_{delay} = (\text{FixedImpact} + \text{DailyPenalty} \cdot T_{delay}) \cdot y_{delay}$
- $M_{sec}, M_{air}$: Learned cost multipliers calibrated through the Closed-Loop Feedback Engine.

### Constraints Implemented
1. **Demand Fulfillment**:
   $$I_{avail} + x_{prim} + x_{sec} + x_{air} + s_{unmet} \ge D_{demand}$$
   Ensuring available inventory $I_{avail}$ is non-negative and counted strictly once without double counting.
2. **Primary Supplier Capacity**:
   $$x_{prim} \le \text{Cap}_{prim}$$
3. **Secondary Supplier Capacity Activation**:
   $$x_{sec} \le \text{Cap}_{sec} \cdot y_{sec}$$
4. **Air Freight Capacity Activation**:
   $$x_{air} \le \text{Cap}_{air} \cdot y_{air}$$
5. **Minimum Service Level**:
   $$I_{avail} + x_{prim} + x_{sec} + x_{air} \ge \text{ServiceLevelTarget} \cdot D_{demand}$$
6. **Non-negativity & Integrality**:
   $$x_{prim}, x_{sec}, x_{air}, s_{unmet} \in \mathbb{Z}_{\ge 0}, \quad y_{air}, y_{sec}, y_{delay} \in \{0, 1\}$$

---

## 2. Solver Statuses Supported

| Solver Status | Meaning & Policy | System Behavior |
|---|---|---|
| `OPTIMAL` | Global optimal solution found by CBC solver. | Generates and ranks recommendation alternatives. |
| `FEASIBLE` | Valid feasible integer solution found within limits. | Generates recommendations with feasibility warning. |
| `INFEASIBLE` | Constraints cannot be satisfied (e.g. no capacity, strict SLA). | Aborts recommendation; logs diagnostic violations. |
| `UNBOUNDED` | Objective value decreases infinitely. | Aborts recommendation; logs formulation error. |
| `NOT_SOLVED` | Solver could not initialize or was interrupted. | Aborts recommendation; triggers fallback diagnostic. |
| `ERROR` | Runtime error or invalid input parameters. | Gracefully caught and returned with diagnostic error dict. |

---

## 3. Core Demonstration Scenario (Microchip X1 — EVT-001)

- **Product:** Microchip X1 High-Density MCU (`P001`)
- **Primary Supplier:** Pacific Silicon Semiconductor Ltd (`SUP-001`)
- **Predicted Disruption:** 14-day delay ($P(\text{delay}) = 0.87$, Composite Risk = 0.82 `CRITICAL`)
- **Gross Demand:** 1,000 units; Available Inventory: 0 units.

### Evaluated Alternatives

| Alternative | Strategy Type | Quantity | Unit / Base Cost | Total Estimated Cost | Delivery Time | Risk Score | Feasibility |
|---|---|---|---|---|---|---|---|
| **Alternative A** | Air Freight Expedited | 1,000 | \$15,000.00 base | \$15,000.00 | 2.0 days | 0.08 | **Feasible (Optimal)** |
| **Alternative B** | Secondary Supplier | 1,000 | 10% premium (\$138.05/unit) | \$138,050.00 | 7.0 days | 0.15 | **Feasible** |
| **Alternative C** | Delay Product Launch | 0 (postponed) | \$1,200/day + \$5k fixed | \$21,800.00 | 14.0 days | 0.35 | **Feasible (High impact)** |

---

## 4. Test Verification Summary

The suite in `tests/optimization/test_solver.py` covers:
1. Operational feasibility validation with valid and invalid bounds.
2. Unconstrained cost-optimal solver execution.
3. Alternative A: Air freight forced evaluation.
4. Alternative B: Secondary supplier capacity and premium validation.
5. Alternative C: Launch delay penalty calculation.
6. Zero-demand and zero-capacity edge cases.
7. Infeasible scenario with strict shortages disabled.

