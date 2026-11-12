# Supply Prescript — Optimization Methodology
**Solver:** PuLP (Python Linear Programming) with COIN-OR CBC  
**Formulation:** Mixed-Integer Linear Program (MILP)  

---

## 1. Problem Definition

When an operational supply-chain disruption occurs (such as a 14-day delay on a critical shipment), continuing with the status quo leads to stockouts, assembly line starvation, and severe penalties.

The Optimization Engine determines the optimal combination of recovery strategies:
1. **Primary Supplier:** Standard contract rate, but delayed delivery.
2. **Secondary Supplier:** Emergency sourcing with a premium percentage ($\rho_{sec} = 10\%$).
3. **Air Freight:** Expedited charter with 2-day delivery and fixed/variable freight charges.
4. **Launch Postponement:** Shifting the product launch to absorb delay, incurring a postponement penalty.
5. **Shortage Penalty:** Absorbing unmet demand when capacity cannot meet customer orders.

---

## 2. Decision Variables

- $x_{prim} \in \mathbb{Z}_{\ge 0}$: Quantity allocated to primary supplier.
- $x_{sec} \in \mathbb{Z}_{\ge 0}$: Quantity allocated to secondary supplier.
- $x_{air} \in \mathbb{Z}_{\ge 0}$: Quantity allocated to expedited air cargo.
- $s_{unmet} \in \mathbb{Z}_{\ge 0}$: Unmet demand units incurring shortage penalty.
- $y_{air} \in \{0, 1\}$: Binary activation flag for air freight expedition.
- $y_{sec} \in \{0, 1\}$: Binary activation flag for secondary supplier contracting.
- $y_{delay} \in \{0, 1\}$: Binary activation flag for launch postponement.

---

## 3. Objective Functions

### Minimum Cost (`MIN_COST`)
$$\min Z_{cost} = c_{unit} x_{prim} + c_{unit}(1 + \rho_{sec}) M_{sec} x_{sec} + C_{air}(x_{air}, y_{air}) M_{air} + P_{shortage} s_{unmet} + C_{delay}(y_{delay})$$

Where $M_{sec}, M_{air}$ are learned multipliers from the Closed-Loop Feedback Engine.

### Minimum Risk (`MIN_RISK`)
$$\min Z_{risk} = 0.20 \cdot \left(\frac{Z_{cost}}{C_{ref}}\right) + 0.80 \cdot \left(0.08 y_{air} + 0.15 y_{sec} + 0.35 y_{delay} + 0.95 \frac{s_{unmet}}{D}\right)$$

### Minimum Delay (`MIN_DELAY`)
$$\min Z_{delay} = 0.20 \cdot \left(\frac{Z_{cost}}{C_{ref}}\right) + 0.80 \cdot \left(\frac{2.0 y_{air} + 7.0 y_{sec} + T_{delay} y_{delay}}{T_{ref}}\right)$$

### Balanced Multi-Objective (`BALANCED`)
$$\min Z_{balanced} = w_{cost} \cdot \left(\frac{Z_{cost}}{C_{ref}}\right) + w_{risk} \cdot R_{composite} + w_{time} \cdot \left(\frac{T_{lead}}{T_{ref}}\right)$$

Where weights sum to 1.0 and all terms are dimensionless ratios.

---

## 4. Operational Constraints

1. **Demand Fulfillment:**
   $$I_{avail} + x_{prim} + x_{sec} + x_{air} + s_{unmet} + D \cdot y_{delay} \ge D$$
   Inventory $I_{avail}$ is non-negative and deducted once without double counting.
2. **Primary Supplier Capacity:**
   $$x_{prim} \le \text{Cap}_{prim}$$
3. **Secondary Capacity Activation:**
   $$x_{sec} \le \text{Cap}_{sec} \cdot y_{sec}$$
4. **Air Freight Capacity Activation:**
   $$x_{air} \le \text{Cap}_{air} \cdot y_{air}$$
5. **Minimum Service Level Requirement:**
   $$I_{avail} + x_{prim} + x_{sec} + x_{air} + D \cdot y_{delay} \ge \text{ServiceLevelTarget} \cdot D$$

