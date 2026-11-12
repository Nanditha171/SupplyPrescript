"""Business reasoning, advantages, disadvantages, and explanation generator (Day 11)."""

from typing import Tuple, List
from src.recommendation_engine.schemas import RecommendationOption


def generate_strategy_explanation(
    strategy_type: str,
    estimated_cost: float,
    estimated_delay_days: float,
    estimated_risk: float,
    service_level: float,
    product_name: str = "critical component",
) -> Tuple[str, List[str], List[str], str]:
    """Generate business-oriented description, advantages, disadvantages, and decision reasoning.

    Returns:
        Tuple of (strategy_description, advantages, disadvantages, reasoning)
    """
    st = strategy_type.upper()

    if st == "AIR_FREIGHT":
        desc = (
            f"Expedited international air cargo charter to bypass port disruptions and deliver {product_name} within 48 hours."
        )
        adv = [
            "Fastest feasible recovery turnaround (2.0 days).",
            "Eliminates primary shipment delay and preserves product delivery schedule.",
            "Minimal service-level risk (estimated risk: 0.08).",
            "Protects critical assembly timeline and downstream commitments.",
        ]
        disadv = [
            "Significant expedited logistics freight expenditure.",
            "Subject to airline freight capacity and spot-market surcharges.",
        ]
        reason = (
            f"Air freight provides the fastest feasible recovery option ({estimated_delay_days:.1f} days) "
            f"and reduces operational delay risk to {estimated_risk:.2f}, securing a {service_level * 100:.1f}% service level."
        )

    elif st == "SECONDARY_SUPPLIER":
        desc = (
            f"Emergency spot purchase order allocated to verified qualified secondary supplier to manufacture {product_name}."
        )
        adv = [
            "Leverages established secondary supplier manufacturing capacity.",
            "Provides reliable 7-day lead time without relying on disrupted primary logistics.",
            "Moderate risk profile (estimated risk: 0.15).",
        ]
        disadv = [
            "Requires payment of secondary supplier premium (typically 10-15%).",
            "Delivery lead time is slower than air cargo (7.0 days vs 2.0 days).",
        ]
        reason = (
            f"Secondary supplier provides reliable redundant capacity within {estimated_delay_days:.1f} days at "
            f"${estimated_cost:,.2f}, avoiding full air expedite surcharges while mitigating primary supplier failure."
        )

    elif st == "DELAY_LAUNCH":
        desc = (
            f"Formal postponement of product assembly and launch schedule by {estimated_delay_days:.0f} days to await delayed primary shipment."
        )
        adv = [
            "Avoids immediate cash outlay on expedited freight or secondary supplier premiums.",
            "Allows primary shipment to arrive naturally via original ocean/land transit.",
        ]
        disadv = [
            f"Incurs operational postponement penalty (${estimated_cost:,.2f}) and marketing disruption.",
            "Carries high service risk and damages customer delivery commitments.",
        ]
        reason = (
            f"Delaying launch saves expedited transportation premiums but introduces {estimated_delay_days:.0f} days "
            f"of schedule slip with an estimated commercial impact of ${estimated_cost:,.2f}."
        )

    elif st == "PRIMARY_SUPPLIER":
        desc = (
            f"Maintain standard execution with primary supplier, absorbing predicted operational delays."
        )
        adv = [
            "Lowest direct procurement expenditure (standard contract rates).",
            "No setup fees or expedited cargo charges.",
        ]
        disadv = [
            f"Accepts predicted delay of {estimated_delay_days:.1f} days without mitigation.",
            "High vulnerability to severe stockouts and assembly line starvation.",
        ]
        reason = (
            f"Continuing with primary supplier minimizes direct procurement cost but leaves the organization exposed to "
            f"a predicted {estimated_delay_days:.1f}-day delay with elevated operational risk ({estimated_risk:.2f})."
        )

    else:  # PARTIAL_ALLOCATION
        desc = (
            f"Hybrid multi-sourcing allocation splitting demand between secondary supplier and expedited air freight."
        )
        adv = [
            "Balances rapid inventory replenishment with cost containment.",
            "Maintains high service level while capping total logistics premium.",
        ]
        disadv = [
            "Increased administrative complexity managing dual supply channels.",
        ]
        reason = (
            f"Split allocation provides a balanced compromise between lead time ({estimated_delay_days:.1f} days) "
            f"and total expenditure (${estimated_cost:,.2f})."
        )

    return desc, adv, disadv, reason

