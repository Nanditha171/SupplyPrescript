import sys
import unittest
from pathlib import Path

# Add the project root to Python's import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization.prescriptive_solver import recommend_actions


class TestPrescriptiveSolver(unittest.TestCase):

    def test_regular_supply_selected_when_cheapest_and_feasible(self):
        result = recommend_actions(
            baseline_cost=1000,
            air_freight_cost=2000,
            secondary_supplier_cost=1500,
            baseline_delay_days=3,
            air_freight_delay_days=1,
            secondary_supplier_delay_days=2,
            budget=2500,
            max_acceptable_delay=5,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recommended_action"], "Regular supply")
        self.assertLessEqual(result["estimated_cost"], result["budget"])
        self.assertLessEqual(
            result["estimated_delay_days"],
            result["max_acceptable_delay_days"],
        )

    def test_air_freight_selected_when_regular_supply_is_too_slow(self):
        result = recommend_actions(
            baseline_cost=1000,
            air_freight_cost=1800,
            secondary_supplier_cost=1500,
            baseline_delay_days=8,
            air_freight_delay_days=2,
            secondary_supplier_delay_days=5,
            budget=2500,
            max_acceptable_delay=3,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recommended_action"], "Air freight")
        self.assertLessEqual(result["estimated_cost"], result["budget"])
        self.assertLessEqual(
            result["estimated_delay_days"],
            result["max_acceptable_delay_days"],
        )

    def test_secondary_supplier_selected_when_cheapest_feasible_option(self):
        result = recommend_actions(
            baseline_cost=1000,
            air_freight_cost=2500,
            secondary_supplier_cost=1400,
            baseline_delay_days=8,
            air_freight_delay_days=2,
            secondary_supplier_delay_days=4,
            budget=2000,
            max_acceptable_delay=5,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["recommended_action"],
            "Secondary supplier",
        )
        self.assertLessEqual(result["estimated_cost"], result["budget"])
        self.assertLessEqual(
            result["estimated_delay_days"],
            result["max_acceptable_delay_days"],
        )

    def test_no_feasible_solution_when_budget_is_too_low(self):
        result = recommend_actions(
            baseline_cost=1000,
            air_freight_cost=2000,
            secondary_supplier_cost=1500,
            baseline_delay_days=3,
            air_freight_delay_days=1,
            secondary_supplier_delay_days=2,
            budget=500,
            max_acceptable_delay=5,
        )

        self.assertEqual(result["status"], "no_feasible_solution")

    def test_no_feasible_solution_when_all_options_are_too_slow(self):
        result = recommend_actions(
            baseline_cost=1000,
            air_freight_cost=2000,
            secondary_supplier_cost=1500,
            baseline_delay_days=8,
            air_freight_delay_days=6,
            secondary_supplier_delay_days=7,
            budget=3000,
            max_acceptable_delay=4,
        )

        self.assertEqual(result["status"], "no_feasible_solution")


if __name__ == "__main__":
    unittest.main(verbosity=2)