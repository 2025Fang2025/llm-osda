import unittest

import numpy as np

from src.eval.envelope_numerics import (
    build_bid_grid,
    build_random_turn_map,
    compute_empirical_regret,
    compute_envelope_payment,
    piecewise_constant_integral,
)


class EnvelopeRevisionTest(unittest.TestCase):
    def test_step_integral_and_payment_recover_threshold(self):
        grid = np.array([0.1, 0.5, 1.0, 2.0])
        allocation = np.array([0.0, 0.8, 0.8, 0.8])
        self.assertAlmostEqual(
            piecewise_constant_integral(grid, allocation), 1.2
        )
        result = compute_envelope_payment(grid, allocation, true_bid=2.0)
        self.assertAlmostEqual(result["envelope_cpc"], 0.5)

    def test_monotone_step_has_zero_grid_regret(self):
        grid = np.array([0.1, 0.5, 1.0, 2.0])
        allocation = np.array([0.0, 0.8, 0.8, 0.8])
        result = compute_empirical_regret(grid, allocation, true_bid=2.0)
        self.assertAlmostEqual(result["regret"], 0.0)

    def test_grid_contains_truthful_bid_and_threshold(self):
        grid = build_bid_grid(
            1.23, theta_min=0.1, theta_max=4.1, n_grid=8,
            extra_points=[0.77],
        )
        self.assertTrue(np.any(np.isclose(grid, 1.23)))
        self.assertTrue(np.any(np.isclose(grid, 0.77)))

    def test_random_turn_map_is_reproducible_and_non_degenerate(self):
        dialogues = [
            {
                "session_id": f"session-{index}",
                "turns": [{"turn": 1}, {"turn": 2}, {"turn": 3}],
            }
            for index in range(100)
        ]
        first = build_random_turn_map(dialogues, seed=42)
        second = build_random_turn_map(dialogues, seed=42)
        self.assertEqual(first, second)
        self.assertGreater(len(set(first.values())), 1)

if __name__ == "__main__":
    unittest.main()
