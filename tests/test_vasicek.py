from __future__ import annotations

import unittest

import numpy as np

from src.model.vasicek import tail_default_prob


class TailDefaultProbTests(unittest.TestCase):
    def test_extreme_inputs_remain_finite_and_bounded(self) -> None:
        almost_one = np.nextafter(1.0, 0.0)
        pd_vec = np.array([1e-20, 1e-12, 1e-6, 1e-2, almost_one])
        rho_vec = np.array([0.0, 1e-15, 0.2, 0.999999999, 0.5])

        out = tail_default_prob(pd_vec, rho_vec, almost_one)

        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all(out > 0.0))
        self.assertTrue(np.all(out < 1.0))

    def test_tail_probability_is_monotone_in_pd(self) -> None:
        pd_vec = np.array([1e-4, 5e-4, 1e-3, 1e-2, 5e-2])
        rho_vec = np.full_like(pd_vec, 0.2)

        out = tail_default_prob(pd_vec, rho_vec, 0.999)

        self.assertTrue(np.all(np.diff(out) > 0.0))

    def test_tail_probability_is_monotone_in_q(self) -> None:
        pd_vec = np.array([0.01, 0.01, 0.01])
        rho_vec = np.array([0.2, 0.2, 0.2])

        out_low = tail_default_prob(pd_vec, rho_vec, 0.99)[0]
        out_mid = tail_default_prob(pd_vec, rho_vec, 0.995)[0]
        out_high = tail_default_prob(pd_vec, rho_vec, 0.999)[0]

        self.assertLess(out_low, out_mid)
        self.assertLess(out_mid, out_high)


if __name__ == "__main__":
    unittest.main()
