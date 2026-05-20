"""
Tests for GNMax RDP accountant and calibrate_sigma().

Run: pytest tests/test_accountant.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest

from src.accountant import (
    GNMaxRDPAccountant,
    calibrate_sigma,
    compute_epsilon_at_sigma,
)


class TestGNMaxRDPAccountant:
    def test_zero_queries(self):
        acc = GNMaxRDPAccountant(num_teachers=10, sigma=1.0, delta=1e-5)
        eps = acc.get_epsilon()
        assert eps == 0.0 or eps >= 0.0

    def test_epsilon_increases_with_queries(self):
        acc = GNMaxRDPAccountant(num_teachers=10, sigma=1.0, delta=1e-5)
        acc.step(100)
        eps_100 = acc.get_epsilon()
        acc.step(100)
        eps_200 = acc.get_epsilon()
        assert eps_200 > eps_100

    def test_larger_sigma_gives_smaller_epsilon(self):
        acc_small = GNMaxRDPAccountant(num_teachers=10, sigma=0.5, delta=1e-5)
        acc_large = GNMaxRDPAccountant(num_teachers=10, sigma=5.0, delta=1e-5)
        acc_small.step(1000)
        acc_large.step(1000)
        assert acc_large.get_epsilon() < acc_small.get_epsilon()

    def test_step_accumulates(self):
        acc = GNMaxRDPAccountant(num_teachers=10, sigma=1.0, delta=1e-5)
        acc.step(50)
        acc.step(50)
        assert acc.steps == 100

    def test_reset(self):
        acc = GNMaxRDPAccountant(num_teachers=10, sigma=1.0, delta=1e-5)
        acc.step(500)
        acc.reset()
        assert acc.steps == 0


class TestCalibrateσ:
    def test_returned_sigma_satisfies_epsilon(self):
        sigma = calibrate_sigma(
            target_epsilon=10.0,
            num_queries=1000,
            num_teachers=10,
            delta=1e-5,
        )
        achieved_eps = compute_epsilon_at_sigma(sigma, 1000, 10, delta=1e-5)
        assert achieved_eps <= 10.0 + 1e-3  # small numerical tolerance

    def test_tighter_epsilon_yields_larger_sigma(self):
        sigma_loose = calibrate_sigma(10.0, 1000, 10, 1e-5)
        sigma_tight = calibrate_sigma(5.0, 1000, 10, 1e-5)
        assert sigma_tight > sigma_loose

    def test_more_queries_yields_larger_sigma(self):
        sigma_few = calibrate_sigma(10.0, 500, 10, 1e-5)
        sigma_many = calibrate_sigma(10.0, 2000, 10, 1e-5)
        assert sigma_many > sigma_few

    def test_infeasible_raises(self):
        with pytest.raises(ValueError):
            calibrate_sigma(
                target_epsilon=0.001,   # extremely tight
                num_queries=100000,
                num_teachers=10,
                delta=1e-5,
                sigma_hi=1e4,
            )
