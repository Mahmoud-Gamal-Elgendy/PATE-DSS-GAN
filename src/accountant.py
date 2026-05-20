"""
GNMax RDP Accountant for PATE-DSS-GAN.

Tracks cumulative privacy budget (ε, δ) using Rényi Differential Privacy (RDP)
composition for the Gaussian Noisy Max (GNMax) mechanism.

Ported from PATE-TabTransGAN and extended with:
  - calibrate_sigma(): binary search to derive σ from (ε, N, k, δ)
  - Image-domain compatibility (domain-agnostic accounting)
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# RDP base moments for GNMax
# ---------------------------------------------------------------------------

def _log_comb(n: int, k: int) -> float:
    """log C(n, k) via log-gamma for numerical stability."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _gnmax_rdp_moment(alpha: float, sigma: float, num_teachers: int) -> float:
    """
    Compute the RDP ε(α) for a single scalar GNMax query.

    Voting mechanism (voting.py):
      noisy_real = Σ_i votes_i + N(0, σ²)
      label = (noisy_real > k/2)

    The output is a scalar noisy count with L2 sensitivity Δ = 1
    (one teacher changes the real-vote count by exactly 1).
    The standard Gaussian mechanism RDP bound (Mironov 2017) for a
    scalar query with sensitivity Δ = 1 is:

        ε(α) = α · Δ² / (2σ²) = α / (2σ²)

    Note: num_teachers (k) affects the utility of the vote (tied votes
    at k/2 are the worst case for accuracy) but NOT the RDP ε, because
    privacy depends only on the sensitivity of the released statistic,
    not on the number of data points. Sensitivity = 1 regardless of k.

    Parameters
    ----------
    alpha : float
        RDP order (α > 1).
    sigma : float
        Std-dev of Gaussian noise added to the real-vote scalar.
    num_teachers : int
        Number of teacher discriminators (k). Retained for API consistency;
        does not affect the RDP bound for the scalar mechanism.

    Returns
    -------
    float
        RDP ε at order α for one query.
    """
    if sigma <= 0:
        return float("inf")
    # Scalar Gaussian mechanism: sensitivity Δ = 1, RDP ε(α) = α / (2σ²)
    return alpha / (2.0 * sigma ** 2)


def _rdp_to_dp(rdp_eps: float, alpha: float, delta: float) -> float:
    """
    Convert RDP (α, ε_rdp) guarantee to (ε, δ)-DP via the standard conversion:
        ε_dp = ε_rdp + log(1 - 1/α) - (log(δ) + log(1 - 1/α)) / (1 - 1/α)

    Simplified commonly-used form (Mironov 2017, Prop 3):
        ε_dp ≤ ε_rdp + log(1/δ) / (α - 1)
    """
    if alpha <= 1:
        return float("inf")
    return rdp_eps + math.log(1.0 / delta) / (alpha - 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GNMaxRDPAccountant:
    """
    Tracks cumulative RDP privacy budget for GNMax queries.

    Usage
    -----
    acc = GNMaxRDPAccountant(num_teachers=10, sigma=1.0, delta=1e-5)
    acc.step()               # record one query
    eps = acc.get_epsilon()  # current (ε, δ) guarantee
    """

    # RDP orders to evaluate. Higher orders (128, 256) give tighter
    # conversions at very small δ values (e.g. δ = 1e-7 or below).
    ORDERS: List[float] = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]

    def __init__(
        self,
        num_teachers: int,
        sigma: float,
        delta: float = 1e-5,
        orders: Optional[List[float]] = None,
    ) -> None:
        self.num_teachers = num_teachers
        self.sigma = sigma
        self.delta = delta
        self.orders = orders or self.ORDERS
        self._steps: int = 0

    def step(self, n_queries: int = 1) -> None:
        """Record n_queries PATE queries (each releases one noisy vote)."""
        self._steps += n_queries

    def get_rdp(self) -> List[Tuple[float, float]]:
        """Return [(α, ε_rdp(α))] for all tracked orders."""
        per_step = [
            _gnmax_rdp_moment(alpha, self.sigma, self.num_teachers)
            for alpha in self.orders
        ]
        return [(alpha, eps * self._steps) for alpha, eps in zip(self.orders, per_step)]

    def get_epsilon(self) -> float:
        """
        Return the current (ε, δ)-DP guarantee by optimising over RDP orders.
        """
        best = float("inf")
        for alpha, rdp_eps in self.get_rdp():
            eps_dp = _rdp_to_dp(rdp_eps, alpha, self.delta)
            if eps_dp < best:
                best = eps_dp
        return best

    @property
    def steps(self) -> int:
        return self._steps

    def reset(self) -> None:
        self._steps = 0

    def __repr__(self) -> str:
        return (
            f"GNMaxRDPAccountant(k={self.num_teachers}, σ={self.sigma:.4f}, "
            f"δ={self.delta}, queries={self._steps}, ε≈{self.get_epsilon():.4f})"
        )


# ---------------------------------------------------------------------------
# calibrate_sigma — CRITICAL PATH
# ---------------------------------------------------------------------------

def calibrate_sigma(
    target_epsilon: float,
    num_queries: int,
    num_teachers: int,
    delta: float = 1e-5,
    sigma_lo: float = 1e-3,
    sigma_hi: float = 1e4,
    tol: float = 1e-6,
    max_iter: int = 200,
    orders: Optional[List[float]] = None,
) -> float:
    """
    Binary search for the minimum σ such that N queries stay within ε budget.

    This function is the CRITICAL PATH dependency identified in the proposal.
    It derives σ from (ε, N, k, δ) rather than hard-coding per dataset.

    Parameters
    ----------
    target_epsilon : float
        Maximum allowed ε for (ε, δ)-DP.
    num_queries : int
        Total number of PATE queries (N) over the training run.
    num_teachers : int
        Number of teacher discriminators (k).
    delta : float
        DP δ parameter (default 1e-5).
    sigma_lo, sigma_hi : float
        Search interval for σ.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum binary-search iterations.
    orders : list of float, optional
        RDP orders to evaluate; defaults to GNMaxRDPAccountant.ORDERS.

    Returns
    -------
    float
        Minimum σ that satisfies the (ε, δ)-DP constraint.

    Raises
    ------
    ValueError
        If no σ in [sigma_lo, sigma_hi] achieves the target_epsilon.

    Notes
    -----
    ε decreases monotonically as σ increases, so binary search is valid.
    A larger σ means more noise → tighter privacy → lower utility.
    Start validation at ε ∈ {5, 8, 10} before testing ε ∈ {1, 3}.
    """

    def _compute_eps(sigma: float) -> float:
        acc = GNMaxRDPAccountant(
            num_teachers=num_teachers,
            sigma=sigma,
            delta=delta,
            orders=orders,
        )
        acc.step(num_queries)
        return acc.get_epsilon()

    # Sanity: check upper bound is achievable
    if _compute_eps(sigma_hi) > target_epsilon:
        raise ValueError(
            f"Even σ={sigma_hi} cannot achieve ε≤{target_epsilon} with "
            f"N={num_queries} queries, k={num_teachers} teachers, δ={delta}. "
            "Reduce num_queries or increase target_epsilon."
        )

    # Sanity: check sigma_lo actually violates (ensures a crossing exists)
    # If sigma_lo already satisfies, return it (no noise needed beyond minimum)
    if _compute_eps(sigma_lo) <= target_epsilon:
        return sigma_lo

    lo, hi = sigma_lo, sigma_hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if _compute_eps(mid) <= target_epsilon:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break

    return hi  # conservative: return value that provably satisfies constraint


def compute_epsilon_at_sigma(
    sigma: float,
    num_queries: int,
    num_teachers: int,
    delta: float = 1e-5,
) -> float:
    """Convenience wrapper to compute ε for given σ and query count."""
    acc = GNMaxRDPAccountant(num_teachers=num_teachers, sigma=sigma, delta=delta)
    acc.step(num_queries)
    return acc.get_epsilon()
