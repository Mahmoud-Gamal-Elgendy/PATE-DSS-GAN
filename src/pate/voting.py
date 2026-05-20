"""
PATE Vote Aggregation with GNMax Mechanism.

Implements the privacy-correct vote aggregation for PATE-DSS-GAN:
  1. Collect binary votes (real=1 / fake=0) from k teacher discriminators.
  2. Sum votes: v_real = Σ votes  (integer in [0, k]).
  3. Add a single Gaussian noise sample N(0, σ²) to v_real.
  4. Release noisy_real > k/2 as the binary label.
  5. Record the query to the RDP accountant.

Noise model (Fix A — aligned with the accountant):
  The mechanism adds noise to a single scalar (v_real) with L2 sensitivity
  Δ = 1 (one teacher changes v_real by exactly 1). This matches the
  accountant formula ε(α) = α / (2σ²) for the scalar Gaussian mechanism.

  Previous implementation added noise to BOTH bins [v_real, v_fake]
  independently, giving effective noise variance 2σ² on the margin and
  L2 sensitivity √2 on the histogram vector. That caused the accountant
  to underestimate ε by 2× — corrected here.

Privacy correctness notes (from REFACTOR_PLAN):
  - NO confident threshold: every image is queried and recorded.
    The "threshold before noise" pattern is a privacy leak — removed.
  - Sigma σ comes from calibrate_sigma(), NOT hardcoded.
  - Each query costs ε budget; the accountant tracks the running total.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..accountant import GNMaxRDPAccountant


class PATEVoteAggregator:
    """
    GNMax vote aggregator for binary (real/fake) PATE labels.

    Parameters
    ----------
    sigma : float
        Std-dev of Gaussian noise added to the real-vote count.
        Computed by calibrate_sigma() — NOT hardcoded.
        Noise is applied to a single scalar per image (L2 sensitivity = 1),
        consistent with the RDP formula ε(α) = α / (2σ²).
    accountant : GNMaxRDPAccountant
        Shared accountant that tracks cumulative ε.
    device : torch.device
    """

    def __init__(
        self,
        sigma: float,
        accountant: GNMaxRDPAccountant,
        device: Optional[torch.device] = None,
    ) -> None:
        self.sigma = sigma
        self.accountant = accountant
        self.device = device or torch.device("cpu")

    def aggregate(self, votes: Tensor) -> Tensor:
        """
        Aggregate teacher votes with scalar Gaussian noise and return noisy labels.

        Mechanism:
          noisy_real = Σ_i votes_i + N(0, σ²)   ← single noise draw per image
          label = 1  if  noisy_real > k/2         (real wins)
                = 0  otherwise                     (fake wins)

        Sensitivity: one teacher flip changes noisy_real by exactly 1 → Δ = 1.
        This aligns with the accountant: ε(α) = α / (2σ²).

        Parameters
        ----------
        votes : Tensor  (k, B)
            Binary teacher votes: 1 = real, 0 = fake.

        Returns
        -------
        Tensor  (B,) long
            Noisy PATE labels: 1 = real, 0 = fake.

        Side Effects
        ------------
        Calls accountant.step(B) to record B queries.
        """
        k, B = votes.shape
        votes = votes.float().to(self.device)

        # Sum real votes: scalar per image, sensitivity Δ = 1
        real_votes = votes.sum(dim=0)                           # (B,)

        # Add a single Gaussian noise draw per image
        noise = torch.randn(B, device=self.device) * self.sigma  # (B,)
        noisy_real = real_votes + noise                          # (B,)

        # Decision: real wins if noisy vote count exceeds the midpoint
        noisy_labels = (noisy_real > k / 2).long()              # (B,) in {0, 1}

        # Record B queries to the privacy accountant
        self.accountant.step(B)

        return noisy_labels

    def aggregate_with_confidence(self, votes: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Aggregate votes and also return a pre-noise confidence score per image.

        Confidence = |real_votes - k/2| / (k/2)  — normalised margin in [0, 1].
        Useful for monitoring teacher ensemble quality during training.

        IMPORTANT: confidence is computed BEFORE noise is added. Do NOT use it
        to gate which images are released (that would be the privacy-leaking
        confident threshold pattern, which this codebase explicitly avoids).

        Returns
        -------
        noisy_labels : Tensor (B,)
        confidence : Tensor (B,)  in [0, 1]
        """
        k, B = votes.shape
        real_votes = votes.float().to(self.device).sum(dim=0)   # (B,)
        # Normalised margin before noise; purely for monitoring
        confidence = torch.abs(real_votes - k / 2) / (k / 2)   # (B,) in [0, 1]

        noisy_labels = self.aggregate(votes)
        return noisy_labels, confidence
