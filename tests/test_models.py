"""
Smoke tests for model forward passes (CPU-only, no mamba-ssm required).

Run: pytest tests/test_models.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest


class TestTeacherDiscriminator:
    def test_forward_128(self):
        from src.models.teacher import TeacherCNNDiscriminator
        model = TeacherCNNDiscriminator(image_size=128)
        x = torch.randn(4, 3, 128, 128)
        logit = model(x)
        assert logit.shape == (4, 1)

    def test_vote_binary(self):
        from src.models.teacher import TeacherCNNDiscriminator
        model = TeacherCNNDiscriminator(image_size=128)
        x = torch.randn(8, 3, 128, 128)
        votes = model.vote(x)
        assert votes.shape == (8,)
        assert set(votes.tolist()).issubset({0, 1})


class TestMambaStudentDiscriminator:
    @pytest.mark.parametrize("image_size", [128, 256])
    def test_forward(self, image_size):
        from src.models.student import MambaStudentDiscriminator
        model = MambaStudentDiscriminator(
            image_size=image_size,
            base_channels=16,
            num_classes=2,
        )
        x = torch.randn(2, 3, image_size, image_size)
        logit = model(x)
        assert logit.shape == (2,)

    def test_conditional_forward(self):
        from src.models.student import MambaStudentDiscriminator
        model = MambaStudentDiscriminator(image_size=128, base_channels=16, num_classes=3)
        x = torch.randn(2, 3, 128, 128)
        c = torch.randint(0, 3, (2,))
        score_cond = model(x, c)
        score_uncond = model(x, None)
        assert score_cond.shape == (2,)
        assert score_uncond.shape == (2,)
        assert not torch.allclose(score_cond, score_uncond)

    def test_invalid_resolution(self):
        from src.models.student import MambaStudentDiscriminator
        with pytest.raises(AssertionError):
            MambaStudentDiscriminator(image_size=32)


class TestDSSGANGenerator:
    def test_forward_128(self):
        from src.models.generator import DSSGANGenerator
        gen = DSSGANGenerator(
            latent_dim=152,
            num_classes=2,
            image_size=128,
            base_channels=32,
        )
        z = gen.sample_latent(2, torch.device("cpu"))
        c = torch.randint(0, 2, (2,))
        imgs = gen(z, c)
        assert imgs.shape == (2, 3, 128, 128)
        assert imgs.min() >= -1.01 and imgs.max() <= 1.01


class TestPATEVoting:
    def test_aggregate_shape(self):
        from src.accountant import GNMaxRDPAccountant
        from src.pate.voting import PATEVoteAggregator

        acc = GNMaxRDPAccountant(num_teachers=5, sigma=1.0, delta=1e-5)
        agg = PATEVoteAggregator(sigma=1.0, accountant=acc)

        votes = torch.randint(0, 2, (5, 8))  # k=5, B=8
        labels = agg.aggregate(votes)
        assert labels.shape == (8,)
        assert set(labels.tolist()).issubset({0, 1})

    def test_accountant_charges_budget(self):
        from src.accountant import GNMaxRDPAccountant
        from src.pate.voting import PATEVoteAggregator

        acc = GNMaxRDPAccountant(num_teachers=5, sigma=1.0, delta=1e-5)
        agg = PATEVoteAggregator(sigma=1.0, accountant=acc)

        assert acc.steps == 0
        votes = torch.randint(0, 2, (5, 16))
        agg.aggregate(votes)
        assert acc.steps == 16  # B=16 queries recorded
