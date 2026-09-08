"""Tests for Fisher information utilities."""

import torch
import numpy as np

from fisher_utils import cosine_sim, effective_rank, top_k_overlap, vae_layer_category


class TestCosineSim:
    def test_parallel_vectors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert abs(cosine_sim(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert abs(cosine_sim(a, b)) < 1e-6

    def test_antiparallel(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([-1.0, 0.0])
        assert abs(cosine_sim(a, b) - (-1.0)) < 1e-6

    def test_scaled_vectors(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = a * 100.0
        assert abs(cosine_sim(a, b) - 1.0) < 1e-6


class TestEffectiveRank:
    def test_uniform_distribution(self):
        # Uniform over n elements: effective rank = n
        f = torch.ones(10)
        rank = effective_rank(f)
        assert abs(rank - 10.0) < 0.1

    def test_point_mass(self):
        # All weight on one element: effective rank = 1
        f = torch.zeros(10)
        f[0] = 1.0
        rank = effective_rank(f)
        assert abs(rank - 1.0) < 0.1

    def test_rank_between_1_and_n(self):
        f = torch.tensor([10.0, 1.0, 1.0, 1.0])
        rank = effective_rank(f)
        assert 1.0 < rank < 4.0


class TestTopKOverlap:
    def test_identical_vectors(self):
        a = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        assert top_k_overlap(a, a, k=3) == 1.0

    def test_reversed_vectors(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        b = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        # top-2 of a: indices {4,3}, top-2 of b: indices {0,1}
        assert top_k_overlap(a, b, k=2) == 0.0

    def test_partial_overlap(self):
        a = torch.tensor([10.0, 5.0, 1.0, 1.0])
        b = torch.tensor([10.0, 1.0, 5.0, 1.0])
        # top-2 of a: {0,1}, top-2 of b: {0,2}, overlap = {0} = 1/2
        assert top_k_overlap(a, b, k=2) == 0.5


class TestLayerCategories:
    def test_encoder_layer(self):
        assert vae_layer_category("encoder.network.0.weight") == "Encoder"

    def test_bottleneck_mu(self):
        assert vae_layer_category("encoder.fc_mu.weight") == "Bottleneck"

    def test_bottleneck_logvar(self):
        assert vae_layer_category("encoder.fc_logvar.weight") == "Bottleneck"

    def test_decoder_hidden(self):
        assert vae_layer_category("decoder.network.4.weight") == "Decoder hidden"

    def test_decoder_output_mean(self):
        assert vae_layer_category("decoder.fc_mean.0.weight") == "Decoder output"

    def test_decoder_output_dispersion(self):
        assert vae_layer_category("decoder.fc_dispersion.weight") == "Decoder output"


class TestFisherDiagonalProperties:
    """Test Fisher diagonal has expected mathematical properties."""

    def test_fisher_diagonal_nonnegative(self, small_vae, dummy_batch):
        """Fisher diagonal entries are squared gradients, must be >= 0."""
        from torch.utils.data import TensorDataset, DataLoader
        from fisher_utils import compute_vae_fisher

        x, lib = dummy_batch
        dataset = TensorDataset(x, lib)
        loader = DataLoader(dataset, batch_size=8)
        fisher_dict, n_samples = compute_vae_fisher(small_vae, loader, "cpu", damping=0.0)
        for name, f in fisher_dict.items():
            assert (f >= 0).all(), f"{name} has negative Fisher entries"

    def test_damping_adds_floor(self, small_vae, dummy_batch):
        """All Fisher entries should be >= damping value."""
        from torch.utils.data import TensorDataset, DataLoader
        from fisher_utils import compute_vae_fisher

        x, lib = dummy_batch
        dataset = TensorDataset(x, lib)
        loader = DataLoader(dataset, batch_size=16)
        damping = 0.5
        fisher_dict, _ = compute_vae_fisher(small_vae, loader, "cpu", damping=damping)
        for name, f in fisher_dict.items():
            assert (f >= damping - 1e-6).all(), \
                f"{name} has entries below damping floor {damping}"
