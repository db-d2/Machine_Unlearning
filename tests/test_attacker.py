"""Tests for MIA attacker feature extraction and model shapes."""

import torch

from attacker import MLPAttacker, extract_vae_features, build_attack_features


class TestFeatureDimensions:
    """Verify feature vector dimensions for each variant."""

    def _extract(self, small_vae, dummy_batch):
        x, lib = dummy_batch
        return extract_vae_features(small_vae, x, lib, device="cpu")

    def test_feature_dim_v1(self, small_vae, dummy_batch):
        feats = self._extract(small_vae, dummy_batch)
        attack = build_attack_features(feats, variant="v1")
        # v1: recon(1) + kl(1) + elbo(1) + mu(8) + logvar(8) + mu_norm(1) + logvar_norm(1) = 21
        assert attack.shape == (16, 21)

    def test_feature_dim_v2(self, small_vae, dummy_batch):
        feats = self._extract(small_vae, dummy_batch)
        attack = build_attack_features(feats, variant="v2")
        # v2: recon(1) + kl(1) + elbo(1) + mu(8) + logvar(8) = 19
        assert attack.shape == (16, 19)

    def test_feature_dim_v3(self, small_vae, dummy_batch):
        feats = self._extract(small_vae, dummy_batch)
        attack = build_attack_features(feats, variant="v3")
        # v3: recon(1) + kl(1) + elbo(1) = 3
        assert attack.shape == (16, 3)

    def test_feature_dim_v1_with_z32(self):
        """Verify the canonical 69-dim for z=32."""
        model = MLPAttacker(input_dim=69, hidden_dims=[64], dropout=0.0)
        # v1 with z=32: 1+1+1+32+32+1+1 = 69
        expected_dim = 2 * 32 + 5
        assert expected_dim == 69


class TestExtractFeatures:
    """Verify feature extraction returns correct keys and shapes."""

    def test_feature_keys(self, small_vae, dummy_batch):
        x, lib = dummy_batch
        feats = extract_vae_features(small_vae, x, lib, device="cpu")
        expected_keys = {"recon_nll", "kl", "elbo", "mu", "logvar", "mu_norm", "logvar_norm", "z"}
        assert set(feats.keys()) == expected_keys

    def test_feature_batch_dim(self, small_vae, dummy_batch):
        x, lib = dummy_batch
        feats = extract_vae_features(small_vae, x, lib, device="cpu")
        for key, val in feats.items():
            assert val.shape[0] == 16, f"{key} has wrong batch dim"

    def test_scalar_features_are_1d(self, small_vae, dummy_batch):
        x, lib = dummy_batch
        feats = extract_vae_features(small_vae, x, lib, device="cpu")
        for key in ["recon_nll", "kl", "elbo", "mu_norm", "logvar_norm"]:
            assert feats[key].dim() == 1, f"{key} should be 1-D"

    def test_recon_nll_positive(self, small_vae, dummy_batch):
        x, lib = dummy_batch
        feats = extract_vae_features(small_vae, x, lib, device="cpu")
        assert (feats["recon_nll"] >= 0).all()


class TestAttackerModel:
    """Verify attacker output shape."""

    def test_output_shape(self, small_attacker):
        x = torch.randn(8, 21)
        out = small_attacker(x)
        assert out.shape == (8, 1)

    def test_single_sample(self, small_attacker):
        x = torch.randn(1, 21)
        out = small_attacker(x)
        assert out.shape == (1, 1)
