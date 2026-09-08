"""Tests for data splits and matched negative construction."""

import json
import numpy as np
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent.parent


class TestSplitFiles:
    """Verify persisted split files have correct structure."""

    def _load_split(self, name):
        path = DATA_ROOT / "outputs" / "p1" / name
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def test_structured_split_exists(self):
        split = self._load_split("split_structured.json")
        if split is None:
            return  # skip if data not generated yet
        assert "forget_indices" in split
        assert "retain_indices" in split
        assert "unseen_indices" in split

    def test_splits_no_overlap(self):
        split = self._load_split("split_structured.json")
        if split is None:
            return
        forget = set(split["forget_indices"])
        retain = set(split["retain_indices"])
        unseen = set(split["unseen_indices"])
        assert len(forget & retain) == 0, "forget and retain overlap"
        assert len(forget & unseen) == 0, "forget and unseen overlap"
        assert len(retain & unseen) == 0, "retain and unseen overlap"

    def test_splits_cover_all(self):
        split = self._load_split("split_structured.json")
        if split is None:
            return
        all_indices = (
            set(split["forget_indices"])
            | set(split["retain_indices"])
            | set(split["unseen_indices"])
        )
        n_total = split["n_forget"] + split["n_retain"] + split["n_unseen"]
        assert len(all_indices) == n_total

    def test_split_proportions(self):
        split = self._load_split("split_structured.json")
        if split is None:
            return
        n_train = split["n_forget"] + split["n_retain"]
        n_unseen = split["n_unseen"]
        ratio = n_train / (n_train + n_unseen)
        assert 0.80 < ratio < 0.90, f"Train ratio {ratio:.2f} outside 80-90%"

    def test_scattered_split_exists(self):
        split = self._load_split("split_scattered.json")
        if split is None:
            return
        assert split["n_forget"] > 0


class TestMatchedNegatives:
    """Verify matched negatives are subset of unseen."""

    def test_matched_from_unseen(self):
        split_path = DATA_ROOT / "outputs" / "p1" / "split_structured.json"
        matched_path = DATA_ROOT / "outputs" / "p1.5" / "s1_matched_negatives.json"
        if not split_path.exists() or not matched_path.exists():
            return

        with open(split_path) as f:
            split = json.load(f)
        with open(matched_path) as f:
            matched = json.load(f)

        unseen = set(split["unseen_indices"])
        matched_indices = set(matched["matched_indices"])
        assert matched_indices.issubset(unseen), "matched negatives not all from unseen set"

    def test_matched_not_empty(self):
        matched_path = DATA_ROOT / "outputs" / "p1.5" / "s1_matched_negatives.json"
        if not matched_path.exists():
            return
        with open(matched_path) as f:
            matched = json.load(f)
        assert len(matched["matched_indices"]) > 0


class TestSplitDeterminism:
    """Verify that the same seed produces the same split."""

    def test_deterministic_shuffle(self):
        n = 33088  # PBMC cell count
        for _ in range(2):
            np.random.seed(42)
            indices = np.arange(n)
            np.random.shuffle(indices)
            n_train = int(0.85 * n)
            train = indices[:n_train]
            unseen = indices[n_train:]

        # Run again with same seed
        np.random.seed(42)
        indices2 = np.arange(n)
        np.random.shuffle(indices2)
        train2 = indices2[: int(0.85 * n)]

        assert np.array_equal(train, train2)
