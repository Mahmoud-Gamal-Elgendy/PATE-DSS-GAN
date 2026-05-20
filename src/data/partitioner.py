"""
Stratified data partitioning for PATE teacher ensemble.

Partitions a dataset into k non-overlapping shards such that each shard
contains a balanced proportion of each class label. This ensures no teacher
sees a degenerate shard dominated by a single class.

Design note (from REFACTOR_PLAN):
  - Stratification applies only to shard creation.
  - Teachers remain real/fake discriminators, NOT class-label oracles.
  - Stratification merely ensures balanced training for each teacher.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

from torch.utils.data import Dataset

from .dataset import ImageDatasetWrapper


def stratified_partition(
    dataset: ImageDatasetWrapper,
    num_shards: int,
    seed: Optional[int] = 42,
) -> List[ImageDatasetWrapper]:
    """
    Partition dataset into `num_shards` non-overlapping stratified shards.

    Each shard receives approximately 1/num_shards of samples from each class,
    ensuring balanced class representation across teacher training sets.

    Parameters
    ----------
    dataset : ImageDatasetWrapper
        Full private dataset with `.targets` attribute.
    num_shards : int
        Number of teacher shards (k).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of ImageDatasetWrapper
        Length-k list of non-overlapping dataset views.

    Notes
    -----
    If a class has fewer samples than `num_shards`, those samples are
    distributed round-robin to avoid empty shards.
    """
    if seed is not None:
        random.seed(seed)

    targets = dataset.targets
    n = len(targets)

    # Group indices by class label
    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        class_to_indices[label].append(idx)

    # Shuffle within each class for randomness
    for label in class_to_indices:
        random.shuffle(class_to_indices[label])

    # Distribute class samples across shards in round-robin fashion
    shard_indices: List[List[int]] = [[] for _ in range(num_shards)]
    for label, indices in class_to_indices.items():
        for i, idx in enumerate(indices):
            shard_indices[i % num_shards].append(idx)

    # Shuffle each shard so batches are mixed across classes
    for shard in shard_indices:
        random.shuffle(shard)

    shards = [dataset.subset(idxs) for idxs in shard_indices]

    # Log partition statistics
    _log_partition_stats(shards, num_shards)

    return shards


def _log_partition_stats(shards: List[ImageDatasetWrapper], num_shards: int) -> None:
    sizes = [len(s) for s in shards]
    total = sum(sizes)
    print(
        f"[Partitioner] k={num_shards} shards | "
        f"sizes: min={min(sizes)}, max={max(sizes)}, total={total}"
    )
    for i, shard in enumerate(shards):
        class_counts: Dict[int, int] = defaultdict(int)
        for label in shard.targets:
            class_counts[label] += 1
        print(f"  Shard {i}: {dict(sorted(class_counts.items()))}")
