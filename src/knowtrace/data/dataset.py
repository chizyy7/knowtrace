"""PyTorch Dataset for Knowledge Tracing."""

import json
from pathlib import Path
from typing import Dict, List, Iterator, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging

from knowtrace.utils.logger import get_logger

logger = get_logger(__name__)


def load_sequences(jsonl_path: str | Path) -> List[Dict]:
    """
    Load user sequences from JSONL file.

    Args:
        jsonl_path: Path to JSONL file (one JSON object per line)

    Returns:
        List of dictionaries, each representing a user sequence
    """
    logger.info(f"Loading sequences from {jsonl_path}")
    sequences = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                seq = json.loads(line)
                sequences.append(seq)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    logger.info(f"Loaded {len(sequences)} sequences")
    return sequences


class KTDataset(Dataset):
    """
    PyTorch Dataset for Knowledge Tracing data.

    Each item returns a dictionary with:
    - q_ids: LongTensor of question/problem IDs (padded)
    - skill_ids: LongTensor of skill IDs (padded)
    - corrects: FloatTensor of correctness labels (padded)
    - mask: BoolTensor indicating valid positions (not padding)
    """

    def __init__(self, sequences: List[Dict], max_len: int = 100):
        """
        Initialize dataset.

        Args:
            sequences: List of user sequence dictionaries from load_sequences
            max_len: Maximum sequence length for padding/truncation
        """
        self.sequences = sequences
        self.max_len = max_len
        logger.info(f"Initialized KTDataset with {len(sequences)} sequences, max_len={max_len}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single user sequence, padded/truncated to max_len.

        Returns:
            Dictionary with tensorized sequence data
        """
        seq = self.sequences[idx]

        # Extract raw sequences
        q_ids = seq["q_ids"]
        skill_ids = seq["skill_ids"]
        corrects = seq["corrects"]

        # Truncate if too long
        if len(q_ids) > self.max_len:
            q_ids = q_ids[-self.max_len:]  # Keep most recent interactions
            skill_ids = skill_ids[-self.max_len:]
            corrects = corrects[-self.max_len:]

        # Pad if too short
        pad_length = self.max_len - len(q_ids)
        if pad_length > 0:
            q_ids = [0] * pad_length + q_ids  # 0 is padding index
            skill_ids = [0] * pad_length + skill_ids
            corrects = [0.0] * pad_length + corrects  # 0.0 for correctness (will be masked)

        # Create mask: 1 for real data, 0 for padding
        mask = [1 if i < len(seq["q_ids"]) else 0 for i in range(self.max_len)]
        # Alternatively: mask = [1 if x != 0 else 0 for x in q_ids]  # But this assumes no real 0 IDs

        return {
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "skill_ids": torch.tensor(skill_ids, dtype=torch.long),
            "corrects": torch.tensor(corrects, dtype=torch.float),
            "mask": torch.tensor(mask, dtype=torch.bool)
        }


def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to stack tensors.

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        Dictionary with batched tensors
    """
    # Stack each tensor type
    batched = {}
    for key in batch[0].keys():
        batched[key] = torch.stack([item[key] for item in batch])
    return batched


def create_data_loaders(train_sequences: List[Dict], val_sequences: List[Dict],
                       test_sequences: List[Dict], max_len: int = 100,
                       batch_size: int = 64, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        train_sequences: Training user sequences
        val_sequences: Validation user sequences
        test_sequences: Test user sequences
        max_len: Maximum sequence length
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes (0 for main process only)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = KTDataset(train_sequences, max_len)
    val_dataset = KTDataset(val_sequences, max_len)
    test_dataset = KTDataset(test_sequences, max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=pad_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=pad_collate
    )

    logger.info(f"Created DataLoaders: batch_size={batch_size}, "
                f"train_batches={len(train_loader)}, "
                f"val_batches={len(val_loader)}, "
                f"test_batches={len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Simple test when run directly
    import tempfile
    import os

    # Create dummy sequences for testing
    dummy_sequences = [
        {
            "user_id": "user_0",
            "q_ids": [1, 2, 3],
            "skill_ids": [1, 2, 1],
            "corrects": [1, 0, 1]
        },
        {
            "user_id": "user_1",
            "q_ids": [4, 5],
            "skill_ids": [2, 2],
            "corrects": [0, 1]
        }
    ]

    dataset = KTDataset(dummy_sequences, max_len=5)
    print(f"Dataset length: {len(dataset)}")

    for i in range(len(dataset)):
        item = dataset[i]
        print(f"Sequence {i}:")
        print(f"  q_ids: {item['q_ids']}")
        print(f"  skill_ids: {item['skill_ids']}")
        print(f"  corrects: {item['corrects']}")
        print(f"  mask: {item['mask']}")

    # Test collate function
    batch = [dataset[0], dataset[1]]
    collated = pad_collate(batch)
    print(f"\nBatched q_ids shape: {collated['q_ids'].shape}")
    print(f"Batched mask shape: {collated['mask'].shape}")