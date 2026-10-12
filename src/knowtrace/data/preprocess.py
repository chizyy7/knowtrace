"""Data preprocessing pipeline for KnowTrace."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import logging

from knowtrace.utils.logger import get_logger
from knowtrace.utils.seed import set_seed

logger = get_logger(__name__)

@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""
    raw_path: str = "data/raw/skill_builder_data.csv"
    processed_dir: str = "data/processed"
    min_interactions: int = 10
    min_skill_freq: int = 20
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1  # Derived from 1 - train_frac - val_frac
    seed: int = 42

    def __post_init__(self):
        """Validate fractions sum to 1.0."""
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Fractions must sum to 1.0, got {total}")

def preprocess(config: PreprocessConfig) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    """
    Preprocess raw data into user sequences for knowledge tracing.

    Args:
        config: Preprocessing configuration

    Returns:
        Tuple of (processed_sequences, id_maps) where:
        - processed_sequences: dict with keys 'train', 'val', 'test' containing
          lists of user sequences (each sequence is a dict)
        - id_maps: dict mapping original IDs to internal indices for each entity type
    """
    logger.info(f"Starting preprocessing with config: {config}")

    # Set seed for reproducibility
    set_seed(config.seed)

    # Load raw data
    from .load import load_raw
    df = load_raw(config.raw_path)

    # Filter users with insufficient interactions
    user_interaction_counts = df.groupby("user_id").size()
    valid_users = user_interaction_counts[
        user_interaction_counts >= config.min_interactions
    ].index.tolist()
    df = df[df["user_id"].isin(valid_users)].copy()
    logger.info(f"After filtering users with < {config.min_interactions} interactions: "
                f"{len(df)} rows, {df['user_id'].nunique()} users")

    # Filter skills with insufficient frequency
    skill_interaction_counts = df.groupby("skill_id").size()
    valid_skills = skill_interaction_counts[
        skill_interaction_counts >= config.min_skill_freq
    ].index.tolist()
    df = df[df["skill_id"].isin(valid_skills)].copy()
    logger.info(f"After filtering skills with < {config.min_skill_freq} interactions: "
                f"{len(df)} rows, {df['skill_id'].nunique()} skills")

    # Create ID maps (index 0 reserved for padding)
    def create_id_map(series: pd.Series) -> Dict[str, int]:
        """Create mapping from original IDs to consecutive integers starting at 1."""
        unique_ids = serie
        return {uid: idx + 1 for idx, uid in enumerate(unique_ids)}

    user_id_map = create_id_map(df["user_id"])
    skill_id_map = create_id_map(df["skill_id"])
    problem_id_map = create_id_map(df["problem_id"])

    id_maps = {
        "user_id": user_id_map,
        "skill_id": skill_id_map,
        "problem_id": problem_id_map
    }

    logger.info(f"ID maps sizes: users={len(user_id_map)}, "
                f"skills={len(skill_id_map)}, problems={len(problem_id_map)}")

    # Map IDs to indices
    df["user_id_idx"] = df["user_id"].map(user_id_map)
    df["skill_id_idx"] = df["skill_id"].map(skill_id_map)
    df["problem_id_idx"] = df["problem_id"].map(problem_id_map)

    # Build per-user sequences
    user_sequences = []
    for user_id, user_df in df.groupby("user_id"):
        # Sort by order_id to maintain temporal sequence
        user_df = user_df.sort_values("order_id")

        sequence = {
            "user_id": user_id,
            "user_id_idx": int(user_df["user_id_idx"].iloc[0]),  # Same for all rows
            "q_ids": user_df["problem_id_idx"].tolist(),
            "skill_ids": user_df["skill_id_idx"].tolist(),
            "corrects": user_df["correct"].tolist()
        }
        user_sequences.append(sequence)

    logger.info(f"Built {len(user_sequences)} user sequences")

    # Split by user (not interaction) to prevent leakage
    np.random.seed(config.seed)
    user_indices = np.random.permutation(len(user_sequences))

    n_train = int(len(user_sequences) * config.train_frac)
    n_val = int(len(user_sequences) * config.val_frac)

    train_indices = user_indices[:n_train]
    val_indices = user_indices[n_train:n_train + n_val]
    test_indices = user_indices[n_train + n_val:]

    train_sequences = [user_sequences[i] for i in train_indices]
    val_sequences = [user_sequences[i] for i in val_indices]
    test_sequences = [user_sequences[i] for i in test_indices]

    logger.info(f"Split sizes: train={len(train_sequences)}, "
                f"val={len(val_sequences)}, test={len(test_sequences)}")

    # Verify no user overlap between splits
    train_users = set(s["user_id"] for s in train_sequences)
    val_users = set(s["user_id"] for s in val_sequences)
    test_users = set(s["user_id"] for s in test_sequences)

    overlap_train_val = train_users.intersection(val_users)
    overlap_train_test = train_users.intersection(test_users)
    overlap_val_test = val_users.intersection(test_users)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(f"User overlap detected between splits: "
                         f"train-val={len(overlap_train_val)}, "
                         f"train-test={len(overlap_train_test)}, "
                         f"val-test={len(overlap_val_test)}")

    logger.info("Verified no user overlap between train/val/test splits")

    # Save processed data
    processed_dir = Path(config.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save sequences as JSONL (one JSON per line per user)
    def save_sequences(sequences: List[Dict], filename: str):
        """Save user sequences to JSONL file."""
        with open(processed_dir / filename, 'w') as f:
            for seq in sequences:
                # Convert numpy types to native Python for JSON serialization
                json_seq = {
                    "user_id": seq["user_id"],
                    "q_ids": seq["q_ids"],
                    "skill_ids": seq["skill_ids"],
                    "corrects": seq["corrects"]
                }
                f.write(json.dumps(json_seq) + '\n')

    save_sequences(train_sequences, "train.jsonl")
    save_sequences(val_sequences, "val.jsonl")
    save_sequences(test_sequences, "test.jsonl")

    # Save ID maps
    with open(processed_dir / "id_maps.json", 'w') as f:
        # Convert numpy types to native Python
        serializable_maps = {}
        for key, id_map in id_maps.items():
            serializable_maps[key] = {k: int(v) for k, v in id_map.items()}
        json.dump(serializable_maps, f, indent=2)

    logger.info(f"Saved processed data to {processed_dir}")

    # Return processed sequences and id maps for potential direct use
    processed_sequences = {
        "train": train_sequences,
        "val": val_sequences,
        "test": test_sequences
    }

    return processed_sequences, id_maps

def main():
    """Command-line interface for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess ASSISTments data")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data generated by load.py")
    parser.add_argument("--raw", type=str, default="data/raw/skill_builder_data.csv",
                       help="Path to raw CSV file")
    parser.add_argument("--out", type=str, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--min_interactions", type=int, default=10,
                       help="Minimum interactions per user")
    parser.add_argument("--min_skill_freq", type=int, default=20,
                       help="Minimum interactions per skill")
    parser.add_argument("--train_frac", type=float, default=0.8,
                       help="Fraction of users for training")
    parser.add_argument("--val_frac", type=float, default=0.1,
                       help="Fraction of users for validation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    config = PreprocessConfig(
        raw_path=args.raw,
        processed_dir=args.out,
        min_interactions=args.min_interactions,
        min_skill_freq=args.min_skill_freq,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed
    )

    if args.synthetic:
        # Ensure synthetic data exists
        from .load import generate_synthetic
        import os
        if not os.path.exists(args.raw):
            logger.info("Generating synthetic data for preprocessing...")
            df = generate_synthetic(seed=args.seed)
            os.makedirs(os.path.dirname(args.raw), exist_ok=True)
            df.to_csv(args.raw, index=False)
            logger.info(f"Synthetic data saved to {args.raw}")

    preprocess(config)

if __name__ == "__main__":
    main()