"""Data loading utilities for KnowTrace."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import logging

from knowtrace.utils.logger import get_logger

logger = get_logger(__name__)

# Map canonical column names to possible variants in raw data
COLUMN_ALIASES = {
    "user_id": ["user_id", "userId", "userid", "uid"],
    "skill_id": ["skill_id", "skillId", "skillid", "sid"],
    "problem_id": ["problem_id", "problemId", "problemid", "pid"],
    "correct": ["correct", "Correct", "is_correct", "success"],
    "order_id": ["order_id", "orderId", "orderid", "timestamp", "time", "idx"]
}

def _find_column(df: pd.DataFrame, canonical_name: str) -> str:
    """
    Find the actual column name in DataFrame that matches canonical name.

    Args:
        df: Input DataFrame
        canonical_name: Standardized column name to find

    Returns:
        Actual column name found in DataFrame

    Raises:
        KeyError: If no matching column is found
    """
    possible_names = COLUMN_ALIASES[canonical_name]
    for name in possible_names:
        if name in df.columns:
            return name
    raise KeyError(f"Could not find column for '{canonical_name}'. "
                   f"Tried: {possible_names}")

def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """
    Load raw CSV and standardize column names.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with standardized column names:
        ["user_id", "skill_id", "problem_id", "correct", "order_id"]
    """
    logger.info(f"Loading raw data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize column names
    standardized_data = {}
    for canon_name in COLUMN_ALIASES.keys():
        actual_name = _find_column(df, canon_name)
        standardized_data[canon_name] = df[actual_name]

    result = pd.DataFrame(standardized_data)

    # Ensure correct column is numeric (0/1)
    result["correct"] = pd.to_numeric(result["correct"], errors="coerce")
    # Fill any NaN with 0 (incorrect)
    result["correct"] = result["correct"].fillna(0).astype(int)

    logger.info(f"Loaded {len(result)} interactions")
    logger.info(f"Unique users: {result['user_id'].nunique()}")
    logger.info(f"Unique skills: {result['skill_id'].nunique()}")
    logger.info(f"Unique problems: {result['problem_id'].nunique()}")

    return result

def generate_synthetic(n_users: int = 200, n_skills: int = 15,
                      n_problems: int = 60, max_seq_len: int = 80,
                      seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic ASSISTments-like data for testing.

    Each user has latent per-skill ability. Problem difficulty is sampled
    from normal distribution. Probability correct = sigmoid(ability - difficulty).

    Args:
        n_users: Number of synthetic users
        n_skills: Number of synthetic skills
        n_problems: Number of synthetic problems
        max_seq_len: Maximum sequence length per user
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic data matching expected schema
    """
    logger.info(f"Generating synthetic data: {n_users} users, "
                f"{n_skills} skills, {n_problems} problems")

    np.random.seed(seed)

    # Create ID mappings
    user_ids = [f"user_{i}" for i in range(n_users)]
    skill_ids = [f"skill_{i}" for i in range(n_skills)]
    problem_ids = [f"problem_{i}" for i in range(n_problems)]

    # Latent user abilities per skill (users have different strengths)
    user_abilities = np.random.normal(0, 1, size=(n_users, n_skills))

    # Problem difficulties (higher = harder)
    problem_difficulties = np.random.normal(0, 1, size=n_problems)
    # Map problems to skills
    problem_skill_map = np.random.randint(0, n_skills, size=n_problems)

    records = []

    for user_idx, user_id in enumerate(user_ids):
        # Variable sequence length for each user
        seq_len = np.random.randint(5, max_seq_len + 1)

        for _ in range(seq_len):
            # Randomly select a skill and problem
            skill_idx = np.random.randint(0, n_skills)
            skill_id = skill_ids[skill_idx]

            problem_idx = np.random.randint(0, n_problems)
            problem_id = problem_ids[problem_idx]

            # Verify the problem belongs to the selected skill (with some noise)
            if np.random.random() < 0.8:  # 80% chance problem matches skill
                # Ensure problem maps to selected skill
                problem_indices_for_skill = np.where(problem_skill_map == skill_idx)[0]
                if len(problem_indices_for_skill) > 0:
                    problem_idx = np.random.choice(problem_indices_for_skill)
                    problem_id = problem_ids[problem_idx]

            # Get user's ability for this skill
            ability = user_abilities[user_idx, skill_idx]

            # Get problem difficulty
            difficulty = problem_difficulties[problem_idx]

            # Probability correct via sigmoid
            prob_correct = 1 / (1 + np.exp(-(ability - difficulty)))
            correct = np.random.random() < prob_correct

            # Generate order ID (timestamp-like)
            order_id = len(records)  # Simple incrementing ID

            records.append({
                "user_id": user_id,
                "skill_id": skill_id,
                "problem_id": problem_id,
                "correct": int(correct),
                "order_id": order_id
            })

    df = pd.DataFrame(records)
    # Sort by user and order for consistency
    df = df.sort_values(["user_id", "order_id"]).reset_index(drop=True)

    logger.info(f"Generated {len(df)} synthetic interactions")
    logger.info(f"Overall correctness rate: {df['correct'].mean():.3f}")

    return df

def main():
    """Command-line interface for data loading."""
    parser = argparse.ArgumentParser(description="Load or generate ASSISTments data")
    parser.add_argument("--synthetic", action="store_true",
                       help="Generate synthetic data instead of loading from file")
    parser.add_argument("--output", type=str, default="data/raw/skill_builder_data.csv",
                       help="Output CSV file path")
    parser.add_argument("--n_users", type=int, default=200,
                       help="Number of users for synthetic data")
    parser.add_argument("--n_skills", type=int, default=15,
                       help="Number of skills for synthetic data")
    parser.add_argument("--n_problems", type=int, default=60,
                       help="Number of problems for synthetic data")
    parser.add_argument("--max_seq_len", type=int, default=80,
                       help="Max sequence length per user for synthetic data")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for synthetic data")

    args = parser.parse_args()

    if args.synthetic:
        logger.info("Generating synthetic data...")
        df = generate_synthetic(
            n_users=args.n_users,
            n_skills=args.n_skills,
            n_problems=args.n_problems,
            max_seq_len=args.max_seq_len,
            seed=args.seed
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic data saved to {output_path}")
    else:
        parser.error("Either --synthetic must be specified or a CSV file provided")

if __name__ == "__main__":
    main()