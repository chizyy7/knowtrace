"""BKT model training script."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np

from knowtrace.models.bkt import BKT, save_bkt_params
from knowtrace.data.dataset import load_sequences
from knowtrace.utils.logger import get_logger
from knowtrace.utils.seed import set_seed
from knowtrace.utils.config import load_config

logger = get_logger(__name__)


def prepare_data_for_bkt(train_sequences: List[Dict],
                        val_sequences: List[Dict],
                        test_sequences: List[Dict]) -> tuple:
    """
    Prepare data sequences for BKT training and evaluation.

    Args:
        train_sequences: Training user sequences
        val_sequences: Validation user sequences
        test_sequences: Test user sequences

    Returns:
        Tuple of (train_skill_seqs, train_correct_seqs,
                  val_skill_seqs, val_correct_seqs,
                  test_skill_seqs, test_correct_seqs)
    """
    def extract_sequences(sequences: List[Dict]) -> tuple:
        skill_seqs = [seq["skill_ids"] for seq in sequences]
        correct_seqs = [seq["corrects"] for seq in sequences]
        return skill_seqs, correct_seqs

    train_skill_seqs, train_correct_seqs = extract_sequences(train_sequences)
    val_skill_seqs, val_correct_seqs = extract_sequences(val_sequences)
    test_skill_seqs, test_correct_seqs = extract_sequences(test_sequences)

    return (train_skill_seqs, train_correct_seqs,
            val_skill_seqs, val_correct_seqs,
            test_skill_seqs, test_correct_seqs)


def evaluate_bkt(bkt_model: BKT,
                skill_sequences: List[List[int]],
                correct_sequences: List[List[int]]) -> Dict[str, float]:
    """
    Evaluate BKT model on sequences.

    Args:
        bkt_model: Trained BKT model
        skill_sequences: List of skill sequences
        correct_sequences: List of correctness sequences

    Returns:
        Dictionary with evaluation metrics
    """
    all_predictions = []
    all_true_labels = []

    for skill_seq, correct_seq in zip(skill_sequences, correct_sequences):
        if not skill_seq:
            continue

        # Get predictions (probability of knowing after each observation)
        predictions = bkt_model.predict(skill_seq, correct_seq)

        # For evaluation, we predict next correctness
        # P(correct_t) = P(known_t) * (1 - P(S)) + P(unknown_t) * P(G)
        prob_correct_next = []
        for i in range(len(predictions)):
            p_known = predictions[i]
            p_unknown = 1 - p_known
            # Get skill for this timestep to get appropriate P(S) and P(G)
            skill_id = skill_seq[i]
            p_l0, p_t, p_g, p_s = bkt_model._get_skill_parameters(skill_id)
            prob_correct = p_known * (1 - p_s) + p_unknown * p_g
            prob_correct_next.append(prob_correct)

        # True labels are the correctness values (shifted for next step prediction)
        # We predict correctness at time t+1 based on knowledge at time t
        # So we compare predictions[t] with correct_seq[t+1]
        if len(correct_seq) > 1:
            all_predictions.extend(prob_correct_next[:-1])  # Exclude last prediction
            all_true_labels.extend(correct_seq[1:])         # Exclude first label

    if not all_predictions:
        return {"auc": 0.0, "accuracy": 0.0}

    # Calculate metrics
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import Binarizer

    try:
        auc = roc_auc_score(all_true_labels, all_predictions)
    except ValueError:
        # Handle case where only one class is present
        auc = 0.5

    # Convert probabilities to binary predictions using 0.5 threshold
    binary_preds = [1 if p >= 0.5 else 0 for p in all_predictions]
    accuracy = accuracy_score(all_true_labels, binary_preds)

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "num_predictions": len(all_predictions)
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train BKT model")
    parser.add_argument("--config", type=str, default="configs/bkt.yaml",
                       help="Path to configuration file")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="reports",
                       help="Output directory for results")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load configuration
    config = load_config(args.config) if Path(args.config).exists() else {}
    # Override with command line arguments
    config.update(vars(args))

    logger.info(f"Starting BKT training with config: {config}")

    # Load or generate data
    if args.synthetic:
        from knowtrace.data.load import generate_synthetic
        from knowtrace.data.preprocess import preprocess, PreprocessConfig

        # Generate synthetic data
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "skill_builder_data.csv"

        if not raw_path.exists():
            logger.info("Generating synthetic data...")
            df_synthetic = generate_synthetic(seed=args.seed)
            df_synthetic.to_csv(raw_path, index=False)
            logger.info(f"Synthetic data saved to {raw_path}")

        # Preprocess data
        prep_config = PreprocessConfig(seed=args.seed)
        processed_sequences, id_maps = preprocess(prep_config)
        num_skills = len(id_maps["skill_id"])
        logger.info(f"Preprocessed data: {num_skills} skills")
    else:
        # Load existing processed data
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            logger.error("Processed data not found. Run preprocessing first.")
            sys.exit(1)

        # Load sequences
        from knowtrace.data.dataset import load_sequences
        train_sequences = load_sequences(processed_dir / "train.jsonl")
        val_sequences = load_sequences(processed_dir / "val.jsonl")
        test_sequences = load_sequences(processed_dir / "test.jsonl")

        processed_sequences = {
            "train": train_sequences,
            "val": val_sequences,
            "test": test_sequences
        }

        # Load ID maps to get number of skills
        with open(processed_dir / "id_maps.json", 'r') as f:
            id_maps = json.load(f)
        num_skills = len(id_maps["skill_id"])
        logger.info(f"Loaded data: {num_skills} skills")

    # Prepare data for BKT
    train_skill_seqs, train_correct_seqs, \
    val_skill_seqs, val_correct_seqs, \
    test_skill_seqs, test_correct_seqs = prepare_data_for_bkt(
        processed_sequences["train"],
        processed_sequences["val"],
        processed_sequences["test"]
    )

    logger.info(f"Training sequences: {len(train_skill_seqs)}")
    logger.info(f"Validation sequences: {len(val_skill_seqs)}")
    logger.info(f"Test sequences: {len(test_skill_seqs)}")

    # Initialize BKT model
    bkt_config = {
        'num_skills': num_skills,
        'p_l0_init': config.get('p_l0_init', 0.5),
        'p_t_init': config.get('p_t_init', 0.3),
        'p_g_init': config.get('p_g_init', 0.2),
        'p_s_init': config.get('p_s_init', 0.1),
        'global_fallback_threshold': config.get('global_fallback_threshold', 5)
    }

    bkt_model = BKT(**bkt_config)
    logger.info(f"Initialized BKT model with {num_skills} skills")

    # Train model using EM
    logger.info("Starting EM training...")
    param_history = bkt_model.fit_em(
        train_skill_seqs,
        train_correct_seqs,
        max_iterations=config.get('max_em_iterations', 25),
        tolerance=float(config.get('em_tolerance', 1e-4))
    )

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_metrics = evaluate_bkt(bkt_model, val_skill_seqs, val_correct_seqs)
    logger.info(f"Validation metrics: {val_metrics}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_bkt(bkt_model, test_skill_seqs, test_correct_seqs)
    logger.info(f"Test metrics: {test_metrics}")

    # Save parameters
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = bkt_model.get_parameters()
    params_file = output_dir / "bkt_params.json"
    save_bkt_params(params, params_file)

    # Save training history
    history_file = output_dir / "bkt_training_history.json"
    # Convert numpy arrays in history to lists for JSON serialization
    serializable_history = {}
    for key, value in param_history.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], np.ndarray):
                serializable_history[key] = [v.tolist() for v in value]
            else:
                serializable_history[key] = value
        else:
            serializable_history[key] = value

    with open(history_file, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    logger.info(f"Training history saved to {history_file}")

    # Save metrics
    metrics_file = output_dir / "bkt_metrics.json"
    metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "config": bkt_config
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    logger.info("BKT training completed successfully!")


if __name__ == "__main__":
    main()