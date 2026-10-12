"""DKT model training script."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from knowtrace.models.dkt import DKT
from knowtrace.data.dataset import load_sequences, KTDataset, pad_collate, create_data_loaders
from knowtrace.utils.logger import get_logger
from knowtrace.utils.seed import set_seed
from knowtrace.utils.config import load_config

logger = get_logger(__name__)


def prepare_data_for_dkt(train_sequences: List[Dict],
                        val_sequences: List[Dict],
                        test_sequences: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Prepare data sequences for DKT training and evaluation.

    For DKT, we need to shift the correctness labels for teacher forcing:
    - Input at time t: question_id_t
    - Target at time t: correctness_{t+1} (we predict next correctness)
    - However, the spec says: predict corrects[:, 1:] from preds[:, :-1]
    - So we will use the sequences as is, and in the training loop we will shift.

    The dataset returns sequences as-is; we will handle shifting in the training loop.

    Args:
        train_sequences: Training user sequences
        val_sequences: Validation user sequences
        test_sequences: Test user sequences

    Returns:
        Same sequences (the shifting is done in training loop)
    """
    return train_sequences, val_sequences, test_sequences


def train_epoch(model: DKT,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
    """
    Train for one epoch.

    Args:
        model: DKT model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to run on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_batches = 0

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # we'll apply mask manually

    for batch in train_loader:
        # Move batch to device
        q_ids = batch["q_ids"].to(device)          # (batch_size, seq_len)
        skill_ids = batch["skill_ids"].to(device)  # not used in DKT
        corrects = batch["corrects"].to(device)    # (batch_size, seq_len)
        mask = batch["mask"].to(device)            # (batch_size, seq_len) bool

        optimizer.zero_grad()

        # Forward pass: predict logits for each timestep
        logits = model(q_ids)  # (batch_size, seq_len, 1)
        logits = logits.squeeze(-1)  # (batch_size, seq_len)

        # For teacher forcing: predict next correctness
        # We use logits[:, :-1] to predict corrects[:, 1:]
        logits_shifted = logits[:, :-1]      # (batch_size, seq_len-1)
        targets_shifted = corrects[:, 1:]    # (batch_size, seq_len-1)
        mask_shifted = mask[:, 1:]           # (batch_size, seq_len-1)

        # Compute loss per element
        loss_per_element = loss_fn(logits_shifted, targets_shifted.float())  # (batch_size, seq_len-1)

        # Apply mask: zero out loss where mask is 0 (padding)
        masked_loss = loss_per_element * mask_shifted.float()

        # Average over non-masked elements
        loss = masked_loss.sum() / mask_shifted.float().sum()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate(model: DKT,
             data_loader: DataLoader,
             device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: DKT model
        data_loader: DataLoader for evaluation data
        device: Device to run on

    Returns:
        Dictionary with AUC and accuracy
    """
    model.eval()
    all_probs = []
    all_targets = []

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            q_ids = batch["q_ids"].to(device)
            corrects = batch["corrects"].to(device)
            mask = batch["mask"].to(device)

            logits = model(q_ids).squeeze(-1)  # (batch_size, seq_len)

            # Shift for next step prediction
            logits_shifted = logits[:, :-1]
            targets_shifted = corrects[:, 1:]
            mask_shifted = mask[:, 1:]

            # Compute loss
            loss_per_element = loss_fn(logits_shifted, targets_shifted.float())
            masked_loss = loss_per_element * mask_shifted.float()
            loss = masked_loss.sum() / mask_shifted.float().sum()
            total_loss += loss.item()
            total_batches += 1

            # Get probabilities for timesteps we predicted (t=1..T-1)
            probs_shifted = torch.sigmoid(logits_shifted)  # (batch_size, seq_len-1)

            # Store for metric calculation (only where mask is 1)
            probs_list = probs_shifted[mask_shifted.bool()].cpu().numpy()
            targets_list = targets_shifted[mask_shifted.bool()].cpu().numpy()

            all_probs.extend(probs_list)
            all_targets.extend(targets_list)

    if len(all_probs) == 0:
        return {"auc": 0.0, "accuracy": 0.0, "loss": float('inf')}

    # Calculate metrics
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        # Handle case where only one class is present
        auc = 0.5

    # Accuracy using 0.5 threshold
    preds = (np.array(all_probs) >= 0.5).astype(int)
    accuracy = (preds == np.array(all_targets)).mean()

    avg_loss = total_loss / max(total_batches, 1)

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "loss": avg_loss
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DKT model")
    parser.add_argument("--config", type=str, default="configs/dkt.yaml",
                       help="Path to configuration file")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="reports",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use: 'cpu', 'cuda', or 'auto'")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load configuration
    config = load_config(args.config) if Path(args.config).exists() else {}
    # Override with command line arguments
    config.update(vars(args))

    logger.info(f"Starting DKT training with config: {config}")

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

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
        num_questions = len(id_maps["problem_id"])
        logger.info(f"Preprocessed data: {num_questions} questions")
    else:
        # Load existing processed data
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            logger.error("Processed data not found. Run preprocessing first.")
            sys.exit(1)

        # Load sequences
        train_sequences = load_sequences(processed_dir / "train.jsonl")
        val_sequences = load_sequences(processed_dir / "val.jsonl")
        test_sequences = load_sequences(processed_dir / "test.jsonl")

        processed_sequences = {
            "train": train_sequences,
            "val": val_sequences,
            "test": test_sequences
        }

        # Load ID maps to get number of questions
        with open(processed_dir / "id_maps.json", 'r') as f:
            id_maps = json.load(f)
        num_questions = len(id_maps["problem_id"])
        logger.info(f"Loaded data: {num_questions} questions")

    # Prepare data for DKT (no shifting needed here)
    train_seq, val_seq, test_seq = prepare_data_for_dkt(
        processed_sequences["train"],
        processed_sequences["val"],
        processed_sequences["test"]
    )

    logger.info(f"Training sequences: {len(train_seq)}")
    logger.info(f"Validation sequences: {len(val_seq)}")
    logger.info(f"Test sequences: {len(test_seq)}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_seq, val_seq, test_seq,
        max_len=config.get("max_len", 100),
        batch_size=config.get("batch_size", 64),
        num_workers=0  # 0 for simplicity on Windows
    )

    # Initialize DKT model
    dkt_config = {
        'num_questions': num_questions,
        'hidden_dim': config.get('hidden_dim', 128),
        'num_layers': config.get('num_layers', 2),
        'dropout': config.get('dropout', 0.2)
    }

    model = DKT(**dkt_config)
    model = model.to(device)
    logger.info(f"Initialized DKT model: {num_questions} questions")

    # Setup optimizer and training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    epochs = config.get('epochs', 30)
    early_stop_patience = config.get('early_stop_patience', 5)

    # Training loop with early stopping
    best_val_auc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    train_losses = []
    val_metrics_history = []

    logger.info(f"Starting training for {epochs} epochs with early stopping patience {early_stop_patience}")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        val_metrics_history.append(val_metrics)

        logger.info(f"Epoch {epoch:02d}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_auc={val_metrics['auc']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}")

        # Early stopping check
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            logger.info(f"  -> New best validation AUC: {best_val_auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation AUC: {best_val_auc:.4f}")

    # Evaluate on test set using best model
    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"Test results: AUC={test_metrics['auc']:.4f}, "
                f"Accuracy={test_metrics['accuracy']:.4f}, "
                f"Loss={test_metrics['loss']:.4f}")

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    model_path = output_dir / "dkt_best.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': dkt_config,
        'epoch': epoch,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_metrics_history': val_metrics_history,
        'test_metrics': test_metrics,
        'config': {**dkt_config, **{
            'lr': config.get('lr', 0.001),
            'batch_size': config.get('batch_size', 64),
            'epochs': epochs,
            'early_stop_patience': early_stop_patience
        }}
    }
    history_path = output_dir / "dkt_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    # Save metrics
    metrics = {
        "validation": val_metrics_history[-1] if val_metrics_history else {"auc": 0.0, "accuracy": 0.0, "loss": 0.0},
        "test": test_metrics,
        "config": dkt_config
    }
    metrics_path = output_dir / "dkt_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("DKT training completed successfully!")


if __name__ == "__main__":
    main()