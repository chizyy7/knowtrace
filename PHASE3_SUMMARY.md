# KnowTrace Project - Phase 3: Deep Knowledge Tracing (DKT) Summary

## Phase 3: DKT - COMPLETED

### Files Created:
- `src/knowtrace/models/dkt.py`: Deep Knowledge Tracing model implementation
  - Question embedding layer (with padding index 0)
  - LSTM layers (configurable hidden size, layers, dropout)
  - Linear output layer for logits
  - Forward pass and prediction methods

- `src/knowtrace/train/train_dkt.py`: DKT training script
  - Teacher forcing: predicts corrects[:, 1:] from logits[:, :-1]
  - Masked BCE loss (ignores padding positions)
  - Adam optimizer with learning rate from config
  - Early stopping on validation AUC (patience configurable)
  - Saves best model checkpoint to `reports/dkt_best.pt`
  - Saves training history and metrics

- `configs/dkt.yaml`: DKT configuration file
  - max_len: 100
  - hidden_dim: 128
  - num_layers: 2
  - dropout: 0.2
  - lr: 0.001
  - batch_size: 64
  - epochs: 30
  - early_stop_patience: 5

- `tests/test_dkt.py`: Comprehensive DKT test suite (6 tests, all passing)
  - Model initialization and layer shapes
  - Forward pass output shape
  - Prediction probabilities in [0,1]
  - Masking behavior (padding index 0 embeddings zeroed)
  - Gradient flow verification
  - Device compatibility (CPU/CUDA)

### How to Run Phase 3:
```powershell
# 1. Ensure Phase 1 is complete (processed data exists)
# If not, run Phase 1 steps first:
# python -m knowtrace.data.load --synthetic
# python -m knowtrace.data.preprocess --synthetic

# 2. Train DKT model
python -m knowtrace.train.train_dkt --synthetic

# 3. Verify outputs
dir reports
# Should see: dkt_best.pt, dkt_training_history.json, dkt_metrics.json

# 4. Run DKT tests
pytest tests/test_dkt.py -v
```

### Expected Output:
- DKT model trains successfully with masked loss
- Best model checkpoint saved based on validation AUC
- Training history includes train loss and validation metrics per epoch
- Test set evaluation provides AUC, accuracy, and loss
- All tests pass

### Design Choices:
- **Teacher Forcing**: Standard approach for sequence prediction in DKT
- **Masked Loss**: Padding positions excluded from loss computation
- **Early Stopping**: Prevents overfitting, uses validation AUC as metric
- **Configuration Driven**: All hyperparameters configurable via YAML file
- **Modular Design**: Separate DKT class from training script for reusability

### Next Steps:
Proceed to Phase 4: DKT+ (auxiliary features) implementation, which will add:
- Skill embeddings
- Correctness embeddings (previous answer)
- Optionally difficulty embeddings from question text