"""Tests for DKT+ model implementation."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from knowtrace.models.dkt_plus import DKTPlus


def test_dkt_plus_initialization():
    """Test DKT+ model initialization."""
    num_questions = 5
    num_skills = 3
    model = DKTPlus(num_questions=num_questions, num_skills=num_skills, hidden_dim=32, num_layers=2, dropout=0.1)

    # Check that embedding layers have correct shape
    # Question embedding: num_questions + 1 for padding idx 0
    assert model.question_embedding.num_embeddings == num_questions + 1
    assert model.question_embedding.embedding_dim == 32

    # Skill embedding: num_skills + 1 for padding idx 0
    assert model.skill_embedding.num_embeddings == num_skills + 1
    assert model.skill_embedding.embedding_dim == 32

    # Previous correctness embedding: 3 for incorrect/correct/padding
    assert model.correctness_embedding.num_embeddings == 3
    assert model.correctness_embedding.embedding_dim == 32

    # Check LSTM parameters
    assert model.lstm.hidden_size == 32
    assert model.lstm.num_layers == 2

    # Check output layer
    assert model.output_layer.in_features == 32
    assert model.output_layer.out_features == 1


def test_dkt_plus_forward_pass_shape():
    """Test that DKT+ forward pass returns correct shape."""
    num_questions = 10
    num_skills = 5
    batch_size = 4
    seq_len = 8

    model = DKTPlus(num_questions=num_questions, num_skills=num_skills, hidden_dim=16, num_layers=2, dropout=0.1)

    # Create input tensors (question IDs, skill IDs, and previous correctness, padded with 0)
    q_ids = torch.randint(1, num_questions + 1, (batch_size, seq_len))
    skill_ids = torch.randint(1, num_skills + 1, (batch_size, seq_len))
    prev_correct = torch.randint(0, 2, (batch_size, seq_len))  # 0 or 1
    # Add some padding
    q_ids[0, -2:] = 0
    skill_ids[0, -2:] = 0

    logits = model(q_ids, skill_ids, prev_correct)

    # Should output logits of shape (batch_size, seq_len, 1)
    assert logits.shape == (batch_size, seq_len, 1)
    assert logits.dtype == torch.float32


def test_dkt_plus_predict_proba_shape_and_range():
    """Test predict_proba method returns probabilities in [0,1]."""
    num_questions = 5
    num_skills = 3
    batch_size = 2
    seq_len = 6

    model = DKTPlus(num_questions=num_questions, num_skills=num_skills, hidden_dim=8, num_layers=1, dropout=0.0)

    q_ids = torch.randint(1, num_questions + 1, (batch_size, seq_len))
    skill_ids = torch.randint(1, num_skills + 1, (batch_size, seq_len))
    prev_correct = torch.randint(0, 2, (batch_size, seq_len))  # 0 or 1
    q_ids[0, -3:] = 0  # padding
    skill_ids[0, -3:] = 0  # padding

    probs = model.predict_proba(q_ids, skill_ids, prev_correct)

    assert probs.shape == (batch_size, seq_len, 1)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_dkt_plus_masking_behavior():
    """Test that padding positions (index 0) produce consistent embeddings."""
    num_questions = 5
    num_skills = 3
    model = DKTPlus(num_questions=num_questions, num_skills=num_skills, hidden_dim=4, num_layers=1, dropout=0.0)

    # Create two sequences: one with padding at end, one without
    # For no-padding sequence, we need to use valid indices (1 to num_*)
    q_ids1 = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)  # padded
    skill_ids1 = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)  # padded
    prev_correct1 = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.long)  # padded correctness

    q_ids2 = torch.tensor([[1, 2, 3, 1, 2]], dtype=torch.long)  # no padding (using valid indices 1-3)
    skill_ids2 = torch.tensor([[1, 2, 3, 1, 2]], dtype=torch.long)  # no padding (using valid indices 1-3)
    prev_correct2 = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.long)  # no padding

    # Get embeddings directly from embedding layers
    q_emb1 = model.question_embedding(q_ids1)  # (1, 5, 4)
    s_emb1 = model.skill_embedding(skill_ids1)  # (1, 5, 4)
    c_emb1 = model.correctness_embedding(prev_correct1)  # (1, 5, 4)

    q_emb2 = model.question_embedding(q_ids2)  # (1, 5, 4)
    s_emb2 = model.skill_embedding(skill_ids2)  # (1, 5, 4)
    c_emb2 = model.correctness_embedding(prev_correct2)  # (1, 5, 4)

    # Padding positions (index 3,4) in first sequence should have zero vector for question and skill embeddings
    # (because padding_idx=0) and whatever the correctness embedding gives for index 0
    assert torch.all(q_emb1[0, 3:, :] == 0), "Padding positions should have zero question embedding"
    assert torch.all(s_emb1[0, 3:, :] == 0), "Padding positions should have zero skill embedding"
    # Correctness embedding for index 0 is learned, not necessarily zero

    # Non-padding positions should have non-zero embeddings (assuming weights not zero)
    # We'll check that at least some are non-zero
    assert not torch.all(q_emb1[0, :3, :] == 0), "Non-padding positions should not be all zero for questions"
    assert not torch.all(s_emb1[0, :3, :] == 0), "Non-padding positions should not be all zero for skills"


def test_dkt_plus_gradient_flow():
    """Test that gradients flow through the model."""
    num_questions = 5
    num_skills = 3
    model = DKTPlus(num_questions=num_questions, num_skills=num_skills, hidden_dim=8, num_layers=2, dropout=0.1)

    # Create dummy input and target
    q_ids = torch.randint(1, num_questions + 1, (2, 4))
    skill_ids = torch.randint(1, num_skills + 1, (2, 4))
    q_ids[0, -1] = 0  # some padding
    skill_ids[0, -1] = 0  # some padding
    prev_correct = torch.randint(0, 2, (2, 4))  # random 0,1

    # Forward pass
    logits = model(q_ids, skill_ids, prev_correct).squeeze(-1)  # (batch, seq_len)

    # Shift for next-step prediction
    logits_shifted = logits[:, :-1]  # predict t+1 from t
    # Dummy target: random 0/1
    targets = torch.randint(0, 2, (2, 3)).float()

    # Loss
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits_shifted, targets)

    # Backward pass
    loss.backward()

    # Check that gradients exist for key parameters
    assert model.question_embedding.weight.grad is not None
    assert model.skill_embedding.weight.grad is not None
    assert model.correctness_embedding.weight.grad is not None
    assert model.lstm.weight_ih_l0.grad is not None
    assert model.lstm.weight_hh_l0.grad is not None
    assert model.output_layer.weight.grad is not None
    assert model.output_layer.bias.grad is not None

    # Check that gradients are not all zero (or at least some are non-zero)
    grad_emb_q = model.question_embedding.weight.grad
    grad_emb_s = model.skill_embedding.weight.grad
    grad_emb_c = model.correctness_embedding.weight.grad
    grad_lstm = model.lstm.weight_ih_l0.grad
    grad_out = model.output_layer.weight.grad

    assert torch.abs(grad_emb_q).sum() > 0, "Question embedding gradient should not be zero"
    assert torch.abs(grad_emb_s).sum() > 0, "Skill embedding gradient should not be zero"
    assert torch.abs(grad_emb_c).sum() > 0, "Correctness embedding gradient should not be zero"
    assert torch.abs(grad_lstm).sum() > 0, "LSTM gradient should not be zero"
    assert torch.abs(grad_out).sum() > 0, "Output gradient should not be zero"


def test_dkt_plus_device_compatibility():
    """Test that model works on both CPU and CUDA if available."""
    num_questions = 5
    num_skills = 3
    model = DKTPlus(num_questions=num_questions, num_skills=num_skills)

    # CPU test
    q_ids_cpu = torch.randint(1, num_questions + 1, (2, 3))
    skill_ids_cpu = torch.randint(1, num_skills + 1, (2, 3))
    prev_correct_cpu = torch.randint(0, 2, (2, 3))
    logits_cpu = model(q_ids_cpu, skill_ids_cpu, prev_correct_cpu)
    assert logits_cpu.shape == (2, 3, 1)

    # CUDA test if available
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        q_ids_cuda = q_ids_cpu.cuda()
        skill_ids_cuda = skill_ids_cpu.cuda()
        prev_correct_cuda = prev_correct_cpu.cuda()
        logits_cuda = model_cuda(q_ids_cuda, skill_ids_cuda, prev_correct_cuda)
        assert logits_cuda.shape == (2, 3, 1)
        # Check that output is on GPU
        assert logits_cuda.is_cuda


if __name__ == "__main__":
    # Run tests
    test_dkt_plus_initialization()
    test_dkt_plus_forward_pass_shape()
    test_dkt_plus_predict_proba_shape_and_range()
    test_dkt_plus_masking_behavior()
    test_dkt_plus_gradient_flow()
    test_dkt_plus_device_compatibility()
    print("All DKT+ tests passed!")