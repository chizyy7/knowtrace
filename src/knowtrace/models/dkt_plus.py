"""Deep Knowledge Tracing plus (DKT+) model implementation.

Extends DKT with skill embeddings and previous correctness embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import logging

from knowtrace.utils.logger import get_logger

logger = get_logger(__name__)


class DKTPlus(nn.Module):
    """
    Deep Knowledge Tracing plus model.

    Architecture:
    - Question embedding: Embedding(num_questions + 1, hidden_dim)  # +1 for padding idx 0
    - Skill embedding: Embedding(num_skills + 1, hidden_dim)       # +1 for padding idx 0
    - Previous correctness embedding: Embedding(3, hidden_dim)      # 0: incorrect, 1: correct, 2: padding
    - LSTM: LSTM(input_size=3*hidden_dim, hidden_size=hidden_dim, num_layers, batch_first=True, dropout=dropout)
    - Output layer: Linear(hidden_dim, 1)  # logits for binary classification

    The model returns logits; masking is applied in the loss function.
    """

    def __init__(self,
                 num_questions: int,
                 num_skills: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """
        Initialize DKTPlus model.

        Args:
            num_questions: Number of unique questions (excluding padding index 0)
            num_skills: Number of unique skills (excluding padding index 0)
            hidden_dim: Size of embedding and LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(DKTPlus, self).__init__()

        self.num_questions = num_questions
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # We allocate +1 to account for index 0 being reserved for padding
        self.total_question_indices = num_questions + 1
        self.total_skill_indices = num_skills + 1
        # For previous correctness: 0 (incorrect), 1 (correct), 2 (padding)
        self.total_correctness_indices = 3

        # Question embedding layer (index 0 is for padding and will be masked)
        self.question_embedding = nn.Embedding(self.total_question_indices, hidden_dim, padding_idx=0)

        # Skill embedding layer (index 0 is for padding and will be masked)
        self.skill_embedding = nn.Embedding(self.total_skill_indices, hidden_dim, padding_idx=0)

        # Previous correctness embedding layer: 0 for incorrect, 1 for correct, 2 for padding
        self.correctness_embedding = nn.Embedding(3, hidden_dim)

        # LSTM layer: input size is 3 * hidden_dim (concat of three embeddings)
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # PyTorch dropout between layers
        )

        # Output layer: maps LSTM hidden state to logit (binary classification)
        self.output_layer = nn.Linear(hidden_dim, 1)

        logger.info(f"Initialized DKTPlus model: {num_questions} questions, {num_skills} skills, "
                    f"hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")

    def forward(self, q_ids: torch.Tensor, skill_ids: torch.Tensor, prev_correct: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DKTPlus model.

        Args:
            q_ids: LongTensor of question IDs with shape (batch_size, seq_len)
                   Padded with index 0.
            skill_ids: LongTensor of skill IDs with shape (batch_size, seq_len)
                       Padded with index 0.
            prev_correct: LongTensor of previous correctness labels with shape (batch_size, seq_len)
                          Values: 0 (incorrect), 1 (correct)

        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, 1)
                   Raw logits (before sigmoid) for each timestep.
        """
        # Embed questions: (batch_size, seq_len, hidden_dim)
        q_embedded = self.question_embedding(q_ids)
        # Embed skills: (batch_size, seq_len, hidden_dim)
        s_embedded = self.skill_embedding(skill_ids)
        # Embed previous correctness: (batch_size, seq_len, hidden_dim)
        c_embedded = self.correctness_embedding(prev_correct)

        # Concatenate embeddings: (batch_size, seq_len, 3 * hidden_dim)
        embedded = torch.cat([q_embedded, s_embedded, c_embedded], dim=-1)

        # Pass through LSTM: output shape (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(embedded)

        # Output layer: (batch_size, seq_len, 1)
        logits = self.output_layer(lstm_out)

        return logits

    def predict_proba(self, q_ids: torch.Tensor, skill_ids: torch.Tensor, prev_correct: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities of correct answer (after sigmoid).

        Args:
            q_ids: LongTensor of question IDs with shape (batch_size, seq_len)
            skill_ids: LongTensor of skill IDs with shape (batch_size, seq_len)
            prev_correct: LongTensor of previous correctness labels with shape (batch_size, seq_len)

        Returns:
            probs: FloatTensor of shape (batch_size, seq_len, 1)
                   Probabilities after sigmoid activation.
        """
        logits = self.forward(q_ids, skill_ids, prev_correct)
        probs = torch.sigmoid(logits)
        return probs


if __name__ == "__main__":
    # Simple test when run directly
    print("Testing DKTPlus model...")

    # Create a simple test case
    num_questions = 10
    num_skills = 5
    batch_size = 4
    seq_len = 8

    model = DKTPlus(num_questions=num_questions, num_skills=num_skills, hidden_dim=32, num_layers=2, dropout=0.1)

    # Create dummy input (question IDs, skill IDs, and previous correctness, padded with 0/2)
    # Ensure we don't use index 0 for actual questions/skills (reserved for padding)
    q_ids = torch.randint(1, num_questions + 1, (batch_size, seq_len))
    skill_ids = torch.randint(1, num_skills + 1, (batch_size, seq_len))
    # Previous correctness: 0, 1, or 2 (padding). We'll set first timestep to 2 (padding) for first sequence
    prev_correct = torch.randint(0, 3, (batch_size, seq_len))  # random 0,1,2
    prev_correct[0, 0] = 2  # force padding at first timestep for first sequence

    print(f"Input q_ids shape: {q_ids.shape}")
    print(f"Input skill_ids shape: {skill_ids.shape}")
    print(f"Input prev_correct shape: {prev_correct.shape}")

    # Forward pass
    logits = model(q_ids, skill_ids, prev_correct)
    probs = model.predict_proba(q_ids, skill_ids, prev_correct)

    print(f"\nLogits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")

    print(f"\nLogits sample (first sequence):\n{logits[0, :, 0].detach().numpy()}")
    print(f"Probs sample (first sequence):\n{probs[0, :, 0].detach().numpy()}")

    # Check that padding positions produce valid outputs (should not be NaN)
    assert not torch.isnan(logits).any(), "Logits contain NaN values"
    assert not torch.isnan(probs).any(), "Probs contain NaN values"
    assert torch.all(probs >= 0) and torch.all(probs <= 1), "Probs out of [0,1] range"

    print("\nDKTPlus model test completed successfully!")