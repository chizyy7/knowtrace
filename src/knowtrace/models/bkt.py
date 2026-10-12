"""Bayesian Knowledge Tracing model implementation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from pathlib import Path

from knowtrace.utils.logger import get_logger

logger = get_logger(__name__)


class BKT:
    """
    Bayesian Knowledge Tracing model with 4 parameters per skill:
    - P(L0): Initial probability of knowing the skill
    - P(T): Probability of learning the skill (transition from not known to known)
    - P(G): Probability of guessing (correct answer when not known)
    - P(S): Probability of slipping (incorrect answer when known)
    """

    def __init__(self,
                 num_skills: int,
                 p_l0_init: float = 0.5,
                 p_t_init: float = 0.3,
                 p_g_init: float = 0.2,
                 p_s_init: float = 0.1,
                 global_fallback_threshold: int = 5):
        """
        Initialize BKT model.

        Args:
            num_skills: Number of unique skills (excluding padding index 0)
            p_l0_init: Initial value for P(L0) parameter
            p_t_init: Initial value for P(T) parameter
            p_g_init: Initial value for P(G) parameter
            p_s_init: Initial value for P(S) parameter
            global_fallback_threshold: Minimum number of observations per skill
                                     to use skill-specific parameters (else use global)
        """
        self.num_skills = num_skills
        self.global_fallback_threshold = global_fallback_threshold
        # We allocate num_skills + 1 to account for index 0 being reserved for padding
        # Actual skills will use indices 1 through num_skills
        self.total_indices = num_skills + 1

        # Initialize parameters (index 0 is for padding and will not be used)
        self.p_l0 = np.full(self.total_indices, p_l0_init)
        self.p_t = np.full(self.total_indices, p_t_init)
        self.p_g = np.full(self.total_indices, p_g_init)
        self.p_s = np.full(self.total_indices, p_s_init)

        # Track observation counts per skill for fallback decision
        # Index 0 is for padding and will not be used
        self.skill_observation_counts = np.zeros(self.total_indices)

        # Global parameters (fallback when skill has insufficient data)
        self.global_p_l0 = p_l0_init
        self.global_p_t = p_t_init
        self.global_p_g = p_g_init
        self.global_p_s = p_s_init

        logger.info(f"Initialized BKT model with {num_skills} skills "
                   f"(indices 1-{num_skills}, index 0 reserved for padding)")

    def _get_skill_parameters(self, skill_id: int) -> Tuple[float, float, float, float]:
        """
        Get parameters for a specific skill, using global fallback if needed.

        Args:
            skill_id: Index of the skill (should be >= 1, with 0 reserved for padding)

        Returns:
            Tuple of (p_l0, p_t, p_g, p_s) for the skill
        """
        # Ensure skill_id is within valid range (1 to num_skills)
        # Index 0 is reserved for padding and should not be used for actual skills
        if skill_id < 1 or skill_id >= self.total_indices:
            raise ValueError(f"Invalid skill_id: {skill_id}. "
                           f"Skill IDs should be in range [1, {self.num_skills}] "
                           f"with 0 reserved for padding.")

        if self.skill_observation_counts[skill_id] < self.global_fallback_threshold:
            # Use global parameters
            return (self.global_p_l0, self.global_p_t, self.global_p_g, self.global_p_s)
        else:
            # Use skill-specific parameters
            return (self.p_l0[skill_id], self.p_t[skill_id],
                    self.p_g[skill_id], self.p_s[skill_id])

    def predict(self,
                skill_sequence: List[int],
                correct_sequence: List[int]) -> List[float]:
        """
        Predict probability of knowing each skill after each observation.

        Args:
            skill_sequence: List of skill indices for each observation
            correct_sequence: List of correctness (0 or 1) for each observation

        Returns:
            List of probabilities of knowing the skill after each observation
        """
        if len(skill_sequence) != len(correct_sequence):
            raise ValueError("Skill sequence and correct sequence must have same length")

        # Initialize knowledge state
        # For the first observation, we use the initial P(L0) for that skill
        if skill_sequence:
            first_skill_id = skill_sequence[0]
            p_l0, _, _, _ = self._get_skill_parameters(first_skill_id)
            p_known = p_l0
        else:
            p_known = 0.0

        predictions = []

        for i, (skill_id, correct) in enumerate(zip(skill_sequence, correct_sequence)):
            # Get skill-specific parameters
            p_l0, p_t, p_g, p_s = self._get_skill_parameters(skill_id)

            if i == 0:
                # First observation: p_known_before is the initial P(L0)
                p_known_before = p_l0
            else:
                # Subsequent observations: update knowledge state based on previous observation
                # p_known_before = P(known at t-1 | observation at t-1)
                p_known_before = self._update_knowledge_state(
                    p_known_before, correct, p_l0, p_t, p_g, p_s
                )

            # Predict probability of knowing after this observation
            p_known_after = self._update_knowledge_state(
                p_known_before, correct, p_l0, p_t, p_g, p_s
            )

            predictions.append(p_known_after)
            p_known = p_known_after

            # Update observation count for this skill
            self.skill_observation_counts[skill_id] += 1

        return predictions

    def _update_knowledge_state(self,
                                p_known_before: float,
                                correct: int,
                                p_l0: float,
                                p_t: float,
                                p_g: float,
                                p_s: float) -> float:
        """
        Update knowledge state using BKT equations.

        Args:
            p_known_before: Probability of knowing before observation
            correct: Correctness of observation (0 or 1)
            p_l0, p_t, p_g, p_s: BKT parameters

        Returns:
            Updated probability of knowing after observation
        """
        # Probability of correct answer given current knowledge state
        p_correct = p_known_before * (1 - p_s) + (1 - p_known_before) * p_g

        # Avoid division by zero
        if p_correct == 0:
            p_correct = 1e-10
        if p_correct == 1:
            p_correct = 1 - 1e-10

        # Posterior probability of knowing given observation
        if correct == 1:
            # Correct answer
            p_known_given_correct = (p_known_before * (1 - p_s)) / p_correct
        else:
            # Incorrect answer
            p_known_given_correct = (p_known_before * p_s) / (1 - p_correct)

        # Probability of knowing after potential learning
        p_known_after = p_known_given_correct + (1 - p_known_given_correct) * p_t

        return p_known_after

    def fit_em(self,
               skill_sequences: List[List[int]],
               correct_sequences: List[List[int]],
               max_iterations: int = 25,
               tolerance: float = 1e-4) -> Dict[str, list]:
        """
        Fit BKT parameters using Expectation-Maximization algorithm.

        Args:
            skill_sequences: List of skill sequences for each student
            correct_sequences: List of correctness sequences for each student
            max_iterations: Maximum EM iterations
            tolerance: Convergence tolerance

        Returns:
            Dictionary with parameter history for monitoring convergence
        """
        logger.info(f"Starting EM fitting for {len(skill_sequences)} students")

        # Reset observation counts for fitting
        self.skill_observation_counts = np.zeros(self.total_indices)

        # Count observations per skill during fitting
        for skill_seq in skill_sequences:
            for skill_id in skill_seq:
                if 1 <= skill_id < self.total_indices:
                    self.skill_observation_counts[skill_id] += 1

        # Track parameter history for convergence monitoring
        param_history = {
            'p_l0': [self.p_l0.copy()],
            'p_t': [self.p_t.copy()],
            'p_g': [self.p_g.copy()],
            'p_s': [self.p_s.copy()],
            'log_likelihood': []
        }

        prev_log_likelihood = float('-inf')

        for iteration in range(max_iterations):
            # E-step: Compute expected sufficient statistics
            expected_stats = self._e_step(skill_sequences, correct_sequences)

            # M-step: Update parameters
            self._m_step(expected_stats)

            # Compute log-likelihood for convergence check
            log_likelihood = self._compute_log_likelihood(skill_sequences, correct_sequences)
            param_history['log_likelihood'].append(log_likelihood)

            # Store current parameters
            param_history['p_l0'].append(self.p_l0.copy())
            param_history['p_t'].append(self.p_t.copy())
            param_history['p_g'].append(self.p_g.copy())
            param_history['p_s'].append(self.p_s.copy())

            # Check convergence
            if iteration > 0:
                improvement = log_likelihood - prev_log_likelihood
                if abs(improvement) < tolerance:
                    logger.info(f"EM converged at iteration {iteration+1} "
                              f"(improvement: {improvement:.6f})")
                    break

            prev_log_likelihood = log_likelihood
            if iteration % 5 == 0:
                logger.info(f"EM iteration {iteration+1}, log-likelihood: {log_likelihood:.4f}")

        logger.info(f"EM fitting completed after {iteration+1} iterations")
        return param_history

    def _e_step(self,
                skill_sequences: List[List[int]],
                correct_sequences: List[List[int]]) -> Dict[str, np.ndarray]:
        """
        E-step: Compute expected sufficient statistics.

        Returns:
            Dictionary with expected counts for each parameter update
        """
        # Initialize expected statistics
        # We use total_indices (num_skills + 1) to account for padding index 0
        # Index 0 is for padding and will not be used for actual skills
        expected = {
            'known_init': np.zeros(self.total_indices),           # Expected known initially
            'known_trans': np.zeros(self.total_indices),          # Expected known->known transitions
            'unknown_trans': np.zeros(self.total_indices),        # Expected unknown->known transitions
            'known_correct': np.zeros(self.total_indices),        # Known and correct
            'known_incorrect': np.zeros(self.total_indices),      # Known and incorrect
            'unknown_correct': np.zeros(self.total_indices),      # Unknown and correct
            'unknown_incorrect': np.zeros(self.total_indices),    # Unknown and incorrect
            'total_obs': np.zeros(self.total_indices)             # Total observations per skill
        }

        # Process each student's sequence
        for skill_seq, correct_seq in zip(skill_sequences, correct_sequences):
            if not skill_seq:
                continue

            # Forward-backward algorithm to compute posteriors
            T = len(skill_seq)

            # Initialize forward and backward variables
            alpha = np.zeros((T, 2))  # [not_known, known]
            beta = np.zeros((T, 2))

            # Forward pass
            for t in range(T):
                skill_id = skill_seq[t]
                correct = correct_seq[t]
                p_l0, p_t, p_g, p_s = self._get_skill_parameters(skill_id)

                if t == 0:
                    # Initialization
                    alpha[0, 0] = 1 - p_l0  # P(not known at t=0)
                    alpha[0, 1] = p_l0      # P(known at t=0)
                else:
                    # Induction
                    # Probability of being known at t-1
                    p_known_prev = alpha[t-1, 1]
                    p_unknown_prev = alpha[t-1, 0]

                    # Transition probabilities
                    p_known_to_known = p_known_prev * (1 - p_s) * (1 - p_t) + \
                                       p_known_prev * p_s * p_t
                    p_known_to_unknown = p_known_prev * (1 - p_s) * p_t + \
                                         p_known_prev * p_s * (1 - p_t)
                    p_unknown_to_known = p_unknown_prev * p_g * (1 - p_t) + \
                                         p_unknown_prev * (1 - p_g) * p_t
                    p_unknown_to_unknown = p_unknown_prev * p_g * p_t + \
                                           p_unknown_prev * (1 - p_g) * (1 - p_t)

                    # Normalize
                    alpha[t, 0] = p_known_to_unknown + p_unknown_to_unknown
                    alpha[t, 1] = p_known_to_known + p_unknown_to_known

                # Apply observation (emission probability)
                if correct == 1:
                    emission_known = 1 - p_s
                    emission_unknown = p_g
                else:
                    emission_known = p_s
                    emission_unknown = 1 - p_g

                alpha[t, 0] *= emission_unknown
                alpha[t, 1] *= emission_known

                # Normalize to prevent underflow
                total = alpha[t, 0] + alpha[t, 1]
                if total > 0:
                    alpha[t, 0] /= total
                    alpha[t, 1] /= total

            # Backward pass
            beta[T-1, 0] = 1.0
            beta[T-1, 1] = 1.0

            for t in range(T-2, -1, -1):
                skill_id = skill_seq[t+1]
                correct = correct_seq[t+1]
                p_l0, p_t, p_g, p_s = self._get_skill_parameters(skill_id)

                # Compute beta[t] from beta[t+1]
                beta_sum_known = 0.0
                beta_sum_unknown = 0.0

                # From known state at t
                for prev_state in [0, 1]:  # 0 = not known, 1 = known
                    p_prev = alpha[t, prev_state]
                    if prev_state == 0:  # Was not known
                        p_to_known = p_g * (1 - p_t) + (1 - p_g) * p_t
                        p_to_unknown = p_g * p_t + (1 - p_g) * (1 - p_t)
                    else:  # Was known
                        p_to_known = (1 - p_s) * (1 - p_t) + p_s * p_t
                        p_to_unknown = (1 - p_s) * p_t + p_s * (1 - p_t)

                    # Emission at t+1
                    if correct_seq[t+1] == 1:
                        emission = 1 - p_s if prev_state == 1 else p_g
                    else:
                        emission = p_s if prev_state == 1 else 1 - p_g

                    contribution = p_prev * emission * beta[t+1, prev_state]
                    if prev_state == 0:
                        beta_sum_unknown += contribution
                    else:
                        beta_sum_known += contribution

                # Normalize
                total_beta = beta_sum_known + beta_sum_unknown
                if total_beta > 0:
                    beta[t, 0] = beta_sum_unknown / total_beta
                    beta[t, 1] = beta_sum_known / total_beta
                else:
                    beta[t, 0] = 0.5
                    beta[t, 1] = 0.5

            # Compute gamma (posterior probabilities)
            gamma = alpha * beta
            # Normalize gamma
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum[gamma_sum == 0] = 1  # Avoid division by zero
            gamma = gamma / gamma_sum

            # Compute expected statistics from gamma
            for t in range(T):
                skill_id = skill_seq[t]
                correct = correct_seq[t]

                # Expected probability of being known/unknown at time t
                p_known = gamma[t, 1]
                p_unknown = gamma[t, 0]

                expected['total_obs'][skill_id] += 1

                if t == 0:
                    # Initial knowledge state
                    expected['known_init'][skill_id] += p_known
                else:
                    # Transitions from t-1 to t
                    # Need xi (joint probability of states at t-1 and t)
                    # Simplified approximation using gamma
                    prev_p_known = gamma[t-1, 1]
                    prev_p_unknown = gamma[t-1, 0]

                    if correct == 1:
                        expected['known_correct'][skill_id] += p_known
                        expected['unknown_correct'][skill_id] += p_unknown
                    else:
                        expected['known_incorrect'][skill_id] += p_known
                        expected['unknown_incorrect'][skill_id] += p_unknown

                    # Approximate transitions (simplified)
                    expected['known_trans'][skill_id] += p_known * prev_p_known
                    expected['unknown_trans'][skill_id] += p_unknown * (1 - prev_p_known)

        return expected

    def _m_step(self, expected_stats: Dict[str, np.ndarray]):
        """
        M-step: Update parameters based on expected sufficient statistics.
        """
        # Avoid division by zero
        eps = 1e-10

        # Update P(L0): probability of knowing initially
        # We exclude index 0 (padding) from the calculation
        # Create a mask for valid skill indices (1 to num_skills)
        valid_indices = slice(1, self.total_indices)  # indices 1 through num_skills

        total_initial = np.sum(expected_stats['known_init'][valid_indices]) + \
                       np.sum((self.num_skills - expected_stats['known_init'][valid_indices]))  # unknown initially
        if total_initial > 0:
            self.p_l0[valid_indices] = expected_stats['known_init'][valid_indices] / (
                expected_stats['known_init'][valid_indices] +
                (self.num_skills - expected_stats['known_init'][valid_indices]) + eps
            )
        else:
            # Fallback to uniform if no data (for valid indices only)
            self.p_l0[valid_indices] = np.full(self.num_skills, 0.5)

        # Update P(T): probability of learning
        total_trans_opportunities = np.sum(expected_stats['known_trans'][valid_indices]) + \
                                   np.sum(expected_stats['unknown_trans'][valid_indices]) + eps
        if total_trans_opportunities > 0:
            self.p_t[valid_indices] = np.sum(expected_stats['unknown_trans'][valid_indices]) / total_trans_opportunities
        else:
            self.p_t[valid_indices] = np.full(self.num_skills, 0.3)

        # Update P(G): probability of guessing
        total_unknown_opportunities = np.sum(expected_stats['unknown_correct'][valid_indices]) + \
                                     np.sum(expected_stats['unknown_incorrect'][valid_indices]) + eps
        if total_unknown_opportunities > 0:
            self.p_g[valid_indices] = np.sum(expected_stats['unknown_correct'][valid_indices]) / total_unknown_opportunities
        else:
            self.p_g[valid_indices] = np.full(self.num_skills, 0.2)

        # Update P(S): probability of slipping
        total_known_opportunities = np.sum(expected_stats['known_correct'][valid_indices]) + \
                                   np.sum(expected_stats['known_incorrect'][valid_indices]) + eps
        if total_known_opportunities > 0:
            self.p_s[valid_indices] = np.sum(expected_stats['known_incorrect'][valid_indices]) / total_known_opportunities
        else:
            self.p_s[valid_indices] = np.full(self.num_skills, 0.1)

        # Ensure parameters are in valid range [0, 0.999] (keeping some margin from 0 and 1)
        # We only clip the valid indices (1 through num_skills), leaving index 0 unchanged
        self.p_l0[valid_indices] = np.clip(self.p_l0[valid_indices], 0.001, 0.999)
        self.p_t[valid_indices] = np.clip(self.p_t[valid_indices], 0.001, 0.999)
        self.p_g[valid_indices] = np.clip(self.p_g[valid_indices], 0.001, 0.999)
        self.p_s[valid_indices] = np.clip(self.p_s[valid_indices], 0.001, 0.999)

    def _compute_log_likelihood(self,
                                skill_sequences: List[List[int]],
                                correct_sequences: List[List[int]]) -> float:
        """
        Compute log-likelihood of the data given current parameters.
        """
        total_log_likelihood = 0.0

        for skill_seq, correct_seq in zip(skill_sequences, correct_sequences):
            if not skill_seq:
                continue

            # Compute likelihood for this sequence using forward algorithm
            T = len(skill_seq)
            alpha = np.zeros((T, 2))  # [not_known, known]

            for t in range(T):
                skill_id = skill_seq[t]
                correct = correct_seq[t]
                p_l0, p_t, p_g, p_s = self._get_skill_parameters(skill_id)

                if t == 0:
                    # Initialization
                    alpha[0, 0] = (1 - p_l0) * (p_g if correct == 1 else (1 - p_g))
                    alpha[0, 1] = p_l0 * ((1 - p_s) if correct == 1 else p_s)
                else:
                    # Induction
                    # Transition probabilities
                    p_kk = (1 - p_s) * (1 - p_t) + p_s * p_t  # known -> known
                    p_ku = (1 - p_s) * p_t + p_s * (1 - p_t)   # known -> unknown
                    p_uk = p_g * (1 - p_t) + (1 - p_g) * p_t   # unknown -> known
                    p_uu = p_g * p_t + (1 - p_g) * (1 - p_t)   # unknown -> unknown

                    alpha[t, 0] = (alpha[t-1, 0] * p_uu + alpha[t-1, 1] * p_ku) * \
                                 (p_g if correct == 1 else (1 - p_g))
                    alpha[t, 1] = (alpha[t-1, 0] * p_uk + alpha[t-1, 1] * p_kk) * \
                                 ((1 - p_s) if correct == 1 else p_s)

            # Sequence likelihood is sum of final state probabilities
            seq_likelihood = alpha[T-1, 0] + alpha[T-1, 1]
            if seq_likelihood > 0:
                total_log_likelihood += np.log(seq_likelihood)
            else:
                total_log_likelihood += np.log(1e-10)  # Very small probability

        return total_log_likelihood

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get current parameter values.

        Returns:
            Dictionary containing p_l0, p_t, p_g, p_s arrays
        """
        return {
            'p_l0': self.p_l0.copy(),
            'p_t': self.p_t.copy(),
            'p_g': self.p_g.copy(),
            'p_s': self.p_s.copy()
        }

    def set_parameters(self, p_l0: np.ndarray, p_t: np.ndarray,
                      p_g: np.ndarray, p_s: np.ndarray):
        """
        Set parameter values.

        Args:
            p_l0: Array of P(L0) values
            p_t: Array of P(T) values
            p_g: Array of P(G) values
            p_s: Array of P(S) values
        """
        self.p_l0 = np.clip(p_l0, 0.001, 0.999)
        self.p_t = np.clip(p_t, 0.001, 0.999)
        self.p_g = np.clip(p_g, 0.001, 0.999)
        self.p_s = np.clip(p_s, 0.001, 0.999)


def save_bkt_params(params: Dict[str, np.ndarray], filepath: Union[str, Path]):
    """
    Save BKT parameters to JSON file.

    Args:
        params: Dictionary containing parameter arrays
        filepath: Path to save the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable_params = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            serializable_params[key] = value.tolist()
        else:
            serializable_params[key] = value

    with open(filepath, 'w') as f:
        json.dump(serializable_params, f, indent=2)

    logger.info(f"BKT parameters saved to {filepath}")


def load_bkt_params(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load BKT parameters from JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary containing parameter arrays
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"BKT parameters file not found: {filepath}")

    with open(filepath, 'r') as f:
        params = json.load(f)

    # Convert lists back to numpy arrays
    for key, value in params.items():
        if isinstance(value, list):
            params[key] = np.array(value)

    logger.info(f"BKT parameters loaded from {filepath}")
    return params


if __name__ == "__main__":
    # Simple test when run directly
    print("Testing BKT model...")

    # Create a simple test case
    num_skills = 5
    bkt = BKT(num_skills=num_skills)

    # Test prediction
    skill_seq = [0, 1, 2, 1, 0]
    correct_seq = [1, 0, 1, 1, 0]

    predictions = bkt.predict(skill_seq, correct_seq)
    print(f"Predictions: {predictions}")

    # Test parameter getting/setting
    params = bkt.get_parameters()
    print(f"Number of skills: {len(params['p_l0'])}")
    print(f"P(L0) range: [{params['p_l0'].min():.3f}, {params['p_l0'].max():.3f}]")
    print(f"P(T) range: [{params['p_t'].min():.3f}, {params['p_t'].max():.3f}]")
    print(f"P(G) range: [{params['p_g'].min():.3f}, {params['p_g'].max():.3f}]")
    print(f"P(S) range: [{params['p_s'].min():.3f}, {params['p_s'].max():.3f}]")

    print("BKT model test completed successfully!")