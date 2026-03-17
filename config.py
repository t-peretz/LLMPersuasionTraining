"""Shared configuration defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Configuration:
    A_MAX_TOKENS: int = 512
    B_MAX_TOKENS: int = 256

    # Training hyperparameters
    A_BATCH_SIZE: int = 8
    B_BATCH_SIZE: int = 8
    A_STEPS_PER_B: int = 5
    A_NUM_GENERATIONS: int = 8
    MAX_B_TRAIN_TOKENS: int = 1024
    B_LEARNING_RATE: float = 1e-5
    GRAD_CLIP_NORM: float = 1.0

    # Sampling hyperparameters
    A_TEMPERATURE: float = 0.7
    A_TOP_P: float = 0.9
    B_TEMPERATURE: float = 0.7
    B_TOP_P: float = 0.95
    B_CLIP_EPS: float = 0.2

    # Reward
    B_CORRECT_REWARD: float = 1.0
    A_ACCURACY_REWARD: float = 0
    A_PERSUASION_REWARD: float = 1.0
    FORMAT_REWARD: float = 0.2
