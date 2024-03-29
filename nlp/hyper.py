from dataclasses import dataclass, asdict
from typing import List


@dataclass
class HyperRelatedness:
    """Represents a collection of hyperparameters for relatedness
    evaluation experiments.
    """

    logger: str
    # The name of the logger to use (e.g., 'wandb', 'print').

    early_patience: int
    # Number of epochs with no validation improvement before stopping.

    early_invert: bool
    # Whether to invert the criterion for early stopping (useful for loss).

    val_metrics: List[str]
    # List of metrics to track on the validation set.

    val_epoch: int
    # How often to perform validation (in epochs).

    max_epoch: int
    # The maximum number of training epochs.

    train_languages: List[str]
    # Languages to train the model on.

    test_languages: List[str]
    # Languages to evaluate the model on.

    seed: int = 42
    # Random seed for reproducibility.

    r: int = 8  # LoRA rank.
    lora_alpha: int = 32  # LoRA scaling factor.
    lora_dropout: float = 0.1  # LoRA dropout.

    dict = asdict
    # Method to convert the dataclass into a dictionary.
