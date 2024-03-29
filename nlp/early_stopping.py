from dataclasses import dataclass, field
from typing import Callable


@dataclass
class EarlyStopping:
    """Implements an early stopping mechanism for model training."""

    patience: int = 4  # Number of epochs without improvement before stopping
    delta: float = 0.0  # Minimum improvement to reset the counter
    invert: bool = False  # Whether to invert the metric (useful for loss)
    debug: bool = False  # Flag to enable debug prints

    # Internal state (initialized after object creation)
    cnt: int = field(init=False, default_factory=lambda: 0)
    min_val: float = field(init=False, default_factory=lambda: float("inf"))

    def __post_init__(self):
        """Initializes the debug function based on the 'debug' flag."""
        self.debug: Callable[..., None] = (
            print if self.debug else lambda *args: None
        )

    def __call__(self, metric: float) -> tuple[bool, bool]:
        """Evaluates whether to stop training based on the provided metric.

        Args:
            metric: The current value of the metric being tracked.

        Returns:
            tuple[bool, bool]:
                * True for early stopping, False otherwise.
                * True to reset best model, False otherwise.
        """

        metric = -metric if self.invert else metric

        # Flag to indicate if the best model should be reset
        reset_model = False
        if metric < self.min_val:
            self.min_val = metric
            self.cnt = 0
            self.debug("Reset")
            reset_model = True

        elif metric >= (self.min_val + self.delta):
            self.debug("Increase")
            self.cnt += 1

        return self.cnt >= self.patience, reset_model
