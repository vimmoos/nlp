from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    patience: int = 4
    delta: float = 0.0
    cnt: int = field(init=False, default_factory=lambda: 0)
    min_val: float = field(init=False, default_factory=lambda: float("inf"))
    invert: bool = False
    debug: bool = False

    def __post_init__(self):
        self.debug = (
            (lambda *args: print(*args)) if self.debug else (lambda *args: None)
        )

    def __call__(self, metric):
        metric = -metric if self.invert else metric
        if metric < self.min_val:
            self.min_val = metric
            self.cnt = 0
            self.debug("Reset")
            return False

        if metric >= (self.min_val + self.delta):
            self.debug("Increase")
            self.cnt += 1

        return self.cnt >= self.patience
