from dataclasses import dataclass, asdict
from typing import List, Callable


@dataclass
class HyperRelatedness:
    logger: str

    early_patience: int
    early_invert: bool

    val_metrics: List[str]

    val_epoch: int
    max_epoch: int

    train_languages: List[str]

    test_languages: List[str]
    seed: int = 42

    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    dict = asdict
