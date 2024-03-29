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

    dict = asdict
