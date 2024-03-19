from scipy.spatial.distance import hamming
from difflib import SequenceMatcher
from typing import Tuple


## Some base metrics that can be used for text comparison


def _adjust_len(pred: str, target: str) -> Tuple[str, str]:
    """Utility function which makes sure
    that pred and target have the same lenght.

    If not it either extend the pred or
    cut the pred based on the target.
    It never touches the target.
    """
    if len(pred) < len(target):
        pred = pred.ljust(len(target), " ")
    if len(pred) > len(target):
        pred = pred[: len(target)]
    return (pred, target)


def hamming_dist(pred: str, target: str) -> float:
    pred, target = _adjust_len(pred, target)
    return hamming(list(pred), list(target))


def similarity(pred: str, target: str) -> float:
    pred, target = _adjust_len(pred, target)
    return SequenceMatcher(None, pred, target).ratio()


def chr_f(pred: str, target: str, beta: float) -> float:
    """From: Maja Popovic(2015)."""
    blocks = SequenceMatcher(None, pred, target).get_matching_blocks()
    match_total = 0
    for block in blocks:
        match_total += block[2]
    p, r = match_total/len(pred), match_total/len(target)
    return (1 + beta*beta) * (p*r)/(beta*beta*p+r)

def accuracy(pred: str, target: str) -> float:
    return float(pred == target)