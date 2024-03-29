from datasets import load_dataset
from typing import Union, Callable, Any, Dict
from pathlib import Path
from transformers import default_data_collator
from torch.utils.data import DataLoader, Subset
from functools import partial
import random


def load_lang_data(lang: str, base_path: Union[str, Path] = Path("datasets")):
    if isinstance(base_path, str):
        base_path = Path(base_path)

    gen_path = lambda mode: str(base_path / (lang + "_" + mode + ".json.gz"))
    return load_dataset(
        "json",
        data_files={k: gen_path(k) for k in ["train", "test", "val"]},
        field="data",
    )


def pre_tok(
    data: dict,
    tok: object,
    max_length: int,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt",
    mask_padding: bool = True,
):
    inputs = [
        lem + "\t" + feat for lem, feat in zip(data["lemma"], data["features"])
    ]
    targets = data["inflected"]
    base_tok_args = dict(
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    model_inputs = tok(inputs, **base_tok_args)
    labels = tok(targets, **base_tok_args)["input_ids"]
    if mask_padding:
        # This is done to ignore these position during the loss
        # ensure that the model learns from the important parts
        labels[labels == tok.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def process_dataset(
    dataset: object,
    proc_fun: Callable[[Dict[str, Any]], Dict[str, Any]],
    rm_column_names: list,
    batched: bool = True,
    load_from_cache_file: bool = False,
    desc: str = "Processing dataset",
):
    return dataset.map(
        proc_fun,
        batched=True,
        remove_columns=rm_column_names,
        load_from_cache_file=False,
        desc=desc,
    )


def get_dataloader(
    dataset: object,
    batch_size: int = 16,
    collate_fn: callable = default_data_collator,
    pin_memory: bool = False,
    shuffle: bool = True,
    **data_loader_kwargs,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **data_loader_kwargs,
    )


def load_data(lang: str, tok: object):
    _dataset = load_lang_data(lang)

    # Note we do this because the training data needs a special treatment:
    # the padding values added by the tokenizer needs to be change to a
    # negative value to ensure that the model learn only from the important parts.
    # See comment in the function data.pre_tok for the mask_padding variable
    _train_dataset = _dataset.pop("train")

    dataset = process_dataset(
        _dataset,
        partial(pre_tok, tok=tok, max_length=128, mask_padding=False),
        _dataset["test"].column_names,
    )

    train_dataset = process_dataset(
        _train_dataset,
        partial(pre_tok, tok=tok, max_length=128),
        _dataset["test"].column_names,
    )
    return {
        "train": get_dataloader(train_dataset),
        "test": get_dataloader(dataset["test"]),
        "val": get_dataloader(dataset["val"]),
    }


def sample_dataset(dataloader, sample_size: int, seed: int = 42):
    """Samples a subset of a dataloader (a PyTorch DataLoader)."""
    return get_dataloader(
        dataloader.dataset.shuffle(seed=seed).select(range(sample_size))
    )
