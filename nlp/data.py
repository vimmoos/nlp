from datasets import load_dataset
from typing import Union, Callable, Any, Dict
from pathlib import Path
from transformers import default_data_collator
from torch.utils.data import DataLoader
from functools import partial


def load_lang_data(
    lang: str,
    base_path: Union[str, Path] = Path("datasets"),
):
    """Loads language-specific data from JSON files.

    Args:
        lang: The language code (e.g., 'eng', 'deu').
        base_path: The base directory to search for data files. Defaults
            to 'datasets'.

    Returns:
        Dataset: A Hugging Face Dataset object containing the loaded data.
    """

    base_path = Path(base_path)  # Convert string paths to Path objects

    gen_path = lambda mode: str(  # Lambda to generate file paths
        base_path / (lang + "_" + mode + ".json.gz")
    )

    return load_dataset(
        "json",
        data_files={k: gen_path(k) for k in ["train", "test", "val"]},
        field="data",
    )


def pre_tok(
    data: dict,
    tok: object,  # Assuming 'tok' is a tokenizer object
    max_length: int,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt",
    mask_padding: bool = True,
) -> dict:
    """Prepares data for tokenization and optionally masks padding tokens.

    Args:
        data: A dictionary containing 'lemma', 'features', and 'inflected' lists.
        tok: A tokenizer object.
        max_length: Maximum sequence length for tokenization.
        padding: Padding strategy ('max_length' is the default).
        truncation: Whether to truncate sequences longer than 'max_length'.
        return_tensors: Format of the output ('pt' for PyTorch tensors).
        mask_padding: Whether to mask padding tokens in the labels.

    Returns:
        dict: A dictionary containing tokenized inputs and labels, ready for a model.
    """

    inputs = [
        lem + "\t" + feat for lem, feat in zip(data["lemma"], data["features"])
    ]  # Combine lemma and features

    targets = data["inflected"]

    base_tok_args = dict(
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )

    model_inputs = tok(inputs, **base_tok_args)  # Tokenize inputs

    labels = tok(targets, **base_tok_args)["input_ids"]  # Tokenize labels

    if mask_padding:
        labels[labels == tok.pad_token_id] = -100  # Mask padding tokens

    model_inputs["labels"] = labels
    return model_inputs


def process_dataset(
    dataset: object,  # Assuming 'dataset' is a Hugging Face Dataset
    proc_fun: Callable[[Dict[str, Any]], Dict[str, Any]],
    rm_column_names: list,
    batched: bool = True,
    load_from_cache_file: bool = False,
    desc: str = "Processing dataset",
) -> object:  # Likely returns a Dataset object
    """Applies a processing function to a Hugging Face Dataset.

    Args:
        dataset: The input Hugging Face Dataset object.
        proc_fun: A callable function that takes a single data example (dictionary)
            and returns a processed dictionary.
        rm_column_names: A list of column names to remove after processing.
        batched: Whether to process in batches for efficiency.
        load_from_cache_file: Whether to load from a cached version (if available).
        desc: A description for the progress bar.

    Returns:
        The processed Dataset object.
    """

    return dataset.map(
        proc_fun,
        batched=batched,
        remove_columns=rm_column_names,
        load_from_cache_file=load_from_cache_file,
        desc=desc,
    )


def get_dataloader(
    dataset: object,  # Assuming a Hugging Face Dataset or PyTorch compatible dataset
    batch_size: int = 16,
    collate_fn: Callable[
        [Any], Any
    ] = default_data_collator,  # Assumes default_data_collator exists
    pin_memory: bool = False,
    shuffle: bool = True,
    **data_loader_kwargs,
) -> DataLoader:
    """Creates a DataLoader with sensible defaults for typical machine learning tasks.

    Args:
        dataset: The input dataset.
        batch_size: The batch size for the DataLoader.
        collate_fn: A function to merge a list of samples into a mini-batch.
        pin_memory: Whether to use pinned memory for faster data transfer to GPU.
        shuffle: Whether to shuffle the data.
        **data_loader_kwargs: Additional arguments for the DataLoader constructor.

    Returns:
        DataLoader: A DataLoader object ready for use in training or evaluation.
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **data_loader_kwargs,
    )


def load_data(lang: str, tok: object) -> Dict[str, DataLoader]:
    """Loads language data, applies preprocessing, and creates DataLoaders.

    Args:
        lang: The language code.
        tok: A tokenizer object.

    Returns:
        Dict[str, DataLoader]: A dictionary containing 'train', 'test', and 'val' DataLoaders.
    """

    _dataset = load_lang_data(lang)  # Load initial dataset

    _train_dataset = _dataset.pop("train")  # Separate train data

    # Process 'test' and 'val' data (no padding masking needed)
    dataset = process_dataset(
        _dataset,
        partial(pre_tok, tok=tok, max_length=128, mask_padding=False),
        _dataset["test"].column_names,
    )

    # Process 'train' data (with padding masking)
    train_dataset = process_dataset(
        _train_dataset,
        partial(
            pre_tok, tok=tok, max_length=128
        ),  # Note: mask_padding=True (default)
        _dataset["test"].column_names,
    )

    # Create DataLoaders
    return {
        "train": get_dataloader(train_dataset),
        "test": get_dataloader(dataset["test"]),
        "val": get_dataloader(dataset["val"]),
    }


def sample_dataset(
    dataloader: DataLoader, sample_size: int, seed: int = 42
) -> DataLoader:
    """Samples a subset of a PyTorch DataLoader.

    Args:
        dataloader: The input DataLoader.
        sample_size: The desired number of samples in the subset.
        seed: Random seed for reproducibility.

    Returns:
        DataLoader: A new DataLoader containing a random subset of the original data.
    """
    return get_dataloader(
        dataloader.dataset.shuffle(seed=seed).select(range(sample_size))
    )
