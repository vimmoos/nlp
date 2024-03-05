from datasets import load_dataset
from typing import Union, Callable, Any, Dict
from pathlib import Path
from transformers import default_data_collator
from torch.utils.data import DataLoader


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
):
    inputs = [lem + "\t" + feat for lem, feat in zip(data["lemma"], data["features"])]
    targets = data["inflected"]
    base_tok_args = dict(
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    model_inputs = tok(inputs, **base_tok_args)
    model_inputs["labels"] = tok(targets, **base_tok_args)["input_ids"]
    return model_inputs


def process_dataset(
    dataset: object,
    proc_fun: Callable[[Dict[str, Any]], Dict[str, Any]],
    batched: bool = True,
    load_from_cache_file: bool = False,
    desc: str = "Processing dataset",
):
    return dataset.map(
        proc_fun,
        batched=True,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc=desc,
    )


def get_dataloader(
    dataset: object,
    batch_size: int = 16,
    collate_fn: callable = default_data_collator,
    pin_memory: bool = True,
    **data_loader_kwargs,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        pin_memory=pin_memory,
        **data_loader_kwargs,
    )
