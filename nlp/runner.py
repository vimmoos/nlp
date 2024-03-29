from nlp.hyper import HyperRelatedness
from nlp import wrapper, data, metric
from functools import partial


def load_dataset(lang: str, tok: object):
    _dataset = data.load_lang_data(lang)

    # Note we do this because the training data needs a special treatment:
    # the padding values added by the tokenizer needs to be change to a
    # negative value to ensure that the model learn only from the important parts.
    # See comment in the function data.pre_tok for the mask_padding variable
    _train_dataset = _dataset.pop("train")

    dataset = data.process_dataset(
        _dataset,
        partial(data.pre_tok, tok=tok, max_length=128, mask_padding=False),
        _dataset["test"].column_names,
    )

    train_dataset = data.process_dataset(
        _train_dataset,
        partial(data.pre_tok, tok=tok, max_length=128),
        _dataset["test"].column_names,
    )
    return {
        "train": data.get_dataloader(train_dataset, pin_memory=False),
        "test": data.get_dataloader(dataset["test"], pin_memory=False),
        "val": data.get_dataloader(dataset["val"], pin_memory=False),
    }


def run_relatedness(conf: HyperRelatedness):
    wmodel = wrapper.Wrapper(
        "google/byt5-small",
        logger=conf.logger,
        val_epoch=conf.val_epoch,
        epoch=conf.max_epoch,
        val_metrics=conf.val_metrics,
        early_stopping_kwargs=dict(
            patience=conf.early_patience, invert=conf.early_invert
        ),
    )

    datasets = {
        lang: load_dataset(lang, wmodel.tokenizer)
        for lang in set(conf.train_languages).union(set(conf.test_languages))
    }

    for lang in conf.train_languages:
        wmodel.train(
            datasets[lang]["train"],
            datasets[lang]["val"],
            log_prefix=lang + "_",
        )

    for lang in conf.test_languages:
        wmodel.evaluate(datasets[lang]["test"], log_prefix="test_" + lang)


if __name__ == "__main__":
    import wandb

    conf = HyperRelatedness(
        logger="wandb",
        early_patience=3,
        early_invert=True,
        val_metrics=[
            "tr_chrf",
            "chr_f",
            "hamming_dist",
            "similarity",
        ],
        val_epoch=1,
        max_epoch=100,
        train_languages=["tur", "eng"],
        test_languages=["eng"],
    )

    run = wandb.init(
        project="test_nlp",
        config=conf.dict(),
    )

    conf.logger = wandb
    conf.val_metrics = [
        metric.create_chrf(),
        metric.chr_f,
        metric.hamming_dist,
        metric.similarity,
    ]

    run_relatedness(conf)

    wandb.finish()
