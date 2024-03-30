from nlp.hyper import HyperRelatedness
from nlp import wrapper, data
from peft import TaskType
from pathlib import Path


def make_wrapper(conf: HyperRelatedness):
    return wrapper.Wrapper(
        "google/byt5-small",
        logger=conf.logger,
        val_epoch=conf.val_epoch,
        epoch=conf.max_epoch,
        val_metrics=conf.val_metrics,
        early_stopping_kwargs=dict(
            patience=conf.early_patience, invert=conf.early_invert
        ),
        peft_config_kwargs=dict(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=conf.r,
            lora_alpha=conf.lora_alpha,
            lora_dropout=conf.lora_dropout,
        ),
    )


def run_baseline(conf: HyperRelatedness):
    """Trains a model on a single language for baseline results."""

    wmodel = make_wrapper(conf)

    lang = conf.train_languages[0]

    datasets = data.load_data(lang, wmodel.tokenizer)

    name_suffix = f"BASELINE_{conf.r}"

    wmodel.train(
        datasets["train"],
        datasets["val"],
        log_prefix=lang + "_",
    )

    # Evaluate on the training language's test set
    wmodel.evaluate(
        datasets["test"],
        log_prefix="test_" + lang,
        save_path=Path("results") / wmodel.get_save_path([lang], name_suffix),
    )

    wmodel.save_model([lang], name_suffix)


def train_model(conf, wmodel):
    name_suffix = f"BASELINE_{conf.r}"
    for lang in conf.train_languages:
        dataset = data.load_data(lang, wmodel.tokenizer)
        wmodel.train(
            dataset["train"],
            dataset["val"],
            log_prefix=lang + "_",
        )

        wmodel.evaluate(
            dataset["test"],
            log_prefix="test_" + lang,
            save_path=Path("results")
            / wmodel.get_save_path([lang], name_suffix),
        )
    wmodel.save_model(conf.train_languages, name_suffix)


def run_relatedness(conf: HyperRelatedness):
    """Trains the model initially on train_languages and then performs zero-shot evaluation
    on test languages. It then incrementally fine-tunes on samples from test language
    datasets.
    """

    if conf.train_languages == conf.test_languages:
        print("PRETRAINED IS THE SAME AS TESTING SKIPPING")
        return

    # Create a Wrapper object
    wmodel = make_wrapper(conf)

    name_suffix = f"BASELINE_{conf.r}"

    base_path = Path("models") / wmodel.get_save_path(
        conf.train_languages, name_suffix
    )
    if not base_path.is_dir():
        print(f"PRETRAIN {base_path} NOT PRESENT")
        print(f"START TRAINING {base_path}")
        train_model(conf, wmodel)

    # YEs or No ?
    # wmodel.epoch = 1

    print(f"LOAD PRETRAINED {base_path}")
    wmodel.load_model(conf.train_languages, name_suffix)

    datasets = {
        lang: data.load_data(lang, wmodel.tokenizer)
        for lang in conf.test_languages
    }
    # Zero-Shot Evaluation and Incremental Fine-Tuning
    sample_sizes = [0, 5, 50, 200, 1000]  # Sample sizes for fine-tuning
    train_langs = f"{'_'.join(conf.train_languages)}_{conf.r}"
    for lang in conf.test_languages:
        # Zero-shot
        wmodel.evaluate(
            datasets[lang]["test"],
            log_prefix=f"test_{lang}_{train_langs}_zero_shot",
            save_path=Path("results")
            / wmodel.get_save_path(
                conf.train_languages, f"{lang}{conf.r}_zero_shot"
            ),
        )

        # Fine-tuning with Increasing Sample Sizes
        for sample_size in sample_sizes[1:]:
            log_prefix = f"{lang}_{train_langs}_sample{sample_size}_"
            save_suffix = f"{lang}{conf.r}_sample{sample_size}"

            # Reload the initial model (trained on train_languages)
            wmodel.load_model(conf.train_languages, name_suffix)

            # Sample dataset
            sampled_dataset = data.sample_dataset(
                datasets[lang]["train"], sample_size, seed=conf.seed
            )

            # Fine-tune on the sampled dataset
            wmodel.train(
                sampled_dataset,
                datasets[lang]["val"],  # Might want to sample val as well
                log_prefix=log_prefix,
            )

            # Evaluate on the test set
            wmodel.evaluate(
                datasets[lang]["test"],
                log_prefix="test_" + log_prefix,
                save_path=Path("results")
                / wmodel.get_save_path(conf.train_languages, save_suffix),
            )
            wmodel.save_model(conf.train_languages, save_suffix)
