from argparse import ArgumentParser
from nlp import runner, hyper, wrapper, metric
import wandb
from functools import partial


logger_choices = ["wandb", "print", "nologger"]
base_args = {
    "--logger": dict(
        type=str,
        choices=logger_choices,
        required=True,
        help=f"Logger to use (choices: {', '.join(logger_choices)})",
    ),
    "--early_patience": dict(
        type=int,
        default=3,
        help="Patience for early stopping (default: 3)",
    ),
    "--val_epoch": dict(
        type=int,
        default=1,
        help="Number of epochs for validation (default: 1)",
    ),
    "--max_epoch": dict(
        type=int,
        default=10,
        help="Maximum number of training epochs (default: 10)",
    ),
    "--rank": dict(
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    ),
    "--lora_alpha": dict(
        type=float,
        default=32,
        help="LoRA alpha (default: 32)",
    ),
    "--lora_dropout": dict(
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)",
    ),
    "--seed": dict(
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    ),
    "--lang": dict(
        type=str,
        required=True,
        help="Language to train the model on",
    ),
    "--full": dict(
        action="store_true",
        help="Perform a fine-tuning on the whole model",
    ),
    "--train_languages": dict(
        type=str,
        nargs="+",
        required=True,
        help="List of languages to train the model on",
    ),
    "--test_languages": dict(
        type=str,
        nargs="+",
        required=True,
        help="List of languages to evaluate the model on",
    ),
    "--model_path": dict(
        type=str,
        default=None,
        help="Model path",
    ),
}

baseline_args = [
    "logger",
    "early_patience",
    "val_epoch",
    "max_epoch",
    "rank",
    "lora_alpha",
    "lora_dropout",
    "seed",
    "lang",
    "full",
]
relatedness_args = [
    "logger",
    "early_patience",
    "val_epoch",
    "max_epoch",
    "rank",
    "lora_alpha",
    "lora_dropout",
    "seed",
    "train_languages",
    "test_languages",
]
evaluation_args = [
    "logger",
    "rank",
    "train_languages",
    "test_languages",
    "model_path",
]


def with_default(arguments, flag):
    return getattr(
        arguments, flag, base_args["--" + flag].get("default", None)
    )


def make_hyper(args):
    full = args.full if args.command == "baseline" else False
    with_d = partial(with_default, args)

    conf = hyper.HyperRelatedness(
        logger=args.logger,
        early_patience=with_d("early_patience"),
        early_invert=True,
        val_metrics=[
            "accuracy",
            "tr_chrf",
            "hamming_dist",
            "similarity",
        ],
        val_epoch=with_d("val_epoch"),
        max_epoch=with_d("max_epoch"),
        train_languages=with_d("train_languages"),
        test_languages=with_d("test_languages"),
        seed=with_d("seed"),
        r=with_d("rank"),
        lora_alpha=with_d("lora_alpha"),
        lora_dropout=with_d("lora_dropout"),
        with_peft=not full,
    )

    print(conf.dict())
    # Configure logger based on arguments
    logger = None
    if args.logger == "wandb":
        run = wandb.init(project="test_nlp", config=conf.dict())
        logger = run
    elif args.logger == "print":
        logger = wrapper.Print_Logger

    conf.logger = logger
    conf.val_metrics = [
        metric.accuracy,
        metric.create_chrf(),
        metric.hamming_dist,
        metric.similarity,
    ]
    return conf


def run_cli():
    parser = ArgumentParser(description="Global command-line tool")
    subparsers = parser.add_subparsers(
        title="Available commands", dest="command"
    )

    # Create subparsers for each command
    baseline_parser = subparsers.add_parser(
        "baseline", help="Run baseline training"
    )
    relatedness_parser = subparsers.add_parser(
        "related", help="Run relatedness experiments"
    )

    evaluation_parser = subparsers.add_parser(
        "evaluate", help="Run evaluation of a model"
    )

    command_functions = {
        "baseline": runner.run_baseline,
        "related": runner.run_relatedness,
        "evaluate": runner.evaluate_model,
    }

    for k, v in base_args.items():
        arg = k[2:]
        if arg in baseline_args:
            baseline_parser.add_argument(k, **v)
        if arg in relatedness_args:
            relatedness_parser.add_argument(k, **v)
        if arg in evaluation_args:
            evaluation_parser.add_argument(k, **v)

    args = parser.parse_args()
    if args.command == "baseline":
        args.train_languages = [args.lang]
        args.test_languages = [args.lang]

    if fun := command_functions.get(args.command, None):
        conf = make_hyper(args)
        if args.command == "evaluate":
            fun(conf, args.model_path)
        else:
            fun(conf)

        # Close W&B run if used
        if args.logger == "wandb":
            wandb.finish()
        exit(0)

    parser.print_help()
