from argparse import ArgumentParser
from nlp import runner, hyper, wrapper, metric
import wandb


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
}


def make_hyper(args):
    conf = hyper.HyperRelatedness(
        logger=args.logger,
        early_patience=args.early_patience,
        early_invert=True,
        val_metrics=[
            "accuracy",
            "tr_chrf",
            "hamming_dist",
            "similarity",
        ],
        val_epoch=args.val_epoch,
        max_epoch=args.max_epoch,
        train_languages=args.train_languages,
        test_languages=args.test_languages,
        seed=args.seed,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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

    command_functions = {
        "baseline": runner.run_baseline,
        "related": runner.run_relatedness,
    }

    for k, v in base_args.items():
        baseline_parser.add_argument(k, **v)
        relatedness_parser.add_argument(k, **v)

    baseline_parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language to train the model on",
    )

    relatedness_parser.add_argument(
        "--train_languages",
        type=str,
        nargs="+",
        required=True,
        help="List of languages to train the model on",
    )
    relatedness_parser.add_argument(
        "--test_languages",
        type=str,
        nargs="+",
        required=True,
        help="List of languages to evaluate the model on",
    )

    args = parser.parse_args()
    if args.command == "baseline":
        args.train_languages = [args.lang]
        args.test_languages = [args.lang]

    if fun := command_functions.get(args.command, None):
        conf = make_hyper(args)
        fun(conf)

        # Close W&B run if used
        if args.logger == "wandb":
            wandb.finish()
        exit(0)

    parser.print_help()
