from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Union, Dict, Any, Optional, Callable, List
from pathlib import Path
import torch as th
from tqdm import tqdm, trange
from nlp.metric import hamming_dist, similarity
from nlp.early_stopping import EarlyStopping
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model

## Some base logger implementation


class No_Logger:
    def log(*args, **kwargs):
        pass


class Print_Logger:
    def log(*args, **_):
        print(*args)


@dataclass
class Wrapper:
    """Main wrapper class.

    Takes at least a model_name and provides an object with two main methods:
    + train
    + evaluate
    """

    model_name: Union[str, Path]

    # If true early stopping based on the val_metrics will be applied.
    # Note: only the first metric in val_metrics will be used.
    early_stopping: bool = True

    early_stopping_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            patience=3,
            invert=False,
        )
    )

    # If tokenizer_name is None than it defaults to the model_name
    tokenizer_name: Optional[Union[str, Path]] = None

    # Max number of epochs to train on
    epoch: int = 10

    # Number of train epochs before validating
    # So if set to 2 the model will be evaluated after train epoch: 2,4,6...
    val_epoch: int = 2

    # List of metrics used during validation
    val_metrics: List[Callable[[str, str], float]] = field(
        default_factory=lambda: [hamming_dist, similarity]
    )

    # device where to send model and computation
    device: str = "cuda"

    # Logger to use.
    # Minimal Functionality: method log which takes variadic arguments.
    logger: Optional[Any] = None

    # Tokenizer class, technically leave the default AutoTokenizer
    # pass it only if you know what you are doing.
    tokenizer_cls: object = field(default_factory=lambda: AutoTokenizer)

    # key words arguments given to the init of the tokenizer class
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())
    # key words arguments passed at every call of the tokenizer encode
    tokenizer_call_args: Dict[str, Any] = field(
        default_factory=lambda: dict(
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
    )
    # key words arguments passed at every call of the tokenizer decode
    decoder_call_args: Dict[str, Any] = field(
        default_factory=lambda: dict(
            skip_special_tokens=True,
            # clean_up_tokenization_spaces=False,
        )
    )

    # Model class, technically leave the default AutoModelForSeq2SeqLM
    # pass it only if you know what you are doing.
    model_cls: object = field(default_factory=lambda: AutoModelForSeq2SeqLM)

    # key words arguments given to the init of the model class
    model_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())

    # Optimizer class
    optimizer_cls: object = field(default_factory=lambda: th.optim.AdamW)
    # key words arguments given to the init of the optimizer class
    optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            lr=1e-3,
        )
    )

    peft_config_cls: object = field(default_factory=lambda: LoraConfig)

    peft_config_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
    )

    lr_scheduler_cls: object = field(
        default_factory=lambda: th.optim.lr_scheduler.LambdaLR
    )

    lr_scheduler_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(lr_lambda=lambda x: 1)
    )

    def __post_init__(self):
        """Initialize all the needed classes and defaults values if needed."""
        self.logger = self.logger or No_Logger()
        self.tokenizer_name = self.tokenizer_name or self.model_name
        self.tokenizer = self.tokenizer_cls.from_pretrained(
            self.tokenizer_name, **self.tokenizer_kwargs
        )

        self.peft_config = self.peft_config_cls(**self.peft_config_kwargs)

        self._model = self.model_cls.from_pretrained(
            self.model_name, **self.model_kwargs
        )
        self.model = get_peft_model(self._model, self.peft_config)
        self.model.to(self.device)

        self.model.print_trainable_parameters()
        self.optimizer = self.optimizer_cls(
            self.model.parameters(), **self.optimizer_kwargs
        )

        self.lr_scheduler = self.lr_scheduler_cls(
            optimizer=self.optimizer, **self.lr_scheduler_kwargs
        )

        self.early_stop = EarlyStopping(**self.early_stopping_kwargs)

    def tok(self, text: str):
        """Encode a given text.
        It automatically sends its to the default device.
        returns only the "input_ids" of the encoded text
        """
        return self.tokenizer(
            text,
            **self.tokenizer_call_args,
        )["input_ids"].to(
            self.device,
        )

    def decode(self, output: th.Tensor):
        """Decode a given tensor."""
        return self.tokenizer.decode(output, **self.decoder_call_args)

    def eval_metrics(self, pred: List[str], target: List[str]):
        """Given some prediction and target calculates
        the mean for each metric.
        """
        metrics = np.array(
            [
                [m(pred, target) for m in self.val_metrics]
                for pred, target in zip(pred, target)
            ]
        )
        return {
            f"val/{m.__name__}": metrics[:, i].mean()
            for i, m in enumerate(self.val_metrics)
        }

    def evaluate(
        self,
        test_loader,
        save_path: Optional[Union[str, Path]] = None,
    ):
        """Performs an evaluation step.
        Takes a dataloader and an optional save_path.
        If the save_path is None then it log the metric.
        Otherwise it will dump the output as a txt file.
        """
        self.model.eval()
        pred_text = []
        target_text = []
        with th.no_grad():
            for batch in tqdm(test_loader):
                out_put = self.model.generate(
                    input_ids=batch["input_ids"].to(self.device)
                )
                pred_text.extend([self.decode(out) for out in out_put])
                target_text.extend(list([self.decode(lab) for lab in batch["labels"]]))

        out_put = self.eval_metrics(pred_text, target_text)
        self.logger.log(out_put)
        stop: bool = False
        if self.early_stopping:
            stop = self.early_stop(out_put[f"val/{self.val_metrics[0].__name__}"])

        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(pred_text))
        return out_put, stop

    def train(self, train_loader, val_loader):
        """Performs training on the model.

        Take two dataloader one for the training and one for the validation.
        Logs the train loss and the validation metrics.
        """
        for epoch in trange(self.epoch):
            self.model.train()
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                loss = output.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.logger.log({"train/loss": loss.detach().item()})
            self.lr_scheduler.step()
            self.logger.log({"train/lr": self.lr_scheduler.get_last_lr()})

            if epoch % self.val_epoch == 0:
                _, stop = self.evaluate(val_loader)
                if stop:
                    return
