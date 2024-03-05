from nlp import wrapper, data, metric
from functools import partial
from transformers import default_data_collator, get_linear_schedule_with_warmup

_dataset = data.load_lang_data("ita")

wmodel = wrapper.Wrapper(
    "google/byt5-small",
    logger=wrapper.Print_Logger,
    val_epoch=1,
    lr_scheduler_cls=get_linear_schedule_with_warmup,
    lr_scheduler_kwargs=dict(num_warmup_steps=0, num_training_steps=3),
)

# Note we do this because the training data needs a special treatment:
# the padding values added by the tokenizer needs to be change to a
# negative value to ensure that the model learn only from the important parts.
# See comment in the function data.pre_tok for the mask_padding variable
_train_dataset = _dataset.pop("train")

dataset = data.process_dataset(
    _dataset,
    partial(data.pre_tok, tok=wmodel.tokenizer, max_length=128, mask_padding=False),
    _dataset["test"].column_names,
)

train_dataset = data.process_dataset(
    _train_dataset,
    partial(data.pre_tok, tok=wmodel.tokenizer, max_length=128),
    _dataset["test"].column_names,
)


train_load = data.get_dataloader(train_dataset, pin_memory=False)
test_load = data.get_dataloader(dataset["test"], pin_memory=False)
val_load = data.get_dataloader(dataset["val"], pin_memory=False)


wmodel.train(train_load, val_load)

wmodel.evaluate(test_load)


# TODO  SAVING AND LOADING PROCEDURE

# peft_model_id = f"{model_name}_{peft_config.peft_type}_{peft_config.task_type}"
# model.save_pretrained(peft_model_id)


# # load model
# from peft import PeftModel, PeftConfig

# config = PeftConfig.from_pretrained(peft_model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
# model = PeftModel.from_pretrained(model, peft_model_id)
