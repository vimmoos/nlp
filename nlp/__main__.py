from nlp import wrapper, data, metric
from functools import partial


_dataset = data.load_lang_data("ita")

wmodel = wrapper.Wrapper("google/byt5-small", logger=wrapper.Print_Logger)

dataset = data.process_dataset(
    _dataset, partial(data.pre_tok, tok=wmodel.tokenizer, max_length=128)
)

train_load = data.get_dataloader(dataset["train"], pin_memory=False)
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
