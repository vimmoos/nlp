from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
    TaskType,
)
import torch
from datasets import load_dataset
from functools import partial
from transformers import default_data_collator, get_linear_schedule_with_warmup

from tqdm import tqdm


# url = Path("data")
# lang = "ita"
# version = "e7f1b32"
# example = "https://github.com/sigmorphon/2023InflectionST/tree/main/part1/data"

# data_file = {
#     "train": str((url / lang / f"{lang}.trn").absolute()),
#     "test": str((url / lang / f"{lang}.tst").absolute()),
#     "validation": str((url / lang / f"{lang}.dev").absolute()),
# }

# dataset = load_dataset("text", data_files=data_file)

# jsons = {}

# for key in data_file.keys():
#     jsons[key] = {
#         "data": [
#             {
#                 k: v
#                 for k, v in zip(
#                     ["lemma", "features", "inflected"], vs["text"].split("\t")
#                 )
#             }
#             for vs in dataset[key]
#         ],
#         "version": version,
#         "info": "The version indicate the commit from the git repo: https://github.com/sigmorphon/2023InflectionST",
#     }


# with gzip.open("train.test.json.gz", "wt") as f:
#     json.dump(jsons["train"], f)


# test = load_dataset("json", data_files="datasets/ita_test.json.gz", field="data")
# # test2 = load_dataset("json", data_files="train.json.gz", field="data")


# for key in data_file.keys():
#     json.dump(json[key])


peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model_name = "google/byt5-small"

_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = get_peft_model(_model, peft_config)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset(
    "json",
    data_files={
        "train": "datasets/ita_train.json.gz",
        "test": "datasets/ita_test.json.gz",
        "val": "datasets/ita_val.json.gz",
    },
    field="data",
)


def preprocess_function(
    data: dict,
    tok,
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
    labels = tok(targets, **base_tok_args)
    labels = labels["input_ids"]
    # IDK why exactly i guess the idea is to clearly
    # indicate if there was an error probably we can remove this
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


processed_datasets = dataset.map(
    partial(preprocess_function, tok=tokenizer, max_length=128),
    batched=True,
    # num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)


# tokenizer(dataset["train"][1:10]["features"])
# tokenizer("\t")

train_dataset = processed_datasets["train"]
val_dataset = processed_datasets["val"]

train_load = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=16,
    # pin_memory=True,
)

val_load = DataLoader(
    val_dataset,
    collate_fn=default_data_collator,
    batch_size=16,
    # pin_memory=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_load) * 3),  # 3 is the number of epochs
)

model = model.to("cuda")

for epoch in range(3):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_load)):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(val_load)):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(
                torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                skip_special_tokens=True,
            )
        )

    eval_epoch_loss = eval_loss / len(val_load)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_load)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


# save model

peft_model_id = f"{model_name}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)


# load model
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

# see results


def test_gen(i):
    model.eval()
    example = dataset["test"]["lemma"][i] + "\t" + dataset["test"]["features"][i]
    inputs = tokenizer(
        example,
        return_tensors="pt",
    )

    print(f"Input: {example}")

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
        # print(outputs)
        print(
            f"Output: {tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}"
        )
        print(f"Desired: {dataset['test']['inflected'][i]}")


for x in range(50):
    test_gen(x)
