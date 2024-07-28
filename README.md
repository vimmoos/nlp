# NLP Project - Investigating LoRA for Cross-Lingual Adaptation in Word Inflection

## Installation

1. **Create a virtual environment:** It's recommended to create a new virtual environment to isolate project dependencies.
2. **Install requirements:** Install the required packages using the provided requirements file:
    ```bash
    pip install -r reqs.txt
    ```

## Configurations
In the `confs` folder the different configurations for the experiments are present

## Pre-trained Models
* All the trained models are in the `models` folder.
* All the results of the model evaluation are in the `results` folder

In case you want to use the trained model outside of the project please follow the instruction for loading PEFT models from here: https://huggingface.co/docs/peft/main/en/tutorial/peft_model_config#peft-models
Generally you can use the following lines of code:

```python
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("NAME_OF_THE_MODEL_FOLDER")
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, "NAME_OF_THE_MODEL_FOLDER")
```

## Results
All the results of this study can be found here: https://wandb.ai/vimmoos/test_nlp
To inspected we suggest to go on the sweep section and then navigate the different experiments from there https://wandb.ai/vimmoos/test_nlp/sweeps



## Getting Started
**NOTE** If you want to use wandb as a logger make sure you are logged in in your machine. It will log everything in the project `test_nlp`

**1. Prepare Dataset**
   * **Use `gen_data`:** For supported datasets, use the following command to download and preprocess data:

      ```bash
      python -m nlp.gen_data -l all  # Download all supported datasets
      ```
      Use `python -m nlp.gen_data -h` for options and help.
	  If you want to download a language which is not included edit the `gen_data.py` file

**2. Run Baseline Experiments**

   * **Check Available Options:**
      ```bash
      python -m nlp baseline --help
      ```
   * **Example Run with Selected Parameters:**
      ```bash
      python -m nlp baseline --logger print --early_patience 5 --val_epoch 2 --max_epoch 1 --lang spa
      ```

**3. Run Relatedness Experiments**
   * **Check Available Options:**
      ```bash
      python -m nlp related --help
      ```

   * **Example Run with Selected Parameters:**
      ```bash
      python -m nlp related --logger print --early_patience 5 --val_epoch 2 --max_epoch 1 --train_languages ita --test_languages spa
	  ```

**4. Evaluate models**
   * **Check Available Options:**
      ```bash
      python -m nlp evaluate --help
      ```


   * **Example Run with Selected Parameters:**
      ```bash
      python -m nlp evaluate  --logger print --rank 8 --train_languages multi_romance --test_languages ita
	  ```

## Code Structure

* **`data.py`** Contains functions for essential data preprocessing tasks like tokenization and the creation of data loaders.
* **`metric.py`**  Implements various metrics for evaluating model performance.
* **`gen_data`** Handles downloading and preparing datasets from a GitHub repository.
* **`early_stopping`** Provides a flexible early stopping mechanism to prevent overfitting during training.
* **`wrapper`**  The `Wrapper` class encapsulates common training and evaluation procedures.
* **`cli.py`** Defines the command-line interface (CLI) structure, including arguments and subcommands (like 'baseline' and 'related').
* **`runner.py`**  Contains functions like `run_baseline` and `run_relatedness`, which hold the core logic for running different types of experiments.
* **`hyper.py`** Defines the `HyperRelatedness` dataclass to store experiment-related hyperparameters.
* **`__main__`** The main script where you configure experiments and launch training/evaluation routines.
