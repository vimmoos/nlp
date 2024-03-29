# NLP Project

## Installation

1. **Create a virtual environment:** It's recommended to create a new virtual environment to isolate project dependencies.
2. **Install requirements:** Install the required packages using the provided requirements file:
    ```bash
    pip install -r reqs.txt
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



## Getting Started

1. Prepare Dataset

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


## TODO
