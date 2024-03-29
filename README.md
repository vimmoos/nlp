# NLP
# Installation
Create a new virtual env and install all the needed requirements with
`pip install -r reqs.txt`

# WIP Code structure
+ `data.py` -> basic data preprocessing (e.g. tokenization, dataloaders ect.)
+ `metric.py` -> Some basic metric implementation
+ `gen_data` -> data generation from github
+ `early_stopping` -> simple (but general) early stopping mechanism
+ `wrapper` -> main wrapper class that performs generic training and evaluation
+ `__main__` -> main script

# WIP main
General main structure is :

+ load the data (main function `load_lang_data` from `data.py`)
+ create the model wrapper (main function `Wrapper` from `wrapper.py`)
+ process data  (main function `process_dataset` from `data.py`)
+ create dataloaders (main function `get_dataloader` from `data.py`)
+ run the `Wrapper.train`  function
+ run the `Wrapper.evaluate`  function
+ save model

**NOTE** all the function have quite some parameters with sensible defaults. Theoretically speaking you should be able to run eveything with the defaults but you might want to change something so please look into the parameters. In the future i plan to write a dataclass that incorporates all the parameters in one place. But for now there is not if you want you need to do it manually.

# Gen Data for Relatedness
Run the following command to generate all data
`python -m nlp.gen_data -l all`
To see all the options please use the help with
`python -m nlp.gen_data -h`


# Running Baselines
Run the following command to get the available options for the baseline
`python -m nlp baseline --help`
Then run choice a set of options and run the program for example:
`python -m nlp baseline --logger print --early_patience 5 --val_epoch 2 --max_epoch 1 --lang spa`


# TODO
+ More documentation and comments
