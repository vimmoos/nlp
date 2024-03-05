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

# Gen Data for Relatedness
Run the following command to generate all data
`python -m nlp.gen_data -l all`
To see all the options please use the help with
`python -m nlp.gen_data -h`


# TODO
+ More documentation and comments
+ Add chrf in the metrics
+ Add an hyperparameters dataclass for experiments
+ Add argument parser for main
