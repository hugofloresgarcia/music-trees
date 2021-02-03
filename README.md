# Deep learning project template

Throughout this template, `NAME` is used to refer to the name of the project
and `DATASET` is used to refer to the name of a dataset.


## Installation

Clone this repo and run `cd NAME && pip install -e .`.

## Usage

### Download data

Place datasets in `data/DATASET`, where `DATASET` is the name of the dataset.


### Partition data

Complete all TODOs in `partition.py`, then run `python -m NAME.partition
DATASET`.


### Preprocess data

Complete all TODOs in `preprocess.py`, then run `python -m NAME.preprocess
DATASET`. All preprocessed data is saved in `cache/DATASET`.


### Train

Complete all TODOs in `data.py` and `model.py`. Then, create a directory in
`runs` for your experiment. Logs, checkpoints, and results should be saved to
this directory. In your new directory, run `python -m NAME.train --dataset
DATASET <args>`. See the [PyTorch Lightning trainer flags](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#trainer-flags)
for additional arguments.


### Evaluate

Complete all TODOs in `evaluate.py`, then run `python -m NAME.evaluate DATASET
<partition> <checkpoint> <file>`, where `<partition>` is the name of the
partition to evaluate, `<checkpoint>` is the checkpoint file to evaluate, and
`<file>` is the json file to write results to.


### Infer

Complete all TODOs in `infer.py`, then run `python -m NAME.infer
<input_file> <output_file> <checkpoint_file>`.


### Monitor

Run `tensorboard --logdir runs/<run>/logs`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser.


### Test

Tests are written using `pytest`. Run `pip install pytest` to install pytest.
Complete all TODOs in `test_model.py` and `test_data.py`, then run `pytest`.
Adding project-specific tests for preprocessing and inference is encouraged.


## FAQ

### What is the directory `NAME/assets` for?

This directory is for
[_package data_](https://packaging.python.org/guides/distributing-packages-using-setuptools/#package-data).
When you pip install a package, pip will
automatically copy the python files to the installation folder (in
`site_packages`). Pip will _not_ automatically copy files that are not Python
files. So if your code depends on non-Python files to run (e.g., a pretrained
model, normalizing statistics, or data partitions), you have to manually
specify these files in `setup.py`. This is done for you in this repo. In
general, only small files that are essential at runtime should be placed in
this folder.


### What if I have an especially complex preprocessing pipeline?

I recommend one of two designs.
1. Replace `preprocess.py` with a `preprocess` submodule. This will
be a directory `NAME/preprocess` that contains a module initialization script
`__init__.py`, an entry point `__main__.py`, and the rest of your preprocessing
code.
2. Implement only the entry point in `preprocess.py`. Move the data
transformations to either a new file `transform.py` or new `transform`
submodule.


### What if my evaluation includes subjective experiments?

In this case, replace the `<file>` argument of `NAME.evaluate` with a
directory. Write any objective metrics to a file within this directory, as well
as any generated files that will be subjectively evaluated. If evaluation
is especially complex, consider making an `evaluate` submodule.


### How do I release my code so that it can be downloaded via pip?

Code release involves making sure that `setup.py` is up-to-date and then
uploading your code to [`pypi`](https://www.pypi.org).
[Here](https://packaging.python.org/tutorials/packaging-projects/) is a good
tutorial for this process.
