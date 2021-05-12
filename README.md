# Music Trees

Leveraging Hierarchical Structures for Few Shot Musical Instrument Recognition


## Installation

first, clone the medleydb repo and install using `pip install -e`:
- medleydb from [marl](https://github.com/marl/medleydb)

then, clone this repo and install with
```bash 
pip install -e .
```

## Replicating Experiments!

### 1. Generate data

Make sure the `MEDLEYDB_PATH` environment variable is set (see the medleydb [repo](https://github.com/marl/medleydb) for more instructions ). Then, run the
generation script:

```bash
python -m music_trees.generate \
                --dataset medleydb \
                --name mdb-aug \
                --example_length 1.0 \
                --augment true \
                --hop_length 0.5 \
                --sample_rate 16000 \
```

This will generate both augmented and unaugmented data for MedleyDB

### 2. Partition data

The partition file used for all experiments is available at `/music_trees/assets/partitions/mdb-aug.json`. 

### 3. Run experiments

The `search` script will train all models for a particular experiment. It will grab as many GPUs are available (use `CUDA_VISIBLE_DEVICES` to change the availability of GPUs) and train as many models as it can in parallel. 

Each model will be stored under `/runs/<NAME>/<VERSION>`.

**Arbitrary Hierarchies**
```bash
export CUDA_VISIBLE_DEVICES=0,1 && python music_trees/search.py --name scrambled-tax
```

**Height Search**
(note that `height=0` and `height=1` are the baseline and proposed model, respectively)
```bash
export CUDA_VISIBLE_DEVICES=0,1 && python music_trees/search.py --name height-v1
```

**Loss Ablation**
```bash
export CUDA_VISIBLE_DEVICES=0,1 && python music_trees/search.py --name loss-alpha
```

### 4. Evaluate

Perform evaluation on a model. Make sure to pass the path to the run that you wish to evaluate. 

To evaluate a model:
```bash
python music_trees/eval.py --exp_dir <PATH_TO_RUN>/<VERSION>
```

Each model will store its evaluation results under `/results/<NAME>/<VERSION>`

### 5. Analyze

To compare models and generate analysis figures and tables, place of all the results folders you would like to analyze under a single folder. The resulting folder should look like this:

```bash
my_experiment/trial1/version_0
my_experiment/trial2/version_0
my_experiment/trial3/version_0
```

Then, run analysis using 
```bash
python music_trees analyze.py my_experiment/<OUTPUT_NAME> 
```

the figures will be created under `/analysis/<OUTPUT_NAME>`


To generate paper-ready figures, see `scripts/figures.ipynb`. 

<!-- ### Hyperparameter Search

To run a hyperparameter search, add a new dictionary to CONFIGS in `music_trees/search.py` and run:

(pass a list of ints to CUDA_VISIBLE_DEVICES if you want to search on multiple GPUs)
```bash
export CUDA_VISIBLE_DEVICES=0,1 && python music_trees/search.py --name <CONFIG_NAME>
```

where `num_samples` is the number of trials to run in the search is `gpu_capacity` is an estimate of the amount of GPU memory (as a fraction) that each run will take. 

### Analyze

To run a model comparison, generate figures, etc. use analyze.py:

```bash
python music_trees/analyze.py <PATH_TO_RESULTS> <OUTPUT_NAME>
```

### Monitor

Run `tensorboard --logdir runs/`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser.  -->



<!-- ### Preprocess data

Preprocessing and caching is done on the fly, so no need to worry about running a script. 

The `MetaDataset` object stores cached **entries** (aka dicts of inputs, targets and metadata) as `pickle` objects in `/cache/<DATASET_NAME>/<TRANSFORM>`, where `<TRANSFORM>` is the preprocessing transform (most of the time a spectrogram). 

For deterministic purposes (validation and evaluation), set `deterministic=True`. This way, episode metadata is also cached for a particular combination of `n_shot`, `n_query`, `n_class`. 

### Train

Training runs are stored under `/runs/<NAME>/<VERSION>`. You shouldn't have to specify a version when running the script, as the version number is inferred automatically. If you are resuming from a checkpoint, then do specify the version. 

see `python music_trees/train.py -h` for the full list of args provided by the pytorch lightning trainer. 

**note**: because we want swap models without modifying the overall training logic, 
the actual model architecture is wrapped in `models.task.MetaTask` object, which takes care of defining optimizers, loading the actual model, and logging. One of the args required by `MetaTask` is `model_name`, which is the name of the actual model to load. 

To view what models are available, see `MetaTask.load_model_parser`. Note that each model under `load_model_parser` has its own set of required hyperparameters. For example, the required args for `hprotonet` (`HierarchicalProtoNet`) are `d_root`, `height`, and `taxonomy_name`. Each model has its own `add_model_specific_args` function, where you can look at the required arguments, default values, and help strings. 


```bash
# trains a protonet with hierarchical loss with height 2 and a loss decay of 1
export CUDA_VISIBLE_DEVICES='0' && python music_trees/train.py --model_name hprotonet --height 4 --d_root 128 --loss_alpha 1 --name <NAME> --dataset mdb-augmented --num_workers 20  --learning_rate 0.03  
```

a new experiment will be created under `runs/<NAME>/<VERSION>`. checkpoints, embedding spaces, and other goodies will be stored there. 

##### training with a batch size greater than 1

The code is structured so that the model only receives one episode (which can be thought of as a batch in itself), but if you want to batch episodes together for gradient purposes, use the `--accumulate_grad_batches` flag provided by pytorch lightning.  -->
