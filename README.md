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

TODO: (aldo) add instructions to runs BCE baseline (and change the name of the loss from cross-entropy to BCE)

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
python music_trees analyze.py my_experiment   <OUTPUT_NAME> 
```

the figures will be created under `/analysis/<OUTPUT_NAME>`


To generate paper-ready figures, see `scripts/figures.ipynb`. 