# Music Trees

Supplementary code for the experiments described in the 2021 ISMIR submission: Leveraging Hierarchical Structures for Few Shot Musical Instrument Recognition. 

##  train-test splits and hierarchies. 

- For all experiments, we used the instrument-based split in `/music_trees/assets/partitions/mdb-aug.json`. 
- To view our Hornbostel-Sachs class hierarchy, see `/music_trees/assets/taxonomies/deeper-mdb.yaml`. Note that not all of the instruments on this taxonomy are used in our experiments. 
- All random taxonomies are in `/music_trees/assets/taxonomies/scrambled-*.yaml` 


## Installation

first, clone the medleydb repo and install using `pip install -e`:
- medleydb from [marl](https://github.com/marl/medleydb)

install some utilities for visualizing the embedding space:
```bash
git clone https://github.com/hugofloresgarcia/embviz.git
cd embviz
pip install -e .
```

then, clone this repo and install with
```bash 
pip install -e .
```

## Usage

### 1. Generate data

Make sure the `MEDLEYDB_PATH` environment variable is set (see the medleydb [repo](https://github.com/marl/medleydb) for more instructions ). Then, run the
generation script:

```bash
python -m music_trees.generate \
                --dataset mdb \
                --name mdb-aug \
                --example_length 1.0 \
                --augment true \
                --hop_length 0.5 \
                --sample_rate 16000 \
```

This will generate both augmented and unaugmented data for MedleyDB. **NOTE**: There was a bug in the code that disabled data augmentation silently. This bug has been left in the code for the sake of reproducibility. This is why we don't report any data augmentation in the paper, as none was applied at the time of experiments.

### 2. Partition data

The partition file used for all experiments is available at `/music_trees/assets/partitions/mdb-aug.json`. 

### 3. Run experiments

The `search` script will train all models for a particular experiment. It will grab as many GPUs are available (use `CUDA_VISIBLE_DEVICES` to change the availability of GPUs) and train as many models as it can in parallel. 

Each model will be stored under `/runs/<NAME>/<VERSION>`.

**Arbitrary Hierarchies**
```bash
python music_trees/search.py --name scrambled-tax
```

**Height Search**
(note that `height=0` and `height=1` are the baseline and proposed model, respectively)
```bash
python music_trees/search.py --name height-v1
```

**Loss Ablation**
```bash
python music_trees/search.py --name loss-alpha
```

train the additional BCE baseline:
```bash
python music_trees/train.py --model_name hprotonet --height 4 --d_root 128 --loss_alpha 1 --name "flat (BCE)" --dataset mdb-aug --learning_rate 0.03 --loss_weight_fn cross-entropy
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
python music_trees analyze.py my_experiment   <OUTPUT_NAME> 
```

the figures will be created under `/analysis/<OUTPUT_NAME>`


To generate paper-ready figures, see `scripts/figures.ipynb`. 