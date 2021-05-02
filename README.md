# Music Trees

Leveraging Hierarchical Structures for Open World, Few Shot Musical Instrument Recognition

requirements from github repos: 

- medleydb from [marl](https://github.com/marl/medleydb)
- audio-utils made my [me](https://github.com/hugofloresgarcia/audio-utils)

## Installation

clone the repo and install with
```bash 
pip install -e .
```

## Usage

### Generate data

##### MedleyDB
Make sure the `MEDLEYDB_PATH` environment variable is set. Then, run the
generation script:

```bash
python -m music_trees.generate \
                --dataset medleydb \
                --name mdb \
                --example_length 1.0 \
                --augment false \
                --hop_length 1.0 \
                --sample_rate 16000 \
```

##### Katunog
In `/generate/katunog.py`, change `RAW_DOWNLOAD_DIR` to the path where you would like to store the raw dataset data (do **not** store in `/data/`).

Run the generation script:

```bash
python -m music_trees.generate \
                --dataset katunog \
                --name katunog \
                --example_length 1.0 \
                --augment false \
                --hop_length 1.0 \
                --sample_rate 16000 \
```

*hugo*: added an mdb-augmented dataset:

```bash
python -m music_trees.generate \
                --dataset mdb \
                --name mdb-augmented \
                --example_length 1.0 \
                --augment true \
                --hop_length 0.5 \
                --sample_rate 16000
```

### Partition data

Partitions are written to `music_trees/assets/<DATASET_NAME>/partition.json`. 

Create a hierarchical train-val split with depth 1 for medleydb using the `joint-taxonomy` file:

```bash
python music_trees/partition.py  --taxonomy deeper-mdb  --name mdb --partitions train val --sizes 0.7 0.3 --depth 4
```


### Preprocess data

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

The code is structured so that the model only receives one episode (which can be thought of as a batch in itself), but if you want to batch episodes together for gradient purposes, use the `--accumulate_grad_batches` flag provided by pytorch lightning. 

### Evaluate

Perform evaluation on a model. Make sure to pass the path to the run that you wish to evaluate. 

To evaluate a model:
```bash
export CUDA_VISIBLE_DEVICES='0' && python music_trees/eval.py --exp_dir <PATH_TO_RUN/version_X>
```

### Hyperparameter Search

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

### Infer

(todo)

### Generate Random Taxonomies
To generate random taxonomies use the generate-random-taxonomies script. Here is how:

 ```bash
 python music_trees/generate-random-taxonomies.py --num_taxonomies <NUMBER OF TAXONOMIES> --sizes <LIST OF SIZES> --size_type <SINGLE OR MULTI>
 ```

 num_taxonomies specifies how many random taxonomies you would like to generate. sizes is the sizes of each random taxonomy. For example `--sizes "[[1], [1, 2, 3]]"` would generate taxonomies of 1 group, and taxonomies with 1 group containing 2 sub-groups containing 3 sub-groups of the sub-group. If you want all taxonomies to have the same structure you can pass a single list: `--sizes "[1]"` size_type, if passing a single list for sizes use single, if passing a lists of lists use multi.  

### Monitor

Run `tensorboard --logdir runs/`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser. 

TODO: add docs for visualizing with emb-viz

*or through visual studio code if you're cool*
