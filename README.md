# Music Trees

Code todos:
- [ ] fix audio augmentation bug in generate/core.py
- [ ] finish writing TreeModel, MetaTree, ProtoTree, and add load_model() to MetaTask. 

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

### Partition data

Partitions are written to `music_trees/assets/<DATASET_NAME>/partition.json`. 

Create a hierarchical train-val split with depth 1 for medleydb:

```bash
python music_trees/partition.py \
                --taxonomy joint-taxonomy \
                --name mdb 
                --partitions train val
                --sizes 0.7 0.3
                --depth 1
```


### Preprocess data

Preprocessing and caching is done on the fly, so no need to worry about running a script. 

The `MetaDataset` object stores cached **entries** (aka dicts of inputs, targets and metadata) as `pickle` objects in `/cache/<DATASET_NAME>/<TRANSFORM>`, where `<TRANSFORM>` is the preprocessing transform (most of the time a spectrogram). 

For deterministic purposes (validation and evaluation), set `deterministic=True`. This way, episode metadata is also cached for a particular combination of `n_shot`, `n_query`, `n_class`. 

### Train

Training runs are stored under `/runs/<NAME>/<VERSION>`. You shouldn't have to specify a version when running the script, as the version number is inferred automatically. If you are resuming from a checkpoint, then do specify the version. 

see `python train.py -h` for the full list of args provided by the pytorch lightning trainer. 

```bash 
export CUDA_VISIBLE_DEVICES='0' && python music_trees/train.py --name <NAME> --dataset mdb --num_workers 20  --learning_rate 0.03  
```

### Evaluate

Both name and version are required here, since we're loading a previously trained model. 

To evaluate a model:
```bash
export CUDA_VISIBLE_DEVICES='0' && python music_trees/eval.py --name <NAME> --version <VERSION>
```

### Infer

(todo)


### Monitor

Run `tensorboard --logdir runs`. If you are running training
remotely, you must create a SSH connection with port forwarding to view
Tensorboard. This can be done with `ssh -L 6006:localhost:6006
<user>@<server-ip-address>`. Then, open `localhost:6006` in your browser. 

TODO: add docs for visualizing with emb-viz

*or through visual studio code if you're cool*