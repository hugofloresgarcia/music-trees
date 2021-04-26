from logging import disable
import os
import json
import glob
import yaml
from pathlib import Path
from typing import List
import collections

import music_trees as mt

import numpy as np
import pandas as pd
import tqdm
from tqdm.contrib.concurrent import process_map

"""
record utils (for records that only contain one event)
"""


def make_entry(signal, dataset: str, uuid: str, format: str, example_length: float, hop_length: float,
               sample_rate: int, label: str, **extra):
    """ create a new dataset entry
    """
    assert signal.sample_rate == sample_rate

    return dict(dataset=dataset, uuid=uuid, format=format, example_length=example_length,
                hop_length=hop_length, sample_rate=sample_rate, label=label, **extra)


def get_path(entry):
    """ returns an entry's path without suffix
    add .wav for audio, .json for metadata
    """
    return mt.DATA_DIR / entry['dataset'] / entry['label'] / entry['uuid']


def list_subdir(path):
    """ list all subdirectories given a directory"""
    return [o for o in os.listdir(path) if os.path.isdir(path / o)]


def get_classlist(records):
    """ iterate through records and get the set
        of all labels
    """
    all_labels = [entry['label'] for entry in records]

    classlist = list(set(all_labels))
    classlist.sort()
    return classlist


def get_one_hot(label: str, classes: List[str]):
    """
    given a label and its classlist, 
    returns an np array one-hot of it
    """
    if label not in classes:
        raise ValueError(f"{label} is not in {classes}")

    return np.array([1 if label == c else 0 for c in classes])


def get_class_frequencies(records: List[dict]):
    """ counts the number of examples belonging to each label, and returns a dict
    """
    all_labels = [entry['label'] for entry in records]
    counter = collections.Counter(all_labels)
    return dict(counter)


def filter_records_by_class_subset(records: List[dict], class_subset: List[str]):
    """ remove all records that don't belong to the provided class subset"""
    subset = [entry for entry in records if entry['label'] in class_subset]
    return subset


def filter_unwanted_classes(records, unwanted_classlist):
    """ given a list of unwanted classes, remove all records that match those """
    subset = [entry for entry in records if not entry['label']
              in unwanted_classlist]
    return subset


"""
glob
"""


def glob_all_metadata_entries(root_dir, pattern='**/*.json'):
    """ reads all metadata files recursively and loads them into
    a list of dicts
    """
    pattern = os.path.join(root_dir, pattern)
    filepaths = glob.glob(pattern, recursive=True)
    # metadata = tqdm.contrib.concurrent.process_map(load_yaml, filepaths, max_workers=20, chunksize=20)
    # records = [load_entry(path) for path in tqdm.tqdm(
    #     filepaths, disable=mt.TQDM_DISABLE)]
    records = process_map(
        load_entry, filepaths, disable=mt.TQDM_DISABLE, max_workers=os.cpu_count() // 8)
    return records


"""
json and yaml 
"""


def _add_file_format_to_filename(path: str, file_format: str):
    if '.' not in file_format:
        file_format = f'.{file_format}'

    if Path(path).suffix != file_format:
        path = Path(path).with_suffix(file_format)
    return str(path)


def save_entry(entry, path, format='json'):
    """ save to json (or yaml) """
    os.makedirs(Path(path).parent, exist_ok=True)
    path = _add_file_format_to_filename(path, format)
    if format == 'json':
        with open(path, 'w') as f:
            json.dump(entry, f)
    elif format == 'yaml':
        with open(path, 'w') as f:
            yaml.dump(entry, f)


def load_entry(path, format='json'):
    """ load json (or yaml) """
    entry = None
    if format == 'json':
        with open(path, 'r') as f:
            entry = json.load(f)
    elif format == 'yaml':
        with open(path, 'r') as f:
            entry = yaml.load(f)
    else:
        raise ValueError(f'unsupported format: {format}')

    return entry


"""
csv
"""


def save_records_csv(records, path_to_records):
    pd.DataFrame(records).to_csv(path_to_records, index=False)


def load_records_csv(path_to_records):
    assert os.path.exists(path_to_records), f"{path_to_records} does not exist"
    records = pd.read_csv(path_to_records).to_dict('records')
    return records
