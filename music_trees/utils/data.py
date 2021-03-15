import os
import json
import glob
import yaml
from pathlib import Path
import warnings
import logging
import collections

import numpy as np
import pandas as pd 
import tqdm
from sklearn.model_selection import train_test_split

"""
sed utils
"""
def quantize_ceil(value, numerator, denominator, num_decimals=4, floor_threshold=0.10):
    ratio = (numerator/denominator)
    quant = round((value // ratio + 1) * ratio, num_decimals)
    if quant - value > ((1 - floor_threshold) * ratio):
        return quant - ratio
    else: 
        return quant

def quantize_floor(value, numerator, denominator, num_decimals=4, ceil_threshold=0.9):
    ratio = (numerator/denominator)
    quant = round(value // ratio * ratio, num_decimals)
    if value - quant > (ceil_threshold * ratio):
        return quant + ratio
    else:
        return quant

def get_one_hot_matrix(record, classlist: list, resolution: float = 1.0):
    # get duration from file metadata
    duration = record['duration']
    events = record['events']

    # determine the number of bins in the time axis
    assert duration % resolution == 0, \
        f'resolution {resolution} is not divisible by audio duration: {duration}'
    num_time_bins = int(duration // resolution)

    # make an empty matrix shape (time, classes)
    one_hot = np.zeros((num_time_bins, len(classlist)))
    time_axis = list(np.arange(0.0, duration+resolution, resolution))

    # get the indices for each label
    for event in events:
        start_time = event['start_time']
        end_time = event['end_time']

        start_idx = time_axis.index(quantize_floor(start_time, duration, duration / resolution))

        ceil = quantize_ceil(end_time, duration, duration / resolution)
        # truncate to the last bin (determined by the audio duration)
        # events cant be longer than the length of the track anyway
        ceil = ceil if ceil < duration else time_axis[-1]
        end_idx = time_axis.index(ceil)

        label_idx = classlist.index(event['label'])

        # now, index
        one_hot[start_idx:end_idx, label_idx] = 1
    
    return one_hot

"""
record utils
"""
def list_subdir(path):
    return [o for o in os.listdir(path) if os.path.isdir(path / o)]

def get_all_labels(record):
    """returns a list with all labels present in a 
    particular entry
    """
    labels = list(set([event['label'] for event in record['events']]))
    return labels

def get_classlist(records):
    """ iterate through records and get the set
        of all labels
    """
    all_labels = []
    for entry in records:
        for event in entry['events']:
            all_labels.append(event['label'])
    
    classlist = list(set(all_labels))
    classlist.sort()
    return classlist

def get_class_frequencies(records):
    """ 
    """
    all_labels = []
    for entry in records:
        for event in entry['events']:
            all_labels.append(event['label'])
    
    counter = collections.Counter(all_labels)
    return dict(counter)

def filter_records_by_class_subset(records, class_subset):
    subset = [entry for entry in records \
                    if all([l in class_subset for l in get_all_labels(entry)])]
    return subset

def filter_unwanted_classes(records, unwanted_classlist):
    subset = [entry for entry in records \
                if any([l not in unwanted_classlist for l in get_all_labels(entry)])]
    return subset

def train_test_split_by_entry_key(records, key='track_id', 
                                 train_size=0.8, test_size=0.2, seed=42):
    all_keys = list(set([e[key] for e in records]))

    train_keys, test_keys = train_test_split(all_keys, test_size=test_size, 
                              train_size=train_size, random_state=seed)

    train_records = [e for e in records if e[key] in train_keys]
    test_records = [e for e in records if e[key] in test_keys]

    return train_records, test_records

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
    records = [load_metadata_entry(path) for path in tqdm.tqdm(filepaths)]
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

def save_metadata_entry(entry, path, format='json'):
    os.makedirs(Path(path).parent, exist_ok=True)
    path = _add_file_format_to_filename(path, format)
    if format == 'json':
        with open(path, 'w') as f:
            json.dump(entry, f)
    elif format == 'yaml':
        with open(path, 'w') as f:
            yaml.dump(entry, f)
    
def load_metadata_entry(path, format='json'):
    entry = None
    if format == 'json':
        with open(path, 'r') as f:
            entry = json.load(f)
    elif format == 'yaml':
        with open(path, 'r') as f:
            entry = yaml.load(f, allow_pickle=True)
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