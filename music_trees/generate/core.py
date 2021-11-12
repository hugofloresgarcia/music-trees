import music_trees as mt
from copy import deepcopy

import os
import re
from pathlib import Path
import uuid
import logging
import warnings

from nussl import AudioSignal

import tqdm
from tqdm.contrib.concurrent import thread_map, process_map

NUM_AUGMENT_FOLDS = 2


def _generate_records_from_file(item: dict):
    """ expects a dict with structure:
    {
        'path': path to the audio file
        'instrument': instrument label
    }
    """
    # load the audio
    signal = AudioSignal(path_to_input_file=item['path'])
    signal.resample(item['sample_rate'])

    # remove all silence
    signal = mt.utils.audio.trim_silence(signal, item['sample_rate'])

    # window signal
    window_len = int(item['sample_rate'] * item['example_length'])
    hop_len = int(item['sample_rate'] * item['hop_length'])

    windows = mt.utils.audio.window(signal, window_len, hop_len)

    # augment if necessary
    if item['augment']:
        for i in range(NUM_AUGMENT_FOLDS):
            logger = logging.getLogger()
            logger.disabled = True
            clips = [mt.utils.effects.augment_from_audio_signal(
                sig) for sig in deepcopy(windows)]
            windows.extend(clips)
            logger.disabled = False

    # create and save a new record for each window
    for sig in windows:
        extra = dict(item)
        del extra['path']
        entry = mt.utils.data.make_entry(sig, uuid=str(uuid.uuid4()), format='wav',
                                         **extra)

        if hasattr(sig, '_effect_params'):
            entry['effect_params'] = sig._effect_params
        else:
            entry['effect_params'] = {}

        output_path = mt.utils.data.get_path(entry)
        output_path.parent.mkdir(exist_ok=True)

        assert sig.has_data, f"attemped to write an empty audio_file: {entry}"
        sig.write_audio_to_file(output_path.with_suffix('.wav'),
                                sample_rate=entry['sample_rate'])
        mt.utils.data.save_entry(entry, output_path.with_suffix('.json'))


def clean(string: str):
    string = re.sub("[\(\[].*?[\)\]]", "", string)
    string = string.strip().strip("',/\n").lower().replace(' ', '_')
    string = string.replace('/', '-')
    return string


def generate_data(dataset: str, name: str, example_length: float,
                  augment: bool, hop_length: float, sample_rate: int):
    # set an output dir
    output_dir = mt.DATA_DIR / name
    output_dir.mkdir(exist_ok=True, parents=True)

    # grab the dict of all files
    assert dataset in ('mdb', 'katunog')
    loader_fn = mt.generate.mdb.loader_fn if dataset == 'mdb'\
        else mt.generate.katunog.loader_fn
    file_records = loader_fn()
    # add hop, example, hop, and sample rate data
    for r in file_records:
        r.update({
            'dataset': name,
            'example_length': example_length,
            'hop_length': hop_length,
            'sample_rate': sample_rate,
            'track': Path(r['path']).stem,
            'augment': augment,
        })

    # for f in tqdm.tqdm(file_records):
    #     _generate_records_from_file(f)

    # do the magic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        process_map(_generate_records_from_file, file_records,
                    max_workers=os.cpu_count() // 2)
