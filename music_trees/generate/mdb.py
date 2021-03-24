""" generate/medleydb.py - TODO: write me """
import logging
import music_trees as mt
logging.basicConfig(level=logging.ERROR) # override the config from root

import medleydb as mdb
from nussl import AudioSignal

from tqdm.contrib.concurrent import process_map
import uuid
import warnings
import os
from pathlib import Path

# these classes do not fit nicely in our hierarchy, either
# because they're too general (horn section) or not an instrument (sampler)
UNWANTED_CLASSES = ('Main System', 'fx/processed sound', 'sampler', 'horn sec`tion',
                    'string section', 'brass section', 'castanet', 'electronic organ', 'scratches', 'theremin', )

# because instrument sections are the same as the instrument, 
# we want them to be considered as one instrument
REMAP_CLASSES = {'violin section': 'violin',
                 'viola section': 'viola',
                 'french horn section': 'french horn',
                 'trombone section': 'trombone',
                 'flute section': 'flute',
                 'trumpet section': 'trumpet',
                 'clarinet section': 'clarinet',
                 'cello section': 'cello'}

def load_unique_instrument_list():
    # the first thing to do is to partition the MDB track IDs 
    # and stem IDs into only the ones we will use.
    mtracks = mdb.load_all_multitracks(['V1', 'V2'])

    # gather an instrument list
    instruments = []
    for mtrack in mtracks:
        instruments.extend([stem.instrument[0]
                           for stem in mtrack.stems.values() if stem.audio_path is not None])

    instruments = list(set(instruments))

    # filter out classes
    instruments = [
        i for i in instruments if i not in UNWANTED_CLASSES and i not in REMAP_CLASSES]
    logging.info(f'classlist is: {instruments}')

    return instruments

def get_files_for_instrument(instrument: str):
    files = list(mdb.get_files_for_instrument(instrument))
    # find out if we have any other files from the remapped section
    for key, val in REMAP_CLASSES.items():
        if instrument == val:
            files.extend(list(mdb.get_files_for_instrument(key)))
    return files

def load_all_filepaths():
    # keep track of each instrument and its list of files
    path_to_all_files = mt.ASSETS_DIR / 'medleydb' / 'mdb-files.json'
    path_to_all_files.parent.mkdir(parents=True, exist_ok=True)

    if not path_to_all_files.exists():
        FILES = {inst: get_files_for_instrument(
            inst) for inst in load_unique_instrument_list()}
        mt.utils.data.save_entry(FILES, path_to_all_files)
    else:
        FILES = mt.utils.data.load_entry(path_to_all_files)

    records = []
    for instrument, paths in FILES.items():
        for path in paths:
            records.append({'path': path, 'label': instrument})
    
    return records

def generate_medleydb_data(name: str, example_length: float, 
                           hop_length: float, sample_rate: int):
    # set an output dir
    output_dir = mt.DATA_DIR / name
    output_dir.mkdir(exist_ok=False)

    # grab the dict of all files
    file_records = load_all_filepaths()
    # add hop, example, hop, and sample rate data
    for r in file_records:
        r.update({
            'dataset': name,
            'example_length': example_length, 
            'hop_length': hop_length, 
            'sample_rate': sample_rate,
            'track': Path(r['path']).stem
        })

    # do the magic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        process_map(_generate_records_from_file, file_records, 
                    max_workers=os.cpu_count() // 2)
    
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

    # create and save a new record for each window
    for sig in windows:
        extra = dict(item)
        del extra['path']
        entry = mt.utils.data.make_entry(sig, uuid=str(uuid.uuid4()), format='wav', 
                                         **extra)

        output_path = mt.utils.data.get_path(entry)
        output_path.parent.mkdir(exist_ok=True)

        sig.write_audio_to_file(output_path.with_suffix('.wav'))
        mt.utils.data.save_entry(entry, output_path.with_suffix('.json'))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--example_length', type=float, default=0.5)
    parser.add_argument('--hop_length', type=float, default=0.125)
    parser.add_argument('--sample_rate', type=float, default=16000)

    args = parser.parse_args()

    generate_medleydb_data(**vars(args))
