""" generate/medleydb.py"""
import medleydb as mdb
import logging
import music_trees as mt
import librosa
logging.basicConfig(level=logging.ERROR)  # override the config from root


# these classes do not fit nicely in our hierarchy, either
# because they're too general (horn section) or not a physical instrument (sampler)
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
    path_to_all_files = mt.ASSETS_DIR / 'mdb' / 'mdb-files.json'
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

    dur = 0
    for record in records:
        dur += librosa.core.get_duration(filename=record['path'])
    return records


loader_fn = load_all_filepaths
