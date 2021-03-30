import music_trees as mt

import requests
from pathlib import Path
import warnings
import urllib.request
import logging
import os
import zipfile
import re
import glob
import uuid

from tqdm.contrib.concurrent import process_map
from nussl import AudioSignal

logger = logging.getLogger()
logger.setLevel(logging.INFO)
RAW_DOWLOAD_DIR = Path('/media/CHONK/data/katunog')
BASE_URL = 'https://katunog.asti.dost.gov.ph/'

def clean(string: str):
    string = re.sub("[\(\[].*?[\)\]]", "", string)
    return string.strip().strip("',/\n").lower().replace(' ', '_')

def extract_nested_zip(zippedFile, toFolder):
    """ Extract a zip file including any nested zip files
        Delete the zip file(s) after extraction
    """
    with zipfile.ZipFile(zippedFile, 'r') as zfile:
        zfile.extractall(path=toFolder)
    os.remove(zippedFile)

    for root, dirs, files in os.walk(toFolder):
        dirs[:] = [d for d in dirs if not d[0] == '_']
        for filename in files:
            if re.search(r'\.zip$', filename):
                fileSpec = os.path.join(root, filename)
                logging.info(f'extracting: {fileSpec}')
                filename = filename[0:-4]
                extract_nested_zip(fileSpec, os.path.join(root, filename))

def create_entry(args):
    path, root_dir = args
    path = Path(path)
    filename = path.name
    logging.info(f'processing: {filename}')
    if 'viola_D6_05_piano_arco-normal.mp3' in filename or \
        'saxophone_Fs3_15_fortissimo_normal.mp3' in filename or \
        "guitar_Gs4_very-long_forte_normal.mp3" in filename or \
            "bass-clarinet_Gs3_025_piano_normal.mp3" in filename:
        os.remove(path)
        return
    if path.suffix == '.mp3':
        # convert mp3 to wav
        src = path
        dst = path.with_suffix('.wav')

        # convert wav to mp3
        sound = pydub.AudioSegment.from_mp3(src)
        sound.export(dst, format='wav')

        os.remove(src)

        path = dst
        filename = path.name

        fsplit = filename.split('_')

        metadata = {
            'instrument': fsplit[0],
            'pitch': fsplit[1],
            'path_relative_to_root': path.parent.relative_to(root_dir),
            'filename': filename,
            'note_length': fsplit[2],
            'dynamic': fsplit[3],
            'articulation': fsplit[4]
        }
    return metadata

def query_instrument_metadata():
    """ finds all instruments offered by the api
    and its control numbers for download"""
    # this is a weird request format
    # but its copy pasted from the api docs
    data = {
        "query": "{instruments(page: 1, limit: 500) {page, pages, hasNext, hasPrev, objects {id, controlNumber, localName, englishName, hornbostel { name }, fileSet {edges {node {name, size, caption, fileType, isPublic, uploadDone, path, isBackground, number}}}}}}"
    }

    response = requests.post(BASE_URL + 'api/', data)
    inst_records = response.json()['data']['instruments']['objects']

    instrument_metadata = [dict(name=clean(r['localName']), 
                                hornbostel=clean(r['hornbostel']['name']),
                                ctrl_num=int(r['controlNumber'].strip('PIISD'))) for r in inst_records]
    return instrument_metadata

def glob_audio_files(subdir: Path):
    # get the list of all valid audio files
    valid_audio_formats = ('**/*.wav', '**/*.mp3')
    files = []
    for fmt in valid_audio_formats:
        files.extend(glob.glob(str(subdir / fmt)))

    return files

def download_and_extract_all_files_for_instrument(output_dir, entry: dict):
    url = f'{BASE_URL}/instruments/download_all_files?instrument_id={entry["ctrl_num"]}&file_type=audio'
    zip_path = f"{output_dir}.zip"
    
    logging.info(f'downloading from {url}')
    urllib.request.urlretrieve(url, zip_path)

    extract_nested_zip(zip_path, output_dir)

def download_and_get_katunog_data():
    """returns a dict with format {instrument_name: list of audio files belonging to that instrument}. 
    This will download all audio files from the katunog website if not found in RAW_DOWNLOAD_DIR. 
    """
    instrument_metadata = query_instrument_metadata()

    hierarchy = {}
    files = []
    for instrument in instrument_metadata:
        inst_name = instrument['name']
        hornbostel = instrument['hornbostel']
        
        # save the raw, unadultered audio first, then 
        # partition it and create 
        raw_inst_subdir = RAW_DOWLOAD_DIR / hornbostel / inst_name
        
        # if the dir exists, skip it
        # if raw_inst_subdir.exists():
        #     warnings.warn(f'it looks like there"s already a directory under {raw_inst_subdir}. \
        #         Will not redownload files again')
        # else:
        raw_inst_subdir.mkdir(parents=True, exist_ok=True)
        # download if we gotta
        download_and_extract_all_files_for_instrument(raw_inst_subdir, entry=instrument)

        # grab all the audio files for this instrument
        fs = glob_audio_files(raw_inst_subdir)
        for f in fs:
            files.append({'label': inst_name,
                          'hornbostel': hornbostel,
                           'path': f})
        
        if hornbostel not in hierarchy:
            hierarchy[hornbostel] = []
        
        if inst_name not in hierarchy[hornbostel]:
            hierarchy[hornbostel].append(inst_name)

    # save the instrument hierarchy in our assets
    hierarchy_path = mt.ASSETS_DIR / 'taxonomies' / 'katunog.yaml'
    hierarchy_path.parent.mkdir(parents=False, exist_ok=True)
    mt.utils.data.save_entry(hierarchy, hierarchy_path, format='yaml')

    # log our findings    
    logging.info(f'found {len(files)} entries for instruments')
    logging.info(f'{mt.utils.data.get_class_frequencies(files)}')

    # breakpoint()
    return files

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

        assert signal.has_data, f"attemped to write an empty audio_file: {entry}"
        sig.write_audio_to_file(output_path.with_suffix('.wav'),
                                sample_rate=entry['sample_rate'])
        mt.utils.data.save_entry(entry, output_path.with_suffix('.json'))

def generate_katunog_data(name: str, example_length: float,
                           hop_length: float, sample_rate: int):
    # set an output dir
    output_dir = mt.DATA_DIR / name
    output_dir.mkdir(exist_ok=False)

    # grab the list of all files
    file_records = download_and_get_katunog_data()
    # add hop, example, hop, and sample rate data
    for r in file_records:
        r.update({
            'dataset': name,
            'example_length': example_length,
            'hop_length': hop_length,
            'sample_rate': sample_rate,
            'track': Path(r['path']).stem
        })

    # for r in file_records:
    #     _generate_records_from_file(r)
    # do the magic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        process_map(_generate_records_from_file, file_records,
                    max_workers=os.cpu_count() // 2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--example_length', type=float, default=0.5)
    parser.add_argument('--hop_length', type=float, default=0.125)
    parser.add_argument('--sample_rate', type=int, default=16000)

    args = parser.parse_args()

    generate_katunog_data(**vars(args))
