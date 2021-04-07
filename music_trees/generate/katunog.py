import music_trees as mt

import requests
from pathlib import Path
import urllib.request
import logging
import os
import zipfile
import glob

import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)
RAW_DOWLOAD_DIR = Path('/media/CHONK/data/katunog-eng')
BASE_URL = 'https://katunog.asti.dost.gov.ph/'


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

    instrument_metadata = [dict(name=mt.generate.core.clean(r['englishName']),
                                hornbostel=mt.generate.core.clean(
                                    r['hornbostel']['name']),
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
        download_and_extract_all_files_for_instrument(
            raw_inst_subdir, entry=instrument)

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


loader_fn = download_and_get_katunog_data
