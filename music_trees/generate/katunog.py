import requests
from pathlib import Path

RAW_DOWLOAD_DIR = Path('/media/CHONK/data/katunog')
BASE_URL = 'https://katunog.asti.dost.gov.ph/'
API_URL = BASE_URL + 'api/'

def query_records_from_api():
    # this is a weird request format
    data = {
        "query": "{instruments(page: 1, limit: 500) {page, pages, hasNext, hasPrev, objects {id, controlNumber, localName, hornbostel { name }, fileSet {edges {node {name, size, caption, fileType, isPublic, uploadDone, path, isBackground, number}}}}}}"
    }
    response = requests.post(API_URL, data)

    inst_records = response.json()['data']['instruments']['objects']
    breakpoint()
    # grab all the audio files we can find
    output_records = []
    for record in inst_records:
        name = record['localName']
        files = record['fileSet']['edges']
        hornbostel = record['hornbostel']
        ctrl_num = int(record['controlNumber'].strip('PIISD'))

        # some of these files are images and videos, 
        # so we only want to keep the audio 
        audio_files = []
        for file_node in files:
            entry = file_node['node']
            if 'audio' in entry['fileType']:
                audio_entry = {
                    'label': name,
                    'hornbostel': hornbostel, 
                    'urlpath': entry['path']
                }
                audio_files.append(audio_entry)
    
        output_records.extend(audio_files)
    
    return output_records

records = query_records_from_api()
breakpoint()

def download_all_files_for_instrument(output_path, dirname, ctrl_num):
    response = requests.get(f'{BASE_URL}/instruments/download_all_files?instrument_id={ctrl_num}&file_type=audio')

    breakpoint()

'https://katunog.asti.dost.gov.ph/instruments/download_all_files?instrument_id=2607&file_type=audio'
