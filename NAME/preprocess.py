"""preprocess.py - data preprocessing"""


import argparse

import NAME


###############################################################################
# Preprocess
###############################################################################


def dataset(name):
    """Preprocess dataset in data directory and save in cache

    Arguments
        name - string
            The name of the dataset to preprocess
    """
    # Get input files and metadata to be preprocessed
    # TODO - replace with your datasets
    if name == 'DATASET':
        inputs = DATASET_inputs()
    else:
        raise ValueError(f'Dataset {name} is not implemented')

    input_directory = NAME.DATA_DIR / name
    output_directory = NAME.CACHE_DIR / name

    # TODO - Perform preprocessing.
    #
    #        Note that you will need to preprocess a single example in
    #        infer.py. It is recommended to design your code accordingly (e.g.,
    #        with a from_file() function that loads, preprocesses, and returns
    #        preprocessed features).
    #
    #        If your preprocessing task is suitable for multiprocessing, also
    #        implement that here (e.g., with multiprocessing.Pool).
    #
    #        If your preprocessing takes a long time, consider adding
    #        monitoring via tqdm. Note that monitoring a multithreaded job
    #        requires additional steps (see
    #        https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar).
    raise NotImplementedError


###############################################################################
# Dataset-specific
###############################################################################


def DATASET_inputs():
    """Get a list of preprocessing inputs

    Returns
        inputs - list
            The files and metadata needed to preprocess each item in the
            dataset. The exact type of each element is project-specific.
            For image classification, inputs is a list of filenames of images
            to preprocess. For text-to-speech, inputs is a pair of filenames
            (the text file and speech file). All datasets within a project
            should use the same type for each element.
    """
    # TODO
    raise NotImplementedError


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        help='The name of the dataset to preprocess')
    return parser.parse_args()


if __name__ == '__main__':
    dataset(**vars(parse_args()))
