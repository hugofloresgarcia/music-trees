"""infer.py - model inference"""


import argparse
from pathlib import Path


###############################################################################
# Infer
###############################################################################


def from_file(input_file, checkpoint_file):
    """Run inference on one example on disk

    Arguments
        input_file - string
            The file containing input data
        checkpoint_file - string
            The model checkpoint file

    Returns
        TODO - define the return value for your project
    """
    # TODO - load input and call one()
    raise NotImplementedError


def from_file_to_file(input_file, output_file, checkpoint_file):
    """Run inference on one example on disk and save to disk

    Arguments
        input_file - string
            The file containing input data
        output_file - string
            The file to write inference results
        checkpoint_file - string
            The model checkpoint file
    """
    # Load and run inference
    result = from_file(input_file, checkpoint_file)

    # TODO - save to disk
    raise NotImplementedError


def one(input, checkpoint_file):
    """Run inference on one example

    Arguments
        input - TODO - define the input for your project
        checkpoint_file - string
            The model checkpoint file

    Returns
        TODO - define the return value for your project
    """
    # TODO - preprocess, collate, place on device, and infer
    raise NotImplementedError


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path, help='The input file')
    parser.add_argument('output_file', type=Path, help='File to save results')
    parser.add_argument('checkpoint_file', type=Path, help='Model weight file')
    return parser.parse_args()


if __name__ == '__main__':
    from_file_to_file(**vars(parse_args()))
