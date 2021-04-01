from music_trees.generate.core import generate_data
import argparse
from str2bool import str2bool

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True,
                    help='name of dataset to generate, either mdb or katunog')
parser.add_argument('--name', type=str, required=True,
                    help='output name of generated data')
parser.add_argument('--example_length', type=float, default=0.5)
parser.add_argument('--augment', type=str2bool, default=False)
parser.add_argument('--hop_length', type=float, default=0.125)
parser.add_argument('--sample_rate', type=int, default=16000)

args = parser.parse_args()

generate_data(**vars(args))
