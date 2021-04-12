"""core.py - root module exports"""

from pathlib import Path
import logging

import pytorch_lightning as pl
import nussl

###############################################################################
# Constants
###############################################################################


# Static directories
ASSETS_DIR = Path(__file__).parent / 'assets'
ROOT_DIR = Path(__file__).parent.parent
CACHE_DIR = ROOT_DIR / 'cache'
DATA_DIR = ROOT_DIR / 'data'
RUNS_DIR = ROOT_DIR / 'runs'
RESULTS_DIR = ROOT_DIR / 'results'

SEED = 42
pl.seed_everything(SEED)
nussl.utils.seed(SEED)

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

# SPEC PARAMS
HOP_LENGTH = 128
WIN_LENGTH = 512
