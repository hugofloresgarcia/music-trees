"""core.py - root module exports"""

from pathlib import Path
import logging

import pytorch_lightning as pl

###############################################################################
# Constants
###############################################################################


# Static directories
ASSETS_DIR = Path(__file__).parent / 'assets'
CACHE_DIR = Path(__file__).parent.parent / 'cache'
DATA_DIR = Path(__file__).parent.parent / 'data'

SEED = 42
pl.seed_everything(SEED)

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)