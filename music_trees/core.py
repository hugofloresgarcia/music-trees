"""core.py - root module exports"""

from pathlib import Path
import logging

import pytorch_lightning as pl
import nussl
import os
import random
import numpy as np
import torch

# Static directories
ASSETS_DIR = Path(__file__).parent / 'assets'
ROOT_DIR = Path(__file__).parent.parent
CACHE_DIR = ROOT_DIR / 'cache'
DATA_DIR = ROOT_DIR / 'data'
RUNS_DIR = ROOT_DIR / 'runs'
RESULTS_DIR = ROOT_DIR / 'results'

TQDM_DISABLE = False

SEED = 42
pl.seed_everything(SEED)
nussl.utils.seed(SEED)

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

def super_seed():
    pl.seed_everything(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED) 
    # for cuda
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

# SPEC PARAMS
HOP_LENGTH = 128
WIN_LENGTH = 512
