"""core.py - root module exports"""


from pathlib import Path


###############################################################################
# Constants
###############################################################################


# Static directories
ASSETS_DIR = Path(__file__).parent / 'assets'
CACHE_DIR = Path(__file__).parent.parent / 'cache'
DATA_DIR = Path(__file__).parent.parent / 'data'
