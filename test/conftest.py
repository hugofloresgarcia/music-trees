from pathlib import Path

import pytest

import NAME


TEST_ASSETS_DIR = Path(__file__).parent / 'assets'


###############################################################################
# Pytest fixtures
###############################################################################


@pytest.fixture(scope='session')
def dataset():
    """Preload the dataset"""
    return NAME.Dataset('DATASET', 'valid')


@pytest.fixture(scope='session')
def datamodule():
    """Preload the datamodule"""
    return NAME.DataModule('DATASET', batch_size=4, num_workers=0)


@pytest.fixture(scope='session')
def model():
    """Preload the model"""
    return NAME.Model()
