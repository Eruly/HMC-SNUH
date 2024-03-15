from HMSDatasets import EEGDataset
import pytest
import pandas as pd
import numpy as np
from datasets import Dataset
# Path: tests/test_dataset.py

@pytest.fixture(scope='module')
def df_file():
    df = pd.read_csv('/data/hms/train.csv')
    return df

@pytest.fixture(scope='module')
def eegs_file():
    eegs = np.load('/data/hms/eegs.npy', allow_pickle=True).item()
    return eegs

def test_EEGDataset(df_file, eegs_file):
    """
    Test EEGDataset class.
    """
    # Load data
    dataset = EEGDataset(df_file, eegs_file)
    assert len(dataset) == 106800
    assert dataset[0]['X'].shape == (2000, 8)


def test_convert_to_hf_DS(df_file, eegs_file):
    """
    Test convert_to_hf_DS function.
    """
    # Load data
    dataset = EEGDataset(df_file, eegs_file)
    hf_dataset = Dataset.from_list(dataset)