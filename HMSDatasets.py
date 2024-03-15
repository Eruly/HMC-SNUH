"""
Load EEG dataset from kaggle like below:

Dataset 1. EEG signal (Time Series)

- TRAIN
    - train.csv
    - train_eegs
        - train_eegs/0.parquet
        - train_eegs/1.parquet
        - train_eegs/2.parquet
        - ...
        - train_eegs/106799.parquet
- TEST
    - test.csv
    - test_eegs
        - test_eegs/0.parquet
        ...

Dataset 2. Kaggle Spectrogram (Image) 
- TRAIN
    - train.csv
    - train_spectrograms
        - train_spectrograms/0.npy
        - train_spectrograms/1.npy
        - train_spectrograms/2.npy
        - ...
        - train_spectrograms/106799.npy

Dataset 3. User made Spectrogram (Image)
- TRAIN
    - train.csv
    - train_spectrograms

Dataset 4. 19 channel EEG signal (Time Series)?

"""

from torch.utils.data import DataLoader, Dataset
from typing import Dict, List
import numpy as np
import pandas as pd
from utils.util import butter_lowpass_filter
import torch
import doctest
import glob

label_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
eeg_features = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
feature_to_index = {x:y for x,y in zip(eeg_features, range(len(eeg_features)))}

class FullDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        eegs: Dict[int, np.ndarray],
        specs: Dict[int, np.ndarray],
        eeg_specs: Dict[int, np.ndarray],
        input_list: List[str] = ["eegs"],
        augment: bool = False, 
        mode: str = 'train',
    ): 
        self.df = df
        self.augment = augment
        self.mode = mode
        self.eegs = eegs
        self.spectograms = specs
        self.eeg_spectograms = eeg_specs
        self.input_list = input_list
        
    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return len(self.df)
        
    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        eegs, spectrogram, eegs_spectrograms, y = self.__data_generation(index)
        inputs =[]
        for input in self.input_list:
            if input == 'spectrogram':
                X = spectrogram
            elif input == 'eegs spectrograms':
                X=eegs_spectrograms
            elif input == 'eegs':
                X = eegs
            else:
                raise ValueError(f"input_list should be one of ['spectrogram', 'eegs spectrograms', 'eegs']")
            # if self.augment:
            #     X = self.__transform(X) 
            #     X = torch.tensor(X, dtype=torch.float32)
            inputs.append(X)

        return *inputs, torch.tensor(y, dtype=torch.float32)
                        
    def __data_generation(self, index):
        """
        Generates data containing batch_size samples.
        """
        X1 = np.zeros((128, 256, 8), dtype='float32')
        y = np.zeros(6, dtype='float32')
        img = np.ones((128,256), dtype='float32')
        row = self.df.iloc[index]
        if self.mode=='test': 
            r = 0
        else: 
            r = int((row['spectrogram_label_offset_seconds_min'] + row['spectrogram_label_offset_seconds_max']) // 4)
            
        for region in range(4):
            img = self.spectograms[row.spectrogram_id][r:r+300, region*100:(region+1)*100].T
            X1[14:-14, :, region] = img[:, 22:-22] / 2.0
            
    
        X2 = self.eeg_spectograms[row.eeg_id]
        
        X0 = np.zeros((10_000, 8), dtype='float32')
        data = self.eegs[row.eeg_id]

        # === Feature engineering ===
        X0[:,0] = data[:,feature_to_index['Fp1']] - data[:,feature_to_index['T3']]
        X0[:,1] = data[:,feature_to_index['T3']] - data[:,feature_to_index['O1']]

        X0[:,2] = data[:,feature_to_index['Fp1']] - data[:,feature_to_index['C3']]
        X0[:,3] = data[:,feature_to_index['C3']] - data[:,feature_to_index['O1']]

        X0[:,4] = data[:,feature_to_index['Fp2']] - data[:,feature_to_index['C4']]
        X0[:,5] = data[:,feature_to_index['C4']] - data[:,feature_to_index['O2']]

        X0[:,6] = data[:,feature_to_index['Fp2']] - data[:,feature_to_index['T4']]
        X0[:,7] = data[:,feature_to_index['T4']] - data[:,feature_to_index['O2']]

        # === Standarize ===
        X0 = np.clip(X0,-1024, 1024)
        X0 = np.nan_to_num(X0, nan=0) / 32.0

        # # === Butter Low-pass Filter ===
        # X0 = butter_lowpass_filter(X0) # TODO 필요한가의 논의?
    


        if self.mode != 'test':
            y = row[label_cols].values.astype(np.float32)
            
        return X0, X1, X2, y
    
    def __transform(self, img):
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
        ])
        return transforms(image=img)['image']

eeg_features = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
feature_to_index = {x:y for x,y in zip(eeg_features, range(len(eeg_features)))}

class EEGDataset(Dataset):
    def __init__(
        self, 
        eeg_dataframe : pd.DataFrame, # -> csv file 위치로
        eegs: Dict[int, np.ndarray], # -> npy file 위치로 
        mode: str = 'train',
        batch_size : int = 32, 
        downsample: int = 5 # TODO downsample을 어떻게 할지 논의 필요
    ): 
        """
        EEG Dataset from kaggle
        
        eeg_df: DataFrame : train.csv ( 예시 : )
        mode: Optional[str] : 'train' or 'test'
        eegs: Dict[int, np.ndarray]
        downsample: int
        ---
        >>> df = pd.read_csv('/data/hms/train.csv') # TODO test dataframe을 만드어주긴 해야할 듯
        >>> eegs = np.load('/data/hms/eegs.npy',allow_pickle=True).item()
        >>> dataset = EEGDataset(eeg_dataframe= df, eegs=eegs)
        """
        self.eeg_df =  eeg_dataframe
        self.eegs = eegs
        if mode not in ['train', 'test']:
            raise ValueError("mode should be 'train' or 'test'")
        self.mode = mode
        self.batch_size = batch_size
        self.downsample = downsample
        
    def __len__(self):
        """
        Length of dataset.
        ---
        >>> df = pd.read_csv('/data/hms/train.csv')
        >>> eegs = np.load('/data/hms/eegs.npy',allow_pickle=True).item()
        >>> dataset = EEGDataset(eeg_dataframe = df, eegs=eegs)
        >>> len(dataset) # 현재는 train.csv 길이
        106800
        """
        return len(self.eeg_df)
        
    def __getitem__(self, index):
        """
        Get one item.
        ---
        >>> df = pd.read_csv('/data/hms/train.csv')
        >>> eegs = np.load('/data/hms/eegs.npy',allow_pickle=True).item()
        >>> dataset = EEGDataset(eeg_dataframe = df, eegs=eegs)
        >>> dataset[0].shape
        torch.Size([2000, 8])
        """
        X, y = self.__data_generation(index)
        X = X[::self.downsample,:] 
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        
        return output
                        
    def __data_generation(self, index):
        """
        데이터 전처리와 feature engineering를 수행하는 함수

        전처리 목차
        1. Featrue engineering
            X0 : Fp1 - T3
            X1 : T3 - O1
            X2 : Fp1 - C3
            X3 : C3 - O1
            X4 : Fp2 - C4
            X5 : C4 - O2
            X6 : Fp2 - T4
            X7 : T4 - O2
        2. Standarize
        3. Butter Low-pass Filter (# TODO 필요한가의 논의?)

        index: int
        """
        row = self.eeg_df.iloc[index]
        X = np.zeros((10_000, 8), dtype='float32')
        y = np.zeros(6, dtype='float32')
        data = self.eegs[row.eeg_id]

        # === Feature engineering ===
        X[:,0] = data[:,feature_to_index['Fp1']] - data[:,feature_to_index['T3']]
        X[:,1] = data[:,feature_to_index['T3']] - data[:,feature_to_index['O1']]

        X[:,2] = data[:,feature_to_index['Fp1']] - data[:,feature_to_index['C3']]
        X[:,3] = data[:,feature_to_index['C3']] - data[:,feature_to_index['O1']]

        X[:,4] = data[:,feature_to_index['Fp2']] - data[:,feature_to_index['C4']]
        X[:,5] = data[:,feature_to_index['C4']] - data[:,feature_to_index['O2']]

        X[:,6] = data[:,feature_to_index['Fp2']] - data[:,feature_to_index['T4']]
        X[:,7] = data[:,feature_to_index['T4']] - data[:,feature_to_index['O2']]

        # === Standarize ===
        X = np.clip(X,-1024, 1024)
        X = np.nan_to_num(X, nan=0) / 32.0

        # === Butter Low-pass Filter ===
        X = butter_lowpass_filter(X) # TODO 필요한가의 논의?
        

        if self.mode != 'test':
            label_cols =  ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
            y = row[label_cols].values.astype(np.float32)
            
        return X, y




"""
DATASET2



"""
target_p = [
    'seizure_vote', 
    'lpd_vote', 
    'gpd_vote', 
    'lrda_vote', 
    'grda_vote', 
    'other_vote'
]
features = [
    'Fp1','T3','C3','O1','Fp2','C4','T4','O2'
]


import typing as tp
from pathlib import Path
import albumentations as A
FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]

class HMSHBACSpecDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        transform: A.Compose,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img = np.load(img_path)  # shape: (Hz, Time) = (400, 300)
        
        # log transform
        img = np.clip(img,np.exp(-4), np.exp(8))
        img = np.log(img)
        
        # normalize per image
        eps = 1e-6
        img_mean = img.mean(axis=(0, 1))
        img = img - img_mean
        img_std = img.std(axis=(0, 1))
        img = img / (img_std + eps)

        img = img[..., None] # shape: (Hz, Time) -> (Hz, Time, Channel)
        img = self._apply_transform(img)

        return {"data": img, "target": label}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img
    
###
# DATASET 3
#
###
SPEC_SIZE  = (512, 512, 3)
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, data, batch_size=32, shuffle=False, mode='train'):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples."""
        # Initialization
        X = np.zeros((len(indexes), *SPEC_SIZE), dtype='float32')
        y = np.zeros((len(indexes), len(CLASSES)), dtype='float32')

        # Generate data
        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            eeg_id = row['eeg_id']
            spec_offset = int(row['spectrogram_label_offset_seconds_min'])
            eeg_offset = int(row['eeg_label_offset_seconds_min'])
            file_path = f'/kaggle/input/3-diff-time-specs-hms/images/{eeg_id}_{spec_offset}_{eeg_offset}.npz'
            data = np.load(file_path)
            eeg_data = data['final_image']
            eeg_data_expanded = np.repeat(eeg_data[:, :, np.newaxis], 3, axis=2)

            X[j] = eeg_data_expanded
            if self.mode != 'test':
                y[j] = row[CLASSES]

        return X, y


if __name__ == "__main__":
    import omegaconf
    cfg = omegaconf.OmegaConf.load("/home/ubuntu/HMC-SNUH/configs/datasets/init.yaml")
    TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
    df = pd.read_csv("/data/SWB_Contribute/Data/original_data/train.csv")
    # Create a new identifier combining multiple columns
    id_cols = ['eeg_id', 'spectrogram_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    df['new_id'] = df[id_cols].astype(str).agg('_'.join, axis=1)
    
    # Calculate the sum of votes for each class
    df['sum_votes'] = df[TARGETS].sum(axis=1)
    
    # Group the data by the new identifier and aggregate various features
    agg_functions = {
        'eeg_id': 'first',
        'eeg_label_offset_seconds': ['min', 'max'],
        'spectrogram_label_offset_seconds': ['min', 'max'],
        'spectrogram_id': 'first',
        'patient_id': 'first',
        'expert_consensus': 'first',
        **{col: 'sum' for col in TARGETS},
        'sum_votes': 'sum',
    }
    grouped_df = df.groupby('new_id').agg(agg_functions).reset_index()

    # Flatten the MultiIndex columns and adjust column names
    grouped_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped_df.columns]
    grouped_df.columns = grouped_df.columns.str.replace('_first', '').str.replace('_sum', '')
    
    # Normalize the class columns
    y_data = grouped_df[TARGETS].values
    y_data_normalized = y_data / y_data.sum(axis=1, keepdims=True)
    grouped_df[TARGETS] = y_data_normalized

    # Split the dataset into high and low quality based on the sum of votes
    high_quality_df = grouped_df[grouped_df['sum_votes'] >= 10].reset_index(drop=True)
    low_quality_df = grouped_df[(grouped_df['sum_votes'] < 10) & (grouped_df['sum_votes'] >= 0)].reset_index(drop=True)
    train = grouped_df
    # load files
    raw_eegs = np.load(cfg.PRE_LOADED_RAW_EEGS, allow_pickle=True).item()
    print(f"Length of eegs: {len(raw_eegs)}")

    READ_SPEC_FILES = False
    paths_spectrograms = glob.glob(cfg.TRAIN_SPECTOGRAMS + "*.parquet")
    print(f'There are {len(paths_spectrograms)} spectrogram parquets')

    all_spectrograms = np.load(cfg.PRE_LOADED_SPECTROGRAMS, allow_pickle=True).item()

    READ_EEG_SPEC_FILES = False
    paths_eegs = glob.glob(cfg.TRAIN_EEG_SPECTROGRAM + "*.npy")
    print(f'There are {len(paths_eegs)} EEG spectrograms')

    all_eegs = np.load(cfg.PRE_LOADED_EEG_SPECTROGRAMS, allow_pickle=True).item()

    print(f"Shape of Kaggle spectrograms: {all_spectrograms[train.loc[0,'spectrogram_id']].shape}")
    print(f"Shape of EEG spectrograms: {all_eegs[train.loc[0,'eeg_id']].shape}")
    print(f"Shape of raw eegs: {raw_eegs[train.loc[0,'eeg_id']].shape}")

    # Datasets
    train_dataset = FullDataset(train, 
                                eegs = raw_eegs,
                                specs = all_spectrograms,
                                eeg_specs = all_eegs,
                                mode="train")
    X0, X1,X2, y = train_dataset[0]
    print(f"X0 shape: {X1.shape}")
    print(f"X1 shape: {X1.shape}")
    print(f"X2 shape: {X2.shape}")
    print(f"y shape: {y.shape}")