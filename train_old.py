import gc
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn
from omegaconf import DictConfig


from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List
import hydra

from utils.util import get_logger, paths, seed_everything, eeg_from_parquet, butter_lowpass_filter, timeSince, AverageMeter
from EEGDatasets import EEGDataset
from model import CustomModel
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

from sklearn.model_selection import KFold, GroupKFold

# from transformers import PatchTSMixerConfig, PatchTSMixerForTimeSeriesClassificatio

def train_epoch(train_loader, model, optimizer, epoch, scheduler, device, config):
    """One epoch training pass."""
    model.train()
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    
    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, batch in enumerate(tqdm_train_loader):
            X = batch.pop("X").to(device) # send inputs to `device`
            y = batch.pop("y").to(device) # send labels to `device`
            batch_size = y.size(0)
            with torch.cuda.amp.autocast(enabled=config.AMP):
                y_preds = model(X)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            
            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                scheduler.step()
            end = time.time()

            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch+1, step, len(train_loader), 
                              remain=timeSince(start, float(step+1)/len(train_loader)),
                              loss=losses,
                              grad_norm=grad_norm,
                              lr=scheduler.get_last_lr()[0]))

    return losses.avg


def valid_epoch(valid_loader, model, device, config):
    model.eval() 
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    prediction_dict = {}
    preds = []
    start = end = time.time()
    criterion = nn.KLDivLoss(reduction="batchmean")
    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
        for step, batch in enumerate(tqdm_valid_loader):
            X = batch.pop("X").to(device) 
            y = batch.pop("y").to(device)
            batch_size = y.size(0)
            with torch.no_grad():
                y_preds = model(X)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy()) 
            end = time.time()

            # ========== LOG INFO ==========
            if step % config.PRINT_FREQ == 0 or step == (len(valid_loader)-1):
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      .format(step, len(valid_loader),
                              remain=timeSince(start, float(step+1)/len(valid_loader)),
                              loss=losses))
                
    prediction_dict["predictions"] = np.concatenate(preds)
    return losses.avg, prediction_dict


## 각 폴드별로 학습을 진행하는 함수
def train_loop(df, fold, config, device, LOGGER, target_preds, all_eegs):
    
    LOGGER.info(f"========== Fold: {fold} training ==========")

    # ======== SPLIT ==========
    train_folds = df[df['fold'] != fold].reset_index(drop=True)
    valid_folds = df[df['fold'] == fold].reset_index(drop=True)
    
    # ======== DATASETS ==========
    train_dataset = CustomDataset(train_folds, config, mode="train", eegs = all_eegs)
    valid_dataset = CustomDataset(valid_folds, config, mode="train", eegs = all_eegs)
    
    # ======== DATALOADERS ==========
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_VALID,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
    
    # ======== MODEL ==========
    # model = CustomModel()
    # model.to(device)c
    from transformers import PatchTSTConfig, PatchTSTForClassification

    # classification task with two input channel2 and 3 classes
    config = PatchTSTConfig(
        num_input_channels=8,
        num_targets=6,
        context_length=2000,
        patch_length=12,
        stride=12,
        use_cls_token=True,
        WEIGHT_DECAY=1e-2,
        EPOCHS = 10,
        AMP = True
    )
    # Initializing a default PatchTSMixer configuration
    model = PatchTSTForClassification(config=config)
    model.to(device)
    # Randomly initializing a model (with random weights) from the configuration
    patch_length = 8

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )

    # ======= LOSS ==========
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    best_loss = np.inf
    # ====== ITERATE EPOCHS ========
    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # ======= TRAIN ==========
        avg_train_loss = train_epoch(train_loader, model, optimizer, epoch, scheduler, device, config)

        # ======= EVALUATION ==========
        avg_val_loss, prediction_dict = valid_epoch(valid_loader, model, device, config)
        predictions = prediction_dict["predictions"]
        
        # ======= SCORING ==========
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                         f"./wavenet_fold_{fold}_best.pth")

    predictions = torch.load(f"./wavenet_fold_{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[target_preds] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

def get_result(oof_df):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[label_cols].values)
    preds = torch.tensor(oof_df[target_preds].values)
    preds = F.log_softmax(preds, dim=1)
    result = kl_loss(preds, labels)
    return result

def train_func(config, device, target_preds, LOGGER, train_df, all_eegs):
    
    oof_df = pd.DataFrame()
    for fold in range(config.FOLDS):
        if fold in [0, 1, 2, 3, 4]:
            _oof_df = train_loop(train_df, fold, config, device, LOGGER, target_preds, all_eegs)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== Fold {fold} finished ==========")
    oof_df = oof_df.reset_index(drop=True)
    return oof_df


def new_func():
    train_df, CREATE_EEGS, all_eegs,  eeg_ids = eeg_preprocess()
    visualize = 1
    for i, eeg_id in tqdm(enumerate(eeg_ids)):  
        # Save EEG to Python dictionary of numpy arrays
        eeg_path = paths.TRAIN_EEGS + str(eeg_id) + ".parquet"
        data = eeg_from_parquet(eeg_path, display=i<visualize)              
        all_eegs[eeg_id] = data
        
        if i == visualize:
            if CREATE_EEGS:
                print(f'Processing {train_df.eeg_id.nunique()} eeg parquets... ',end='')
            else:
                print(f'Reading {len(eeg_ids)} eeg NumPys from disk.')
                break
                
    if CREATE_EEGS: 
        np.save('eegs', all_eegs)
    else:
        all_eegs = np.load('/data/hms/eegs.npy',allow_pickle=True).item()
    breakpoint()
    df = pd.read_csv(paths.TRAIN_CSV)
    label_cols = df.columns[-6:]

    train_df = df.groupby('eeg_id')[['patient_id']].agg('first')
    aux = df.groupby('eeg_id')[label_cols].agg('sum') 

    for label in label_cols:
        train_df[label] = aux[label].values
        
    y_data = train_df[label_cols].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train_df[label_cols] = y_data

    aux = df.groupby('eeg_id')[['expert_consensus']].agg('first')
    train_df['target'] = aux

    train_df = train_df.reset_index()
    train_df = train_df.loc[train_df.eeg_id.isin(eeg_ids)]
    return train_df,all_eegs,eeg_ids

def eeg_preprocess():
    train_df = pd.read_csv(paths.TRAIN_CSV)
    label_cols = train_df.columns[-6:]

    eeg_df = pd.read_parquet(paths.TRAIN_EEGS + "100261680.parquet")
    eeg_features = eeg_df.columns

    eeg_features = ['Fp1','T3','C3','O1','Fp2','C4','T4','O2']
    feature_to_index = {x:y for x,y in zip(eeg_features, range(len(eeg_features)))}

    CREATE_EEGS = False
    all_eegs = {}

    eeg_paths = glob(paths.TRAIN_EEGS + "*.parquet")
    eeg_ids = train_df.eeg_id.unique()
    return train_df,CREATE_EEGS,all_eegs,eeg_ids

@hydra.main(config_path='configs', config_name="config")
def train(config: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TARGET_PREDS = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
    LABEL_TO_NUM = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other':5}
    NUM_TO_LABEL = {v: k for k, v in LABEL_TO_NUM.items()}
    LOGGER = get_logger()
    seed_everything(config.train.SEED)

    train_df, all_eegs, eeg_ids = new_func()

    ## ==== train_df

    gkf = GroupKFold(n_splits=config.train.FOLDS)
    for fold, (train_index, valid_index) in enumerate(gkf.split(train_df, train_df.target, train_df.patient_id)):
        train_df.loc[valid_index, "fold"] = int(fold)
    
    frequencies = [1,2,4,8,16][::-1] # frequencies in Hz
    x = [all_eegs[eeg_ids[0]][:,0]] # select one EEG feature

    for frequency in frequencies:
        x.append(butter_lowpass_filter(x[0], cutoff_freq=frequency))
    train_dataset = CustomDataset(train_df, config.train, mode="train", eegs = all_eegs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=config.train.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    output = train_dataset[0]
    X, y = output["X"], output["y"]
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


    if not config.train.TRAIN_FULL_DATA:
        out_df = train_func(config.train, device, TARGET_PREDS, LOGGER, train_df, all_eegs)

        out_df.to_csv('./oof_df.csv', index=False)
    else:
        raise NotImplementedError

    
if __name__ == "__main__":
    train()