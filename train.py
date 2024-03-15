import sys
import os
import gc
import copy
import yaml
import random
import shutil
from time import time
import typing as tp
import pandas as pd
from pathlib import Path
from utils.util import set_random_seed, to_device, get_path_label, get_transforms

from sklearn.model_selection import StratifiedGroupKFold
import torch
from HMSDatasets import HMSHBACSpecDataset
from model import HMSHBACSpecModel
from loss import KLDivLossWithLogits, KLDivLossWithLogitsForVal

from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint


# "val_loss" metric이 높은 상위 10개 checkpoint를 저장 #TODO
checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="loss",
    mode="min",
    dirpath="checkpoints/",
    filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
)
class HMSLitModule(pl.LightningModule):
    def __init__(self, 
                 model, 
                 loss = KLDivLossWithLogits(),
                 learning_rate_scheduler = None, 
                 weight_decay = 1e-6,
                 max_epoch = 10, 
                 log_tool = "wandb",
                 cfg =None,
                 **kwargs):
        super().__init__()
        self.model = model
        self.loss = loss
        self.log_tool = log_tool
        if self.log_tool  =="wandb":
            wandb.init(project='hms', config=dict(cfg))
    
    def training_step(self, batch, batch_idx):
        x, y = batch["data"], batch["target"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("loss", loss)
        wandb.log({"loss": loss})
        return loss
    
    def validation_epoch_end(self, batch, batch_idx):
        x, y = batch["data"], batch["target"]
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        wandb.log({"val_loss": loss})
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        return optimizer

@hydra.main(version_base=None, config_path="configs", config_name="0305")
def main(cfg : DictConfig) -> None:
    
    torch.backends.cudnn.benchmark = True
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device(cfg.device)

    # Data Load
    train_all = pd.read_csv("fold_dataset/train_folds.csv")

    if cfg.dataset == "Spectrogram":
        train_path_label, val_path_label, _, _ = get_path_label(cfg.val_fold, train_all)
        train_transform, val_transform = get_transforms(cfg)
        train_dataset = HMSHBACSpecDataset(**train_path_label, transform=train_transform)
        val_dataset = HMSHBACSpecDataset(**val_path_label, transform=val_transform)
    elif cfg.dataset == "EEG":
        NotImplementedError
    else:
        raise NotImplementedError


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, num_workers=4, shuffle=False, drop_last=False)
    
    # Model Load
    if cfg.model == "EfficientNet":
        breakpoint()
        model = HMSHBACSpecModel(
            model_name=cfg.model_name, pretrained=True, num_classes=6, in_channels=1)
    else:
        raise NotImplementedError

    #### Train
    model.to(device)
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    trainer = pl.Trainer(callbacks=[checkpoint_callback])
    hms_module = HMSLitModule(model = model, cfg= cfg)
    trainer.fit(hms_module, train_loader)
    wandb.finish()
    

if __name__ == "__main__":
    main()

    