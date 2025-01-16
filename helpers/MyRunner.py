import sys
sys.path.append("src\\")

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel
from helpers.BaseRunner import BaseRunner
from models.general import MyModel

class MyRunner(BaseRunner):
        
    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        model.optimizer = self._build_optimizer(model)
        dataset.actions_before_epoch()
        model.train()
        loss_ls = []
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        for batch in tqdm(dl):
            batch = utils.batch_to_gpu(batch, model.device)
            out_dict = model(batch)
            loss = model.loss(out_dict, batch)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            loss_ls.append(loss.detach().cpu().data.numpy())

        return np.mean(loss_ls).item()
    
    def predict(self, dataset: BaseModel.Dataset, save_prediction = False) -> np.ndarray:
        dataset.model.eval()
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        pred = []
        for batch in tqdm(dl):
            batch = utils.batch_to_gpu(batch, dataset.model.device)
            batch_pred = dataset.model.predict(batch)
            pred.extend(batch_pred.cpu().data.numpy())

        return np.array(pred)