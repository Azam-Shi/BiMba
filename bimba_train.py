#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
from network.bimba_vim import VisionMambaWithFeatures
from network.bimba_vim import get_ml_config

from utils.trainer import fit, evaluate_val
from utils.utils import get_processed
from utils.dataset import PrepDataset

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



DATA_DIR = os.getcwd() + '/test_example/data_preparation/'
print(f"Data directory set to: {DATA_DIR}")

TRAIN_LIST_FILE = './BiMba/data/train_list.txt'
VAL_LIST_FILE = './BiMba/data/val-list.txt'

MODEL_NAME = f'bimba'
MODEL_DIR = f'./savedModels/{MODEL_NAME}'

IMG_SIZE = 18
PATCH_SIZE = 3
STRIDE = 3
DEPTH = 8
D_STATE = 8
PATIENCE = 2
BATCH_SIZE = 1
MAX_EPOCH = 5
MARGIN = 0
TEMP = 0.5

FEATURES_SUBSET = list(range(5))  # channels 0..4
N_FEATURES = len(FEATURES_SUBSET)

DROP_PATH_RATE = 0


config = {}
config['dirs'] = {}
config['dirs']['data_prepare'] = DATA_DIR
config['dirs']['patch_info'] = config['dirs']['data_prepare'] + '08-patch_info/'
config['dirs']['grid'] = config['dirs']['data_prepare'] + '07-grid/'
config['dirs']['tmp'] = './tmp'
config['ppi_const'] = {}
config['ppi_const']["max_shape_size"] = 10
config['ppi_const']["radius"] = 9
os.environ["TMP"] = config['dirs']['tmp']
os.environ["TMPDIR"] = config['dirs']['tmp']
os.environ["TEMP"] = config['dirs']['tmp']


# Functions
def initialize_config():
    config = {}
    config['dirs'] = {
        'data_prepare': DATA_DIR,
        'grid': DATA_DIR + '07-grid/',
        'patch_info': DATA_DIR + '08-patch_info/',
        'tmp': './tmp',
    }
    config['ppi_const'] = {'radius': 9}
    os.environ["TMP"] = config['dirs']['tmp']
    os.environ["TMPDIR"] = config['dirs']['tmp']
    os.environ["TEMP"] = config['dirs']['tmp']
    return config

def compute_mean_std(train_list, config):
    grid_native_list = []
    for ppi in train_list:
        grid_path = f"{config['dirs']['grid']}/{ppi}"
        if os.path.exists(grid_path):
            grid_native_list.append(ppi)

    print(f"Loaded {len(grid_native_list)} proteins/examples")
    all_grid = np.stack(grid_native_list, axis=0)
    
    return all_grid


class Bsite_proto(nn.Module):
    def __init__(self, config, img_size=18, num_classes=2, zero_head=False):
        super(Bsite_proto, self).__init__()
    
        self.img_size = img_size
        self.num_classes = num_classes
        self.zero_head = zero_head        

        
        self.vim = VisionMambaWithFeatures(
                                              config=config,
                 img_size=18, 
                 patch_size=3, 
                 stride=3,
                 depth=8, 
                 embed_dim=config.hidden_size,
                 d_state=8, 
                 channels=5, 
                 num_classes=2,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon = 1e-5, 
                 rms_norm = True, 
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=True,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=True,
                 final_pool_type='none',
                 if_abs_pos_embed=True,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="v2",
                 if_cls_token=True,
                 if_divide_out=True,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
        )
    

    def forward(self, img, res_feats, labels=None):
        x = self.vim(img, res_feats)
 
        if labels is not None:
            focal_loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
            loss = focal_loss_fn(x, labels)
            return x, loss
        else:
            return x
       
        
      
      
def train_Bsite(search_space, train_list, val_list, IMG_SIZE, PATIENCE, MODEL_DIR,
                 MODEL_NAME, patched_dir, pos_grid_dir, MAX_EPOCHS=150, N_FEATURES=5, 
                 feature_subset=None, disable_tqdm=True, print_summary=False, data_prepare_dir = DATA_DIR):
    train_db = PrepDataset(
        train_list, training_mode=True, feature_subset=feature_subset,
        data_prepare_dir=data_prepare_dir)
    val_db = PrepDataset(
        val_list, training_mode=False, feature_subset=feature_subset,
        data_prepare_dir=data_prepare_dir)

    trainloader = DataLoader(train_db, batch_size=1, shuffle=True, pin_memory=True)
    valloader = DataLoader(val_db, batch_size=1, shuffle=False, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = get_ml_config(search_space)
    model = Bsite_proto(
        model_config, img_size=IMG_SIZE, num_classes=2
    )
    model = model.to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=search_space['lr'], weight_decay=search_space['weight_decay'])
    model, history, saved_index = fit(
        MAX_EPOCHS, model, trainloader, valloader, optimizer, model_name=MODEL_NAME,
        image_size=IMG_SIZE, channels=N_FEATURES, device=device, save_model=True,
        saved_model_dir=MODEL_DIR, patience=PATIENCE, print_summary=print_summary,
        disable_tqdm=disable_tqdm)    
    val_loss, val_auc = evaluate_val(valloader, model, device)
    return model, history, saved_index



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.float().view(-1).to(logits.device)

        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p  = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)

        alpha_t = torch.where(
            targets == 1,
            torch.as_tensor(self.alpha, device=logits.device),
            torch.as_tensor(1 - self.alpha, device=logits.device)
        )

        loss = alpha_t * (1 - pt).pow(self.gamma) * ce
        return loss.mean() if self.reduction == 'mean' else (loss.sum() if self.reduction == 'sum' else loss)


def main():
    config = initialize_config()
    train_list = [x.strip('\n') for x in open(TRAIN_LIST_FILE, 'r').readlines()]
    val_list = [x.strip('\n') for x in open(VAL_LIST_FILE, 'r').readlines()]
    train_list_updated = get_processed(train_list, config)
    val_list_updated = get_processed(val_list, config)
    params = {'hidden_size': 128, 'dropout': 0, 'lr': 0.0001,
              'neg_pos_ratio': 1, 'patch_size': 3,
              'weight_decay': 0.0001, 'margin': MARGIN, 'temperature': TEMP}

    model, history, saved_index = train_Bsite(
        params, train_list=train_list_updated, val_list=val_list_updated,
        IMG_SIZE=IMG_SIZE, PATIENCE=PATIENCE, MODEL_NAME=MODEL_NAME, MAX_EPOCHS=MAX_EPOCH,
        N_FEATURES=N_FEATURES, MODEL_DIR=MODEL_DIR, patched_dir=config['dirs']['grid'], 
        pos_grid_dir=config['dirs']['grid'], feature_subset=FEATURES_SUBSET)
        

if __name__ == "__main__":
    main()

