import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from ray import tune
import json
import time
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt
from datetime import datetime

def get_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def set_device(model, device_ids, device):
    if device_ids is None and device is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
        model = model.to(device, non_blocking=False)
    elif device is not None:
        model = model.to(device, non_blocking=False)
    elif device_ids is not None:
        print("Setting up the following GPUs: {}".format(device_ids))
        device=torch.device("cuda:{}".format(device_ids[0]))
        model = nn.DataParallel(model, device_ids=device_ids).to(device, non_blocking=False)
    return model, device

def label_to_tensor(label, n_classes):
    label_tensor = torch.zeros(label.shape[0], n_classes)
    for batch_i in range(label.shape[0]):
        for label_i in range(n_classes):
            label_tensor[batch_i][label_i]=label[batch_i]==label_i
    return label_tensor.long()

def add_to_history(history, train_loss, val_loss,train_auc, val_auc):
    history['train_loss'].append(float(train_loss))
    history['val_loss'].append(float(val_loss))
    history['train_auc'].append(float(train_auc))
    history['val_auc'].append(float(val_auc))
    return history

def compute_performance(logits, label, epoch=None, split=None, csv_path=None, save_every=30):
    probs = torch.sigmoid(logits).cpu().detach().numpy()
    label = label.detach().cpu().numpy()
    auc = metrics.roc_auc_score(label, probs)
    return auc


def plot_metrics(history, saved_model_dir, model_name):
    plt.style.use('ggplot')
    figures_dir = saved_model_dir + '/' + model_name +'_figs/'
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)
    plt.plot(history['train_loss'], marker='.', color='b', label='Train loss')
    plt.plot(history['val_loss'], marker='.', color='r', label='Validation loss')
    plt.legend(loc="upper right")
    plt.savefig(figures_dir+'/loss_{}.png'.format(model_name))
    plt.clf()
    plt.plot(history['train_auc'], marker='.', color='b', label='Train AUC')
    plt.plot(history['val_auc'], marker='.', color='r', label='Validation AUC')
    plt.legend(loc="lower right")
    plt.savefig(figures_dir+'/auc_{}.png'.format(model_name))


def evaluate_val(loader, model, device, criterion=None,epoch=None, preds_csv_path=None):
    batch_losses = []
    batch_aucs = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            image_tiles, label, ppi, res_feats = data
            image_tiles = np.reshape(image_tiles, (image_tiles.shape[0]*image_tiles.shape[1], image_tiles.shape[2], image_tiles.shape[3], image_tiles.shape[4]))
            res_feats = np.reshape(res_feats, (res_feats.shape[0] * res_feats.shape[1], res_feats.shape[2]))
            label = np.reshape(label, (label.shape[0]*label.shape[1]))
            image = image_tiles.to(device, non_blocking=False)
            label = label.to(device, non_blocking=False)
            res_feats = res_feats.to(device, non_blocking=False)

            output = model(image, res_feats, label)
            output, loss = output        
            batch_losses.append(loss.item())
            batch_auc = compute_performance(
                output.detach().cpu(), label.detach().cpu(),
                epoch=epoch, split="val", csv_path=preds_csv_path
            )
            batch_aucs.append(batch_auc)

    val_loss = np.mean(batch_losses)
    val_auc = np.mean(batch_aucs)
    return val_loss, val_auc

def train_one_epoch(model, train_loader, device, optimizer,
                    disable_tqdm=False, epoch=None, preds_csv_path=None):
    batch_losses = []
    batch_aucs = []
    model.train()
    total_patches_seen = 0
    epoch_bar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True, disable=disable_tqdm)

    for i, data in epoch_bar:
        image_tiles, label, ppi, res_feats = data  # (1, N, C, H, W), (1, N)
        image_tiles = np.reshape(image_tiles, (image_tiles.shape[0] * image_tiles.shape[1], image_tiles.shape[2], image_tiles.shape[3], image_tiles.shape[4]))
        res_feats = np.reshape(res_feats, (res_feats.shape[0] * res_feats.shape[1], res_feats.shape[2]))
        label = np.reshape(label, (label.shape[0] * label.shape[1]))   
        
        total_patches_seen += image_tiles.shape[0]
        image = image_tiles.to(device=device, dtype=torch.float)
        label = label.to(device)
        res_feats = res_feats.to(device)
        output = model(image,res_feats, label)
        output, loss = output        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        batch_losses.append(loss.item())
        batch_auc = compute_performance(
            output.detach().cpu(), label.detach().cpu(),
            epoch=epoch, split="train", csv_path=preds_csv_path
        )
        batch_aucs.append(batch_auc)       
        epoch_bar.set_description(f"Epoch Progress [Patches: {total_patches_seen}]")
    
    model.eval()
    train_loss = np.mean(batch_losses)
    train_auc = np.mean(batch_aucs)
    print(f"Average training loss: {train_loss:.4f}; train AUC: {train_auc:.4f};")
    return model, train_loss, train_auc
    
   
def fit(epochs, model, train_loader, val_loader, optimizer, model_name='default', image_size=18, channels=5, device_ids = None,
        device=None, saved_model_dir='./savedModels/', save_model=True, print_summary = True, patience=10, raytune=False,
        disable_tqdm=False):
    """
    :param epochs:
    :param model:
    :param train_loader:
    :param val_loader:
    :param optimizer:
    :param model_name:
    :param image_size:
    :param channels:
    :param device_ids:
    :param device:
    :param saved_model_dir:
    :param save_model:
    :param print_summary:
    :param patience:
    :return:
    """
    start = time.time()
    if not os.path.exists(saved_model_dir):
        os.mkdir(saved_model_dir)

    preds_csv_path = os.path.join(saved_model_dir, f"{model_name}_epoch_preds.csv")
    history = {'train_loss': [], 'val_loss': [],
               'train_auc':[], 'val_auc':[]
               }
    model, device = set_device(model, device_ids, device)
    print("Start training {} model.".format(model_name))
    if print_summary:
        print("Model's Summary")
        summary(model, torch.rand((1, channels, image_size, image_size)).to(device, non_blocking=False))

    min_loss = np.inf
    max_auc = 0
    decrease = 0
    not_improved = 0
    saved_index = 0

    for e in range(epochs):
        print("[{}] Starting training for epoch {}...".format(get_date(), e))
        model, train_loss, train_auc = train_one_epoch(
            model, train_loader, device, optimizer,
            disable_tqdm=disable_tqdm,
            epoch=e, preds_csv_path=preds_csv_path
        )
        val_loss, val_auc = evaluate_val(
            val_loader, model, device, 
            epoch=e, preds_csv_path=preds_csv_path
        )

        print("Average val loss: {}; val AUC: {};".format(val_loss, val_auc))
        if val_auc > max_auc:
            print('AUC increasing.. {:.4f} >> {:.4f} '.format(max_auc, val_auc))
            max_auc = val_auc
            decrease += 1
            print('Saving model on epoch {}...'.format(e))
            torch.save(model.state_dict(), saved_model_dir + '/{}.pth'.format(model_name))
            saved_index = e
            not_improved = 0
            if raytune:
                tune.report(score=float(val_auc))
        else:
            not_improved += 1
            print("Model did not improve {} times...".format(not_improved))
        history = add_to_history(history, train_loss, val_loss, train_auc, val_auc)
        if not_improved == patience:
            print("Stopping training...")
            break
    model.load_state_dict(torch.load(saved_model_dir + '/{}.pth'.format(model_name)))

    print("[{}] Done with training.".format(get_date()))
    print("The model was saved at the {} epoch.".format(saved_index))
    print("Total training time: {} seconds.".format(time.time() - start))
    plot_metrics(history, saved_model_dir, model_name)
    try:
        with open(saved_model_dir + '/history.json', 'w') as outfile:
            json.dump(history, outfile, indent=4)
            outfile.flush() 
            os.fsync(outfile.fileno())
    except Exception as ex:
        print(f"Warning: Could not save history due to error: {ex}")
    return model, history, saved_index

def get_processed(alist, config):
    pos_dir = config['dirs']['grid']
    processed_list = []
    for ppi in alist:
        if os.path.exists(pos_dir+ppi+'.npy'):
            processed_list.append(ppi)
    return processed_list
