import torch
from datetime import datetime, timedelta
from time import time
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
import os
import csv
import shutil
import argparse
import sys
import time
import warnings
import numpy as np
import pandas as pd
from random import sample
from sklearn import metrics
from datetime import datetime
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

#from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
# from dataset.dataset_finetune import collate_pool, get_train_val_test_loader
#from model.cgcnn_finetune import CrystalGraphConvNet

## Hyperparameter optimization
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def train_test_plot(y_train, y_train_pred, y_test, y_test_pred, write_dir):
    plt.rcParams['axes.linewidth'] = 1.5
    fig_train = plt.figure(figsize=[6.4, 6.4])
    ax = fig_train.gca()
    corr_coeff, p_value = pearsonr(y_train, y_train_pred)
    ax_min = min(min(y_train), min(y_train_pred))
    ax_max = max(max(y_train), max(y_train_pred))
    ax_min = int(ax_min)
    ax_max = int(ax_max)
    parity = np.linspace(ax_min, ax_max)
    ax.scatter(y_train, y_train_pred, marker='o', color=(0, 0, 1, 0.5))
    ax.set_xlabel(f'Actual KVRH', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_ylabel(f'Predicted KVRH', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_title(f'Training: R = {format(corr_coeff, ".2")}', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.plot(parity, parity, color=(1, 0, 0, 0.5), linewidth=1.5)
    plt.xticks(fontsize=12, fontweight='bold', family='Helvetica')
    plt.yticks(fontsize=12, fontweight='bold', family='Helvetica')
    plt.tick_params(axis='both', direction='in', width=1, length=6)
    #plt.show()
    fig_train.savefig(f'{write_dir}train.pdf', dpi=600, bbox_inches='tight')

    fig_test = plt.figure(figsize=[6.4, 6.4])
    ax = fig_test.gca()
    corr_coeff, p_value = pearsonr(y_test, y_test_pred)
    ax_min = min(min(y_test), min(y_test_pred))
    ax_max = max(max(y_test), max(y_test_pred))
    ax_min = int(ax_min)
    ax_max = int(ax_max)
    parity = np.linspace(ax_min, ax_max)
    ax.scatter(y_test, y_test_pred, marker='o', color=(0, 0, 1, 0.5))
    ax.set_xlabel(f'Actual KVRH', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_ylabel(f'Predicted KVRH', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_title(f'Test: R = {format(corr_coeff, ".2")}', fontsize=14, fontweight='bold', family='Helvetica')
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.plot(parity, parity, color=(1, 0, 0, 0.5), linewidth=2)
    plt.xticks(fontsize=12, fontweight='bold', family='Helvetica')
    plt.yticks(fontsize=12, fontweight='bold', family='Helvetica')
    plt.tick_params(axis='both', direction='in', width=1, length=6)
    #plt.show()
    fig_test.savefig(f'{write_dir}test.pdf', dpi=600, bbox_inches='tight')


class AverageMeter:
    """Keeps track of the average, sum, and count for a metric."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Resets all statistics."""
        self.val = 0          # Current value
        self.avg = 0          # Average value
        self.sum = 0          # Sum of all values
        self.count = 0        # Number of updates
    
    def update(self, val, n=1):
        """Updates the statistics with a new value and weight.
        
        Args:
            val (float): New value to add.
            n (int): Weight of the value (e.g., batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))

class NeuralNetwork(object):
    def __init__(self, args, train_loader, val_loader, test_loader, model_save, validation, model_save_path=None):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.test_loader = test_loader
        self.model_save= model_save
        self.validation = validation
        self.model_save_path = model_save_path

        self.criterion = nn.MSELoss()
        
        depth = args['depth']
        width = args['width']
        
        for bn, (inputs, target) in enumerate(self.train_loader):
            input_size = inputs.size(-1)
            break
       
        layers = []
        
        for i in range(depth):
            layers.append(nn.Linear(input_size if i == 0 else width, width))
            if self.args['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif self.args['activation'] == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise NameError('Only ReLu or Tanh is allowed as activation function') 
            layers.append(nn.Dropout(self.args['dropout']))
            
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        

    def train(self):
        
        model = self.network.to(self.device)

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self.network.parameters(), lr = self.args['alpha'], weight_decay=self.args['weight_decay'])

        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.network.parameters(), lr = self.args['alpha'], weight_decay=self.args['weight_decay']
            )
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')        
        

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_mae = np.inf
        patience = self.args['early_stopping']
        valid_counter = 0
        model.train()
        
        epoch_list = []
        train_loss_list = []
        valid_loss_list = []
        
        for epoch_counter in range(10000):
            epoch_list.append(epoch_counter+1)
            train_losses = AverageMeter()
            for bn, (inputs, target) in enumerate(self.train_loader):
            
                input_var = inputs.to(self.device)                
               
                target_var = target.unsqueeze(1).to(self.device)

                # compute output
                output = model(input_var)

                loss = self.criterion(output, target_var)
                train_losses.update(loss.data.cpu().item(), target_var.size(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1
                
            train_loss_list.append(train_losses.avg)

            # validate the model requested
            if self.validation:
                valid_loss, valid_mae = self._validate(model, self.valid_loader, epoch_counter)
                valid_loss_list.append(valid_loss) 

                if valid_loss < best_valid_loss:
            
                    best_valid_loss = valid_loss
                    
                else:
                    valid_counter += 1

                if valid_counter == patience:
                    break
                    
                valid_n_iter += 1
        
        if self.model_save:
            torch.save(model.state_dict(), os.path.join(self.model_save_path, f"best_ANN_model.pth"))
        self.model = model
        self.epoch_list = epoch_list
        self.train_loss_list = train_loss_list
        self.valid_loss_list = valid_loss_list

        return best_valid_loss
           

    def _validate(self, model, valid_loader, n_epoch):
        losses = AverageMeter()
        mae_errors = AverageMeter()

        with torch.no_grad():
            model.eval()
            for bn, (inputs, target) in enumerate(valid_loader):
        
                input_var = inputs.to(self.device)
                
        
                target_var = target.unsqueeze(1).to(self.device)

                # compute output
                output = model(input_var)
        
                loss = self.criterion(output, target_var)

                mae_error = mae(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

        model.train()

        return losses.avg, mae_errors.avg

    def train_results(self, model_path, results_save_path, train_mofs):
        
        print(model_path)
        state_dict = torch.load(os.path.join(model_path, 'best_ANN_model.pth'), map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        losses = AverageMeter()
        mae_errors = AverageMeter()

        train_targets = []
        train_preds = []

        with torch.no_grad():
            self.model.eval()
            for bn, (inputs, target) in enumerate(self.train_loader):

                input_var = inputs.to(self.device)
        
                target_var = target.unsqueeze(1).to(self.device)

                # compute output
                output = self.model(input_var)

                loss = self.criterion(output, target_var)

                mae_error = mae(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

                train_pred = output.data.cpu()
                train_target = target
                train_preds += train_pred.view(-1).tolist()
                train_targets += train_target.view(-1).tolist()


        with open(os.path.join(results_save_path, "train_results.csv"), 'w') as f:
            writer = csv.writer(f)
            for mofs, target, pred in zip(train_mofs, train_targets, train_preds):
                writer.writerow((mofs, target, pred))

        self.model.train()

        return train_preds, train_targets, losses.avg, mae_errors.avg
    
    def test(self, model_path, results_save_path, test_mofs):
        # test steps
        print('Test on test set')
        print(model_path)
        state_dict = torch.load(os.path.join(model_path, 'best_ANN_model.pth'), map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        losses = AverageMeter()
        mae_errors = AverageMeter()
        
        test_targets = []
        test_preds = []
     

        with torch.no_grad():
            self.model.eval()
            for bn, (inputs, target) in enumerate(self.test_loader):

                input_var = inputs.to(self.device)
                
                target_var = target.unsqueeze(1).to(self.device)
                

                # compute output
                output = self.model(input_var)
        
                loss = self.criterion(output, target_var)

                mae_error = mae(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                
                test_pred = output.data.cpu()
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()


        with open(os.path.join(results_save_path, "test_results.csv"), 'w') as f:
            writer = csv.writer(f)
    
            for mofs, target, pred in zip(test_mofs, test_targets, test_preds):
                writer.writerow((mofs, target, pred))
        
        self.model.train()

        print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return test_preds, test_targets, losses.avg, mae_errors.avg
    
    def train_curve(self, write_dir):
        plt.rcParams['axes.linewidth'] = 1.5
        fig = plt.figure(figsize=[6.4, 6.4])
        ax = fig.gca()
        ax.plot(self.epoch_list, self.train_loss_list, color='b', linewidth=2, label='Train')
        ax.plot(self.epoch_list, self.valid_loss_list, color='r', linewidth=2, label='Validation')
        
        plt.xticks(fontsize=12, fontweight='bold', family='Helvetica')
        plt.yticks(fontsize=12, fontweight='bold', family='Helvetica')
        plt.tick_params(axis='both', direction='in', width=1, length=6)
        plt.legend(prop={'size': 12, 'weight': 'bold'})
        
        #plt.show()
        fig.savefig(f'{write_dir}train_curve.pdf', dpi=600, bbox_inches='tight')
    
def hyperoptoutput2param(best):   ## Function that obtains the best set of hyperparameters

    '''Change hyperopt output to dictionary with values '''

    for key in best.keys():
        if key in hyper_dict.keys():
            best[key] = hyper_dict[key][ best[key] ]

    return best


def model_eval(args, train_data):   ## Evaluates the model performance during hyperopt

    '''Take suggested arguments and perform model evaluation'''
    k_folds = 5

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=1)
    loss_cv = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_data)):
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(train_data, batch_size=args['batch_size'], sampler=train_sampler)
        val_loader = DataLoader(train_data, batch_size=args['batch_size'], sampler=val_sampler)

        ann_model = NeuralNetwork(args, train_loader, val_loader, val_loader, model_save=False, validation=True)
        loss_val = ann_model.train()
        loss_cv.append(loss_val)

    return sum(loss_cv)/len(loss_cv)

# Generate dataset
class XYDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)  # store X as a pytorch Tensor
        self.y = torch.Tensor(y)  # store y as a pytorch Tensor
        self.len=len(self.X)      # number of samples in the data

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
    

RandomState_dict = {'1inor_1edge': 42, '1inor_1org_1edge': 42,'2inor_1edge': 168, 'all': 336}    ## Specify the random state that was found to be the best 
data_splits = ['all']


for split in data_splits:
    r_state = RandomState_dict[split]
    save_dir = f'./ANN_models/{split}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    write_dir = save_dir

    column_to_drop = ['data_type', 'net', 'KVRH', 'D_func-I-0-all', 'D_func-I-1-all', 'D_func-I-2-all', 'D_func-I-3-all', 
                      'D_func-S-0-all', 'D_func-T-0-all', 'D_func-Z-0-all', 'D_func-alpha-0-all', 'D_func-chi-0-all',
                      'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all', 'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-T-0-all',
                      'D_lc-Z-0-all', 'D_lc-alpha-0-all', 'D_lc-chi-0-all', 'lc-I-0-all', 'D_mc-I-0-all', 
                      'D_mc-I-1-all', 'D_mc-I-2-all', 'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-T-0-all', 'D_mc-Z-0-all',
                      'D_mc-chi-0-all', 'mc-I-0-all']

   
    df_train = pd.read_csv(f'./train_feature_files/net_short_symb_combined_data_frame_{split}.csv')
    df_test = pd.read_csv(f'./test_feature_files/net_short_symb_combined_data_frame_{split}.csv')

    column_names = df_train.columns

    df_train.set_index('name', inplace=True)
    df_test.set_index('name', inplace=True)
   
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)
    
    train_mofs_final = list(df_train.index)
    test_mofs_final = list(df_test.index)
    
    X_train = df_train.drop(column_to_drop, axis=1).to_numpy()
    y_train = np.log(df_train['KVRH'].to_numpy())
    
    X_test = df_test.drop(column_to_drop, axis=1).to_numpy()
    y_test = np.log(df_test['KVRH'].to_numpy())
    

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(X_train.shape)
    print(X_test.shape)
    
    train_data = XYDataset(X_train, y_train)
    test_data = XYDataset(X_test, y_test)
    
    ## Hyperparameter optimization

        
    hyper_dict = {
        "batch_size": [8, 16, 32, 64],
        "depth": [2, 3, 4, 5],
        "width": [192, 256, 512, 1024],
        "activation": ['relu', 'tanh'],
        "alpha": [1e-5, 1e-4, 1e-3],
        "optimizer": ['adam', 'sgd'],
        "early_stopping": [10, 20, 40, 80],
        "dropout": [0, 0.2, 0.4],
        'weight_decay': [1e-7, 1e-6, 1e-5, 1e-4]
    }
     
    
    parameter_space = {
        "batch_size": hp.choice("batch_size", hyper_dict["batch_size"]),
        "depth": hp.choice("depth", hyper_dict['depth']),
        "width": hp.choice("width", hyper_dict['width']),
        "activation": hp.choice("activation", hyper_dict['activation']),
        "alpha": hp.choice("alpha", hyper_dict['alpha']),
        "optimizer": hp.choice("optimizer", hyper_dict["optimizer"]),
        "early_stopping": hp.choice("early_stopping", hyper_dict["early_stopping"]),
        "dropout": hp.choice("dropout", hyper_dict["dropout"]),
        "weight_decay": hp.choice("weight_decay", hyper_dict["weight_decay"])
    }
   
    print("Start trials")

    trials = Trials()
    best = fmin(fn=lambda params: model_eval(params, train_data), space=parameter_space, algo=tpe.suggest, max_evals=700, trials=trials)
    best = hyperoptoutput2param(best)
    print("Best parameter set: {}".format(best))
    with open(f'{write_dir}best_hyperparameters.txt', 'w') as write_file:
        write_file.write(str(best))

    print("Best loss from CV {:.2f}".format(-trials.best_trial['result']['loss']))
    #best = {'batch_size':64, 'lr':0.002, 'optimizer':'Adam', 'weight_decay':1e-06}
    ## Changes to be made
    test_loader = DataLoader(test_data, batch_size=best['batch_size'])

    X_train_only, X_val, y_train_only, y_val, index_train_only, index_val_only = train_test_split(X_train, y_train, range(len(train_mofs_final)), test_size=0.15, random_state=r_state, shuffle=True)
    train_only_data = XYDataset(X_train_only, y_train_only)
    val_data = XYDataset(X_val, y_val)

    train_loader = DataLoader(train_only_data, batch_size=best['batch_size'])
    val_loader = DataLoader(val_data, batch_size=best['batch_size'])

    ann_model = NeuralNetwork(best, train_loader, val_loader, test_loader, model_save=True, validation=True, model_save_path=write_dir)
    mae_val = ann_model.train()
    train_pred, train_target, loss_train, mae_train = ann_model.train_results(write_dir, write_dir, [train_mofs_final[index] for index in index_train_only])
    test_pred, test_target, loss_test, mae_test = ann_model.test(write_dir, write_dir, test_mofs_final)
    ann_model.train_curve(write_dir)

    train_test_plot(train_target, train_pred, test_target, test_pred, write_dir)
