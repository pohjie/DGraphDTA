import sys, os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from gnn import GNNNet
from utils import *
from emetrics import *
from data_process import create_dataset_for_5folds

import time
import pdb

datasets = [['davis', 'kiba'][int(sys.argv[1])]]

cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][int(sys.argv[2])]
print('cuda_name:', cuda_name)
fold = [0, 1, 2, 3, 4][int(sys.argv[3])]
cross_validation_flag = True
# print(int(sys.argv[3]))

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
NUM_EPOCHS = 2000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Main program: iterate over different datasets
result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = GNNNet()
model.to(device)
model_st = GNNNet.__name__
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
for dataset in datasets:
    train_data, valid_data = create_dataset_for_5folds(dataset, fold)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                               collate_fn=collate)

    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    best_ci = -1
    model_file_name = 'models/model_' + model_st + '_' + dataset + '_' + str(fold) + '.model'

    start_time = time.time()
    now_time = time.time()
    for epoch in range(NUM_EPOCHS):
        if epoch % 10 == 0:
            print('time for 10 epochs is: ', (time.time()-now_time)/60, ' minutes.')
            now_time = time.time()
        train(model, device, train_loader, optimizer, epoch + 1)
        # print('training the model takes: ', time.time()-now_time)

        print('predicting for valid data')
        G, P = predicting(model, device, valid_loader)
        # print('predicting takes: ', time.time()-now_time)

        val = get_mse(G, P)
        # print('retrieving val takes: ', time.time()-now_time)
        print('valid result:', val, best_mse)

        ci = get_ci(G, P)
        # print('calculating ci takes: ', time.time()-now_time)

        best_ci = max(best_ci, ci)
        print('ci here is:', ci, '; best ci is:', best_ci)
        if val < best_mse:
            best_mse = val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            print('rmse improved at epoch ', best_epoch, '; best_test_mse', best_mse, model_st, dataset, fold)
        else:
            print('No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, model_st, dataset, fold)
    print('time taken is: ', (time.time()-start_time)/3600)
