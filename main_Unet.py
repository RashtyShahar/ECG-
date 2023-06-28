import torch
import torch.nn as nn
import os
import pandas as pd
import wfdb
from Transform import ECGTransform
from dateset import ECGDataset
from split import PtBXL_set_spit
from figures import figures_RLE ,learning_curve,figures,number_of_leads_plot,calc_score,calc_scores_cominations
from Training_RLE2 import get_real_indexes,forward_epoch_train_UNet
from EarlyStopping import EarlyStopping
import numpy as np
from UResNet import UResNet
import pickle

# lead_indexes_eliminate=[10,9,11,8,7,6,5,4,2,3]
lead_indexes_eliminate=[10,9,11,8,7,6,5,4,2]

task = 'classification' # can be 'classification' or 'age estimation'
# task = 'age estimation'

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(1) if torch.cuda.is_available() else "cpu"

# path at remote server
DB_path = r"/home/rashty/data/"
frequency=100   # 500 Hz or 100 Hz
ECG_path = DB_path + f'records{frequency}/'
table_path = DB_path + 'ptbxl_database.csv'
table_org = pd.read_csv(table_path)
table_org['age']=pd.to_numeric(table_org['age'], errors='coerce')
table=table_org[table_org['age'].notna()]
list_of_folders = os.listdir(ECG_path)
list_of_files = os.listdir(ECG_path + list_of_folders[0])

'''
sanity check:
# print(f'List of folders:\n{list_of_folders}')
# print(f'\nList of files:\n{list_of_files[:10]}')
'''

# extracting and transforming the ECG leads:
record = wfdb.rdrecord(ECG_path + list_of_folders[0] + '/' + list_of_files[0][:-4])
signal = record.p_signal
test_transform = ECGTransform()
transformed = test_transform(signal)


ecg_ds = ECGDataset(ECG_path, table_path , ECGTransform(), task=task,frequency=frequency)
batch_size = 16
dl_train, dl_val, dl_test = PtBXL_set_spit(ecg_ds,table,batch_size, num_workers=0)


# dummy_input = torch.zeros([1,12-len(lead_indexes_eliminate),5000])
# dummy_model = UResNet(([1, 12-len(lead_indexes_eliminate), 5000]))
# output = dummy_model(dummy_input)
# print('input.shape:',dummy_input.shape)
# print('output.shape:',output.shape)

Unet = UResNet([batch_size, 12-len(lead_indexes_eliminate), transformed.shape[1]])
# print(Unet)
learning_rate = 0.001
optimizer = torch.optim.Adam(params=Unet.parameters(), lr=learning_rate)
epochs = 30
loss_function = nn.L1Loss()

early_stopping = EarlyStopping(patience=13, verbose=True,learning_rate=learning_rate,learning_patience=10,path='checkpoint_Unet.pt')
train_loss_vec = []
val_loss_vec = []

for i_epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    print(f'Epoch: {i_epoch + 1}/{epochs}\n')

    train_loss, y_true_train, y_pred_train = forward_epoch_train_UNet(Unet, dl_train, loss_function, optimizer, train_loss,lead_to_eliminate=lead_indexes_eliminate,
                                                           to_train=True, desc='Train', device=device)

    val_loss, y_true_val, y_pred_val = forward_epoch_train_UNet(Unet, dl_val, loss_function, optimizer, val_loss,lead_to_eliminate=lead_indexes_eliminate,
                                                        to_train=False, desc='Validation', device=device)

    # Metrics:
    train_loss = train_loss / len(dl_train)  # we want to get the mean over batches.
    train_loss_vec.append(train_loss)
    val_loss = val_loss / len(dl_val)
    val_loss_vec.append(val_loss)

    early_stopping(val_loss_vec[-1],Unet,optimizer=optimizer)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# load the last checkpoint with the best model
Unet.load_state_dict(torch.load('checkpoint_Unet.pt'))
learning_curve(train_loss_vec,val_loss_vec,task)

test_loss = 0
__, X_true, X_pred = forward_epoch_train_UNet(Unet, dl_test, loss_function, optimizer, test_loss,lead_to_eliminate=lead_indexes_eliminate,
                                                        to_train=False, desc='Test', device=device)

# data = [X_true.cpu().detach().numpy(),X_pred.cpu().detach().numpy()]
# data = [X_true, X_pred]
with open(f'Unet_Restored_from_{12-len(lead_indexes_eliminate)}_leads.pkl', 'wb') as file:
    pickle.dump(X_pred, file)