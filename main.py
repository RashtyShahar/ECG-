import torch
import torch.nn as nn
from pathlib import Path
import os
import pandas as pd
import wfdb
from Transform import ECGTransform
from dateset import ECGDataset
from split import PtBXL_set_spit
from model1 import ecgNet
from figures import figures , learning_curve
from Training import forward_epoch
from model_Rebeiro import EcgModel
from EarlyStopping import EarlyStopping



# task and device definition:
task = 'classification' # can be 'classification' or 'age estimation'
# task = 'age estimation'

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(2) if torch.cuda.is_available() else "cpu"

#this is the path on my computer

# data path and directory:
# DB_path = r"C:\Users\rasht\Documents\Final Project\T2\data_t2"
# ECG_path = DB_path + r'\records100\\'
# table_path = DB_path + '\ptbxl_database.csv'
# table = pd.read_csv(table_path)
# list_of_folders = os.listdir(ECG_path)
# list_of_files = os.listdir(ECG_path + list_of_folders[0])

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



# connecting our dataset with the dataloader:
ecg_ds = ECGDataset(ECG_path, table_path, ECGTransform(), task=task,frequency=frequency)
'''
batch_to_show = 10 
ecg_dl = DataLoader(ecg_ds, batch_size=batch_size, shuffle=True, num_workers=0)
'''



# splitting the data to train, validate and test sets:
'''
#batch_size_split = 512
#dl_train, dl_test = train_test_split(ecg_ds, ratio=0.97, batch_size=batch_size_split)
#missing the validation test set, need to use code from paper
#sanity checks!
print(len(dl_train))
print(len(dl_val))
print(len(dl_test))
'''
batch_size = 256
dl_train, dl_val, dl_test = PtBXL_set_spit(ecg_ds,table,batch_size, num_workers=0)



# defining the network (pipeline):
# ecg_net = ecgNet([batch_size, transformed.shape[0], transformed.shape[1]],task)  # basic model
ecg_net = EcgModel([batch_size, transformed.shape[0], transformed.shape[1]],task)  #Rebeiro's model

'''
ecg_net = ecgNet([4, 12, 1000],task)
batch_size = 4
'''

# loss function, learning rate, optimization and epochs definitions:

# labels = ['1AVB', 'CRBBB', 'CLBBB', 'SBRAD', 'AFIB', 'STACH']
total_examples=21748
pos_weight=torch.tensor([(total_examples-787)/787,(total_examples-538)/538,(total_examples-526)/526,(total_examples-637)/637,(total_examples-1493)/1493,(total_examples-825)/825])
pos_weight = pos_weight.to(device)
# loss_function = nn.L1Loss() if task == 'age estimation' else nn.BCELoss()
classification_loss=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_function = nn.L1Loss() if task == 'age estimation' else classification_loss

learning_rate = 0.0001
optimizer = torch.optim.Adam(params=ecg_net.parameters(), lr=learning_rate)
epochs = 80
'''
# binary classification task: Binary Cross Entropy --->  atrial fibrillation risk prediction
'''

# training the model:
early_stopping = EarlyStopping(patience=13, verbose=True,learning_rate=learning_rate,learning_patience=10)
train_loss_vec = []
val_loss_vec = []

for i_epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    print(f'Epoch: {i_epoch + 1}/{epochs}\n')

    train_loss, y_true_train, y_pred_train = forward_epoch(ecg_net, dl_train, loss_function, optimizer, train_loss,
                                                           to_train=True, desc='Train', device=device)

    val_loss, y_true_val, y_pred_val = forward_epoch(ecg_net, dl_val, loss_function, optimizer, val_loss,
                                                        to_train=False, desc='Validation', device=device)

    # Metrics:
    train_loss = train_loss / len(dl_train)  # we want to get the mean over batches.
    train_loss_vec.append(train_loss)
    val_loss = val_loss / len(dl_val)
    val_loss_vec.append(val_loss)

    early_stopping(val_loss_vec[-1], ecg_net,optimizer=optimizer)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    # load the last checkpoint with the best model
ecg_net.load_state_dict(torch.load('checkpoint.pt'))

#plot learning curve on training and validation set
learning_curve(train_loss_vec,val_loss_vec,task)

# evaluation - y_true and y_pred:
test_loss = 0
__, y_true, y_pred = forward_epoch(ecg_net, dl_test, loss_function, optimizer, test_loss,to_train=False, desc='Test', device=device)


figures(y_true,y_pred,task)