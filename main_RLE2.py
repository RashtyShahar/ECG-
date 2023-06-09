import torch
import torch.nn as nn
import os
import pandas as pd
import wfdb
from Transform import ECGTransform
from dateset import ECGDataset
from split import PtBXL_set_spit
from figures import figures_RLE ,learning_curve,figures,number_of_leads_plot,calc_score,calc_scores_cominations
from Training_RLE2 import forward_epoch_val,forward_epoch_test,forward_epoch_train,get_real_indexes
from EarlyStopping import EarlyStopping
from resnetModel2 import RLE
import numpy as np
from Training import forward_epoch
from model_Rebeiro import UResNet
import pickle

# task and device definition:
# task = 'classification' # can be 'classification' or 'age estimation'
task = 'age estimation'

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(1) if torch.cuda.is_available() else "cpu"

# path at remote server
DB_path = r"/home/rashty/data/"
frequency=500   # 500 Hz or 100 Hz
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
batch_size = 512
dl_train, dl_val, dl_test = PtBXL_set_spit(ecg_ds,table,batch_size, num_workers=0)


# defining the network (pipeline):
ecg_net = RLE([batch_size, transformed.shape[0], transformed.shape[1]],task)  #RLE model
torch.save(ecg_net.state_dict(), f'RLE_checkpoint_initialized.pt')
params_first=ecg_net.parameters()
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

learning_rate = 0.001
optimizer = torch.optim.Adam(params=ecg_net.parameters(), lr=learning_rate)
epochs = 13

'''
# binary classification task: Binary Cross Entropy --->  atrial fibrillation risk prediction
'''

train_loss_vec = []
val_loss_vec = []
lead_indexes_eliminate=[]
# each list belongs to 1 arrhythmia
if task=='classification':
    scores_for_plot = [[] for _ in range(7)]
else:
    scores_for_plot = []


# load variable saved from last run
if os.path.exists(r'scores_for_plot.pkl'):
    with open(r'scores_for_plot.pkl', 'rb') as file:
        data = pickle.load(file)
    lead_indexes_eliminate,scores_for_plot,_ = data

how_many_leads_to_reduce = 5
for i in range(how_many_leads_to_reduce):
    ecg_net.load_state_dict(torch.load('RLE_checkpoint_initialized.pt'))
    optimizer = torch.optim.Adam(params=ecg_net.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=13, verbose=True, learning_rate=learning_rate, learning_patience=3)
    # torch.save(ecg_net.state_dict(), f'RLE_checkpoint{i}.pt')
    for i_epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        print(f'Epoch: {i_epoch + 1}/{epochs}\n')

        train_loss, y_true_train, y_pred_train = forward_epoch_train(ecg_net, dl_train, loss_function, optimizer, train_loss,lead_to_eliminate=lead_indexes_eliminate,
                                                               to_train=True, desc='Train', device=device)
        train_loss = train_loss / len(dl_train)  # we want to get the mean over batches.
        train_loss_vec.append(train_loss)

        val_loss, y_true_val, y_pred_val = forward_epoch_val(ecg_net, dl_val, loss_function,optimizer,val_loss,lead_to_eliminate=lead_indexes_eliminate,
                                                            to_train=False, desc='Validation', device=device)

        val_loss = val_loss / len(dl_val)
        val_loss_vec.append(val_loss)

        early_stopping(val_loss_vec[i_epoch][-1], ecg_net,optimizer=optimizer)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    ecg_net.load_state_dict(torch.load('checkpoint.pt'))

    '''
    Val_los_vec is list with len=number of epochs ,
    each value is a tensor with 13 values as the number of combinations
    by stacking it and taking the first value we get the loss of the first combination at each epoch
    '''
    # val_loss_vec_stacked = torch.stack(val_loss_vec, dim=1)
    # #plot learning curve on training and validation set
    # learning_curve(train_loss_vec,val_loss_vec_stacked[0],task)

    '''
    metric is MAE for age estimation and average_precision_score for classification
    metric = list with average scores of the 6 classes for all combinations
    y_true_val.shape=(13,2186,6) for classification
    '''
    metric,class_scores = calc_scores_cominations(y_true_val,y_pred_val,task)

    ### to eliminate bassed on average score
    if task=='age estimation':
        #metric is MAE
        least_important = np.argmin(metric[:-1])
    else:
        #metric is AUPRC
        least_important = np.argmax(metric[:-1])

    ### try: eliminate based on 2nd class only
    # if task=='age estimation':
    #     #metric is MAE
    #     least_important = np.argmin(metric[:-1])
    # else:
    #     #metric is AUPRC
    #     least_important = np.argmax(class_scores[1])

    lead_indexes_eliminate.append(least_important)
    print(f'we can eliminate the {least_important}th lead')

    # evaluation - y_true and y_pred for test set:
    test_loss = 0
    __, y_true, y_pred = forward_epoch_test(ecg_net, dl_test, loss_function,optimizer,test_loss,lead_to_eliminate=lead_indexes_eliminate, desc='Test', device=device)
    '''
    mean_score is the mean of the 6 classes
    scores_list is a list with 6 values each one presents a class
    '''
    mean_score,scores_list = calc_score(y_true[:-1],y_pred[:-1],task)

    if task == 'classification':
        # scores_for_plot[-1].append(mean_score)
        for i in range(6):
            scores_for_plot[i].append(scores_list[i])
    else:
        scores_for_plot.append(mean_score)
    print(f'lead eliminated:{get_real_indexes(lead_indexes_eliminate)}')

    data = [lead_indexes_eliminate,scores_for_plot,get_real_indexes(lead_indexes_eliminate)]
    with open(r'scores_for_plot.pkl', 'wb') as file:
        pickle.dump(data, file)

if task == 'classification':
    for i,score in enumerate(scores_for_plot):
        number_of_leads_plot(len(lead_indexes_eliminate), score , task ,i=i)
else:
    number_of_leads_plot(len(lead_indexes_eliminate),scores_for_plot,task)
# print(f'lead eliminated:{get_real_indexes(lead_indexes_eliminate)}')



