import torch
import torch.nn as nn
import os
import pandas as pd
import wfdb
from Transform import ECGTransform
from dateset import ECGDataset
from split import PtBXL_set_spit
from figures import figures , learning_curve
from Training import forward_epoch
from model_Rebeiro import EcgModel
from EarlyStopping import EarlyStopping
import pickle
from Training_RLE2 import forward_epoch_with_elimination, forward_epoch_with_pretask
from Training import forward_epoch


# task and device definition:
task = 'classification' # can be 'classification' or 'age estimation'
# task = 'age estimation'



# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(0) if torch.cuda.is_available() else "cpu"

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
ecg_net = EcgModel([batch_size, transformed.shape[0], transformed.shape[1]],task)  #Rebeiro's model



# labels = ['1AVB', 'CRBBB', 'CLBBB', 'SBRAD', 'AFIB', 'STACH']
total_examples=21748
pos_weight=torch.tensor([(total_examples-787)/787,(total_examples-538)/538,(total_examples-526)/526,(total_examples-637)/637,(total_examples-1493)/1493,(total_examples-825)/825])
pos_weight = pos_weight.to(device)
# loss_function = nn.L1Loss() if task == 'age estimation' else nn.BCELoss()
classification_loss=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_function = nn.L1Loss() if task == 'age estimation' else classification_loss

learning_rate = 0.001
optimizer = torch.optim.Adam(params=ecg_net.parameters(), lr=learning_rate)
# epochs = 70

lead_indexes_eliminate=[10,9,11,8,7,6,5,4,2,3]
# lead_indexes_eliminate=[10,9,11,8,7]

#לשנות את השורה הזאת לצקפוינט הנכון אם מחליף מודל####
ecg_net.load_state_dict(torch.load('checkpoint.pt'))

with open(f'Unet_Restored_from_{12-len(lead_indexes_eliminate)}_leads.pkl', 'rb') as file:
    X_hat = pickle.load(file)
print("X_hat shape=",X_hat.shape)

test_loss = 0
print(X_hat)

__, y_true, y_pred = forward_epoch_with_pretask(ecg_net, dl_test,X_hat,batch_size ,loss_function, optimizer, test_loss,to_train=False, desc='Test', device=device)

data = [y_true.cpu().detach().numpy(),y_pred.cpu().detach().numpy()]
with open(f'model_Rebiro_with_PreTask_{task}_{12-len(lead_indexes_eliminate)}.pkl', 'wb') as file:
    pickle.dump(data, file)
figures(y_true,y_pred,task)






