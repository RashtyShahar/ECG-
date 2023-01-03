from torch.utils.data import Dataset
import pandas as pd
from torch.nn.functional import one_hot
import wfdb
import numpy as np
import ast
import torch

class ECGDataset(Dataset):

    def __init__(self, ECG_path, table_path, Transform,task):
        super().__init__()  # When using a subclass, remember to inherit its properties.
        # Define self.ECG_path, self.table (with pandas reading csv) and self.transform (create an object from the transform we implemented):
        self.ECG_path = ECG_path
        self.table = pd.read_csv(table_path)
        self.transform = Transform
        self.task=task

    def get_wfdb_path(self, index):
        # A method to get the correct path according to the data arrangement:
        folder_num = str(int(np.floor(index / 1000)))
        if len(folder_num) < 2:
            folder_num = '0' + folder_num
        wfdb_path = self.ECG_path + folder_num + '000' + '/' + f'{index:05d}' + '_lr'
        return wfdb_path

    def get_label(self, index,task):  #deleted here task arg
        # A method to decide the label:
        if self.task == 'classification':
            prob_dict = ast.literal_eval(self.table['scp_codes'][
                                             int(index - 1)])  # recording 0 is index 1, at its label is in location 0 in the table.
            keys = list(prob_dict.keys())
            # values = list(prob_dict.values())
            # best_key = keys[np.argmax(values)]
            labels = ['1AVB', 'CRBBB', 'CLBBB', 'SBRAD', 'AFIB', 'STACH']
            hot_vector= one_hot(torch.arange(0, 6) , num_classes=6)
            # hot_vector_NORM = torch.cat((hot_vector,torch.zeros(1,6)))
            ONEHOT_dict = dict(zip(labels,hot_vector))
            if type(sum(ONEHOT_dict[val] for val in keys if val in labels))==int:    # if all keys are not one of the labels
                return torch.squeeze(torch.zeros(1, 6))
            else:
                return torch.squeeze(sum(ONEHOT_dict[val] for val in keys if val in labels))

        elif self.task=='age estimation':
            age = self.table['age'][int(index - 1)]
            return age

    def __getitem__(self, index):
        # Read the record with wfdb (use get_wfdb_path) and transform its signal. Assign a label by using get_label.
        record = wfdb.rdrecord(self.get_wfdb_path(index))
        signal = self.transform(record.p_signal)  # get tensor with the right dimensions and type.
        label = self.get_label(index,self.task)
        return signal, label

    def __len__(self):
        return 21748  # database size after removing examples with missing 'age'
