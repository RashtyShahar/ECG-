from torch.utils.data import  DataLoader, SubsetRandomSampler
import torch

def PtBXL_set_spit(dataset,table, batch_size, num_workers):
    '''
    Split the data to val,train,test sets based on strat-fold from ptbxl_database.csv
    :param dataset:
    :param table:
    :param batch_size:
    :param num_workers:
    :return:
    '''
    test_fold = 10
    val_fold = 9
    test=table[table['strat_fold']==test_fold]['ecg_id']
    validation=table[table['strat_fold']==val_fold]['ecg_id']
    train=table[(table['strat_fold']!=val_fold) & (table['strat_fold']!=test_fold)]['ecg_id']
    dl_test = DataLoader(dataset, batch_size=batch_size,sampler=test , num_workers=num_workers)
    dl_val = DataLoader(dataset, batch_size=batch_size, sampler=validation, num_workers=num_workers)
    dl_train = DataLoader(dataset, batch_size=batch_size, sampler=train,num_workers=num_workers)
    return dl_train, dl_val, dl_test
