import numpy as np
import torch

class EarlyStopping:
    """
    Lerning rate  will be is reduced by a factor of 10 whenever the validation loss
    does not present any improvement for learning_patience consecutive epochs.
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, verbose=False, delta=0.0001, path='checkpoint.pt', trace_func=print, learning_rate=0.001,learning_patience=7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            learning_rate(float):learning_rate
                            Default:0.001
            learning_patience(int): How long to wait after last time validation loss improved before changing learning rate
                            Default:7
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.learning_rate = learning_rate
        self.reduce_learning_rate = False
        self.no_improvement_count = 0
        self.learning_patience = learning_patience #should be smaller than patience
        self.counter_of_decreasment=0

    def __call__(self, val_loss, model,optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.no_improvement_count +=1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.trace_func(f'Reduce learning rate counter: {self.no_improvement_count} out of {self.learning_patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            if self.no_improvement_count >= self.learning_patience:
                    self.reduce_learning_rate = True
                    self.no_improvement_count = 0
                    self.counter_of_decreasment+=1
                    if self.counter_of_decreasment>2: # we only allow 2 reduction at the lr
                        self.trace_func(f'leaning rate was reduced by a factor of 100')
                        self.early_stop = True
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.learning_rate / 10
                    self.learning_rate = self.learning_rate / 10
                    self.trace_func(f'learning rate reduced to {self.learning_rate}')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
