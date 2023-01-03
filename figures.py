import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae


def figures(y_test, y_pred, task):
    y_test = y_test.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    if task == 'classification':
        # labels = ['AVB', 'CRBBB', 'CLBBB', 'SBRAD', 'AFIB', 'STACH', 'NORM']
        label = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

        for i in range(y_pred.shape[1]):
            plt.figure()
            # no skill:
            no_skill = len(y_test[:, i][y_test[:, i] == 1]) / len(y_test[:, i])

            # calculating precision and recall:
            precision, recall, thresholds = precision_recall_curve(y_test[:, i], y_pred[:, i])

            # create and plot precision recall curve:
            plt.plot(recall, precision, label='precision recal curve')
            # plot no skill graph:
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

            # add axis labels to plot
            plt.title(f'{label[i]} Precision-Recall Curve')
            plt.ylabel('PPV')
            plt.xlabel('Se')
            plt.legend()

        # display plot
            plt.show()
            #add path to save figure
            # plt.savefig(r'')

    if task == 'age estimation':
        # error calculation:
        R2 = (r2_score(y_test, y_pred)).round(2)
        MAE = (mae(y_test, y_pred)).round(2)
        err = y_test - y_pred
        std = np.round(np.std(err),decimals=2)
        # figures:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot(y_test, y_test)
        a, b = np.polyfit(y_test, y_pred, 1)
        plt.plot(y_test, a * y_test + b, label=f' $R^2$ = {R2}, MAE = {MAE:.2f} + {std:.2f}')
        plt.legend(loc='upper left')
        ax.set_xlabel('True age [years]')
        ax.set_ylabel('estimated age [years]')
        plt.show()

def learning_curve(train_loss_vec,val_loss_vec,task):
    plt.figure()
    plt.plot(train_loss_vec, label='train')
    plt.plot(val_loss_vec, label='val')
    plt.title(f'{task}-Learning curve')
    if task=='age estimation':
        plt.ylabel('L1 Loss')
    else:
        plt.ylabel('BCE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()