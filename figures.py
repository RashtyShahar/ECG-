import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve , average_precision_score , r2_score
from sklearn.metrics import mean_absolute_error as mae


def figures(y_test, y_pred, task):
    y_test = y_test.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    plt.rcParams['font.size'] = 14 # changing the font size

    if task == 'classification':
        # labels = ['AVB', 'CRBBB', 'CLBBB', 'SBRAD', 'AFIB', 'STACH', 'NORM']
        label = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("Precision-Recall Curves",fontsize=20)
        for i in range(y_pred.shape[1]):
            row = i // 3
            col = i % 3
            ax = plt.subplot2grid((2, 3), (row, col))
            # no skill:
            no_skill = len(y_test[:, i][y_test[:, i] == 1]) / len(y_test[:, i])

            # calculating precision and recall:
            precision, recall, thresholds = precision_recall_curve(y_test[:, i], y_pred[:, i])
            average_precision = average_precision_score(y_test[:, i], y_pred[:, i])
            # create and plot precision recall curve:
            ax.plot(recall, precision, color= 'midnightblue',linewidth= 4, label=f'AUPRC={average_precision:.4f}')
            # plot no skill graph:
            ax.plot([0, 1], [no_skill, no_skill], linestyle='--',linewidth= 4, color = 'teal',label='No Skill')

            # add axis labels to plot
            ax.set_title(f'{label[i]}',fontsize=16)
            ax.legend()

        #display figure
        fig.supylabel('PPV')
        fig.supxlabel('Se')
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
        ax.scatter(y_test, y_pred, alpha=0.2, color= 'teal')
        ax.plot(y_test, y_test, color= 'black', label='y prediction = y true')
        a, b = np.polyfit(y_test, y_pred, 1)
        plt.plot(y_test, a * y_test + b,linewidth= 4,color= 'midnightblue',  label=f' $R^2$ = {R2}, MAE = {MAE:.2f} \u00b1 {std:.2f}')
        plt.legend(loc='upper left',fontsize=10)
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
