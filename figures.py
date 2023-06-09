import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve , average_precision_score , r2_score
from sklearn.metrics import mean_absolute_error as mae
import os
import numpy as np

def figures(y_test, y_pred, task, to_display=True):
    #plot precision-recall curve
    y_test = y_test.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    plt.rcParams['font.size'] = 13 # changing the font size

    AUPRC = []
    if task == 'classification':
        # labels = ['AVB', 'CRBBB', 'CLBBB', 'SBRAD', 'AFIB', 'STACH', 'NORM']
        label = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("Precision-Recall Curves Test-set",fontsize=24)
        for i in range(y_pred.shape[1]):
            row = i // 3
            col = i % 3
            ax = plt.subplot2grid((2, 3), (row, col))
            # no skill:
            no_skill = len(y_test[:, i][y_test[:, i] == 1]) / len(y_test[:, i])

            # calculating precision and recall:
            precision, recall, thresholds = precision_recall_curve(y_test[:, i], y_pred[:, i])
            average_precision = average_precision_score(y_test[:, i], y_pred[:, i])
            AUPRC.append(average_precision)
            # create and plot precision recall curve:
            ax.plot(recall, precision, color= 'midnightblue',linewidth= 4, label=f'AUPRC={average_precision:.4f}')
            # plot no skill graph:
            ax.plot([0, 1], [no_skill, no_skill], linestyle='--',linewidth= 4, color = 'teal',label='No Skill')

            # add axis labels to plot
            ax.set_title(f'{label[i]}',fontsize=16)
            ax.legend()

        #display figure
        fig.supylabel('PPV',fontsize=14)
        fig.supxlabel('Se',fontsize=14)
        if to_display:
            plt.show()
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        # Save the plot to the 'graphs' folder
        plt.savefig('graphs/classification.png')

        # return sum(AUPRC)/len(AUPRC)

    if task == 'age estimation':
        # error calculation:
        R2 = (r2_score(y_test, y_pred)).round(2)
        MAE = (mae(y_test, y_pred)).round(2)
        err = y_test - y_pred
        std = np.round(np.std(err),decimals=2)
        # figures:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.2, color= 'teal')
        ax.plot(y_test, y_test, color= 'black', label='Estimated age = True age')
        a, b = np.polyfit(y_test, y_pred, 1)
        plt.plot(y_test, a * y_test + b,linewidth= 4,color= 'midnightblue',  label=f' $R^2$ = {R2}, MAE = {MAE:.2f} \u00b1 {std:.2f}')
        plt.legend(loc='upper left',fontsize=10)
        ax.set_xlabel('True age (years)')
        ax.set_ylabel('Estimated age (years)')
        if to_display:
            plt.show()
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        # Save the plot to the 'graphs' folder
        plt.savefig('graphs/age_estimation.png')
        # return MAE

def learning_curve(train_loss_vec,val_loss_vec,task):
    plt.figure()
    plt.plot(train_loss_vec, label='train')
    plt.plot(val_loss_vec, label='val')
    plt.title(f'{task}-Learning curve')
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    if task=='age estimation':
        plt.ylabel('L1 Loss')
        # Save the plot to the 'graphs' folder
        plt.savefig('graphs/learning_curve_Age_Estimation.png')
    else:
        plt.ylabel('BCE Loss')
        plt.savefig('graphs/learning_curve_classification.png')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()



#for lists outputs
def learning_curve_RLE_lst(train_loss_vec,val_loss_vec,task):
    plt.figure()
    i=1
    for train_loss,val_loss in zip(train_loss_vec,val_loss_vec):
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='val')
        plt.title(f'{task}-Learning curve,results without the {i}th lead')
        if task=='age estimation':
            plt.ylabel('L1 Loss')
        else:
            plt.ylabel('BCE Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
        i+=1

def figures_RLE(y_test_all, y_pred_all, task,to_display=False):
    j=0
    metric = []
    for y_test,y_pred in zip(y_test_all,y_pred_all):
        y_test = y_test.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        plt.rcParams['font.size'] = 13 # changing the font size

        if task == 'classification':

            # labels = ['AVB', 'CRBBB', 'CLBBB', 'SBRAD', 'AFIB', 'STACH', 'NORM']
            label = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

            fig = plt.figure(figsize=(18, 12))
            if j==0:
                fig.suptitle(f"Precision-Recall Curves,with 12 leads",fontsize=24)
            else:
                fig.suptitle(f"Precision-Recall Curves,validiation set results without the {j}th lead",fontsize=24)
            j+=1
            for i in range(y_pred.shape[1]):
                row = i // 3
                col = i % 3
                ax = plt.subplot2grid((2, 3), (row, col))
                # no skill:
                no_skill = len(y_test[:, i][y_test[:, i] == 1]) / len(y_test[:, i])

                # calculating precision and recall:
                ### y_test[:, i] is all examples of a certain disease
                precision, recall, thresholds = precision_recall_curve(y_test[:, i], y_pred[:, i])
                average_precision = average_precision_score(y_test[:, i], y_pred[:, i])
                # create and plot precision recall curve:
                ax.plot(recall, precision, color= 'midnightblue',linewidth= 4, label=f'AUPRC={average_precision:.4f}')
                # # plot no skill graph:
                ax.plot([0, 1], [no_skill, no_skill], linestyle='--',linewidth= 4, color = 'teal',label='No Skill')
                #
                # # add axis labels to plot
                ax.set_title(f'{label[i]}',fontsize=16)
                ax.legend()

            metric.append(average_precision)

            #display figure

            fig.supylabel('PPV',fontsize=14)
            fig.supxlabel('Se',fontsize=14)
            if to_display:
                plt.show()

            #add path to save figure
            # plt.savefig(r'')


        elif task == 'age estimation':
            # error calculation:
            R2 = (r2_score(y_test, y_pred)).round(2)
            MAE = (mae(y_test, y_pred)).round(5)
            metric.append(MAE)
            MAE = MAE.round(2)
            err = y_test - y_pred
            std = np.round(np.std(err),decimals=2)
            # figures:
            fig, ax = plt.subplots()
            fig.suptitle(f"Age Estimation,Results without the {j}nd lead",fontsize=24)
            j += 1
            ax.scatter(y_test, y_pred, alpha=0.2, color= 'teal')
            ax.plot(y_test, y_test, color= 'black', label='Estimated age = True age')
            a, b = np.polyfit(y_test, y_pred, 1)
            plt.plot(y_test, a * y_test + b,linewidth= 4,color= 'midnightblue',  label=f' $R^2$ = {R2}, MAE = {MAE:.2f} \u00b1 {std:.2f}')
            plt.legend(loc='upper left',fontsize=10)
            ax.set_xlabel('True age (years)')
            ax.set_ylabel('Estimated age (years)')
            if to_display:
                plt.show()

    return metric


def number_of_leads_plot(number_of_leads_eliminated,y,task,i=None):
    x = list(range(12,12-number_of_leads_eliminated,-1))
    x_fake = range(len(x))
    labels = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST','Mean Score for 6 classes']
    plt.plot(x_fake,y)
    plt.xticks(x_fake,x)
    if i==None:
        plt.title('RLE')
    else:
        plt.title(f'{labels[i]}')
    plt.xlabel('# of Leads')
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    if task=='classification':
        plt.ylabel('Score[AUPRC]')
        # Save the plot to the 'graphs' folder
        if i == None:
            plt.savefig('graphs/classification_RLE.png')
        else:
            plt.savefig(f'graphs/classification_RLE_{labels[i]}.png')
    elif task == 'age estimation':
        plt.ylabel('Score[MAE]')
        plt.savefig('graphs/Age_Estimation_RLE.png')

    plt.show()


def calc_score(y_test, y_pred, task):
    y_test = y_test.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    AUPRC = []
    if task == 'classification':
        for i in range(y_pred.shape[1]):
            average_precision = average_precision_score(y_test[:, i], y_pred[:, i])
            AUPRC.append(average_precision)
        ##Return the mean AUPRC of the 6 classes and a list of AUPRCs of each class
        return sum(AUPRC)/len(AUPRC) , AUPRC

    if task == 'age estimation':
        # error calculation:
        MAE = mae(y_test, y_pred)
        return MAE , MAE



def calc_scores_cominations(y_test_all, y_pred_all, task):
    '''
    :return: metric = each value is average scores of 6 classes for a certain combination
             class_scores = each value is a list of the all combination's scores of a certatin class
             for example first value will be scores of all combinations for the first class
             len(class_scores)=6 , len(class_scores[i])=number_of_combinations
    '''
    ### to be used for validation set where we get all combinations
    metric = []
    class_scores = [[] for _ in range(6)]
    for y_test,y_pred in zip(y_test_all,y_pred_all):
        '''
        y_test_all shape for validation is (13,2186,6)=(combinations,samples,classes)
        y_test shape is (2186,6)
        '''
        y_test = y_test.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        current_combination_score=[]
        if task == 'classification':
            for i in range(y_pred.shape[1]):
                '''
                y_test[:, i] is all samples of a single disease
                average_precision here is for a single combination and single disease
                '''
                average_precision = average_precision_score(y_test[:, i], y_pred[:, i])
                current_combination_score.append(average_precision)
                class_scores[i].append(average_precision)
            metric.append(np.mean(current_combination_score))

        elif task == 'age estimation':
            MAE = mae(y_test, y_pred)
            metric.append(MAE)
    if task == 'classification':
        return metric,class_scores
    else:
        return metric,metric

