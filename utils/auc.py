import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from PIL import Image

def get_auc(path, predictions, labels, classes=[0, 1, 2]):
    """
    Given predictions and labels, return the AUCs for all classes
    and micro, macro AUCs. Also saves a plot of the ROC curve to the
    path.

    """

    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    if len(classes) > 2:
        # Convert labels to one-hot-encoding
        labels = label_binarize(labels, classes = classes)

        ### Individual class AUC ###
        for i in classes:
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        ### Micro AUC ###
        fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        ### Macro AUC ###
        all_fpr = np.unique(np.concatenate([fpr[i] for i in classes]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in classes:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(classes)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        ### Make plot ###

        plt.figure(figsize=(12, 12))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(classes, colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
    else:
        fpr, tpr, _ = roc_curve(labels, predictions[:,1])
        auc_result = auc(fpr, tpr)

        for i in list(classes) + ['macro', 'micro']:
            roc_auc[i] = auc_result

        plt.figure(figsize=(12, 12))
        plt.plot(fpr, tpr, lw=2,
                 label='ROC curve (area = {0:0.2f})'
                 ''.format(auc_result))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(path)

    return roc_auc#, fpr, tpr
