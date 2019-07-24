import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, cohen_kappa_score, jaccard_similarity_score, log_loss,recall_score, precision_score
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
import os
from itertools import chain


#confidence interval
def get_error(pred, true, classes=[0, 1, 2]):
    """
    Given predictions and labels, return the confidence intervals for all classes
    and micro, macro AUCs. 
    
    """
    num_class=len(classes)

    n_bootstraps = 1000
    rng_seed = {}
    bootstrapped_scores = {}
    sorted_scores = {}
    confidence_lower = {}
    confidence_upper = {}
    rng = {}
    indices = {}
    score = {}
    
    #results will be saved at
    all_cl = {}
    all_cu = {}
    
    if num_class>2:
        true = label_binarize(true, classes = classes)
    else:
        true = label_binarize(true, classes = [0,1,0])
        true = np.hstack((true, 1 - true))
    
    # control reproducibility
    seed=199
    rng_seed[0]= seed
    for c in range(0,num_class+1):
        rng[c]=np.random.RandomState(rng_seed[c])
        seed=seed+1000
        rng_seed[c+1]= seed

    
    true_all=true.ravel()
    pred_all=pred.ravel()
   
    #initilize the bootsrapped scores
    for c in range(0,num_class):
        bootstrapped_scores[c]=[]
    bootstrapped_scores['micro']=[]
        
    for i in range(n_bootstraps):
        
        
        # bootstrap by sampling with replacement on the prediction indices
        indices[0] = rng[c].random_integers(0, len(pred[:,0]) - 1, len(pred[:,0]))
        
        if num_class>2:
            for c in range(1,num_class):   
                indices[c] = rng[c].random_integers(0, len(pred[:,c]) - 1, len(pred[:,c]))
                if len(np.unique(true[indices[c],c])) < 2:
                    continue

            indices['micro'] = rng[num_class].random_integers(0, len(pred_all) - 1, len(pred_all))
                
        score[0]= roc_auc_score(true[indices[0],0], pred[indices[0],0])  
        bootstrapped_scores[0].append(score[0])
        
        if num_class>2:
            for c in range(1,num_class):
                score[c]= roc_auc_score(true[indices[c],c], pred[indices[c],c])
                bootstrapped_scores[c].append(score[c])

            score_micro = roc_auc_score(true_all[indices['micro']], pred_all[indices['micro']])
            bootstrapped_scores['micro'].append(score_micro)
        
    if num_class>2:
        for c in range(0,num_class):
            sorted_scores[c] = np.array(bootstrapped_scores[c])
            sorted_scores[c].sort()
            confidence_lower[c] = sorted_scores[c][int(0.05 * len(sorted_scores[c]))]
            confidence_upper[c] = sorted_scores[c][int(0.95 * len(sorted_scores[c]))]

        #micro
        sorted_scores['micro']=np.array(bootstrapped_scores['micro']) #!!!!!!!!!!!! 
        sorted_scores['micro'].sort()# does sort work like this?                                   
        confidence_lower['micro'] = sorted_scores['micro'][int(0.05 * len(sorted_scores['micro']))]
        confidence_upper['micro'] = sorted_scores['micro'][int(0.95 * len(sorted_scores['micro']))]                                    

        #macro
        all_bs = []
        for c in range(0,num_class):
            all_bs.append(bootstrapped_scores[c])

        #print((np.concatenate((bootstrapped_scores[0],bootstrapped_scores[1]), axis=0)).shape)
        sorted_scores['macro']=np.array(list(chain.from_iterable(all_bs)))
        sorted_scores['macro'].sort()

        confidence_lower['macro'] = sorted_scores['macro'][int(0.05 * len(sorted_scores['macro']))]
        confidence_upper['macro'] = sorted_scores['macro'][int(0.95 * len(sorted_scores['macro']))]                                    


        for c in range(0,num_class):
            all_cl[c] = confidence_lower[c]
            all_cu[c] = confidence_upper[c]
        all_cl['micro'] = confidence_lower['micro']
        all_cu['micro'] = confidence_upper['micro']
        all_cl['macro'] = confidence_lower['macro']
        all_cu['macro'] = confidence_upper['macro']
    
    else:
        sorted_scores[0] = np.array(bootstrapped_scores[0])
        sorted_scores[0].sort()
        confidence_lower[0] = sorted_scores[0][int(0.05 * len(sorted_scores[0]))]
        confidence_upper[0] = sorted_scores[0][int(0.95 * len(sorted_scores[0]))]
        all_cl[0] = confidence_lower[0]
        all_cu[0] = confidence_upper[0]

    return all_cl, all_cu




def get_auc(predictions, labels, class_names, classes=[0, 1, 2]):
    """
    Given predictions and labels, return the AUCs for all classes
    and micro, macro AUCs. 
    
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    label=labels 
    cu =[]
    cl=[]
    
    ax = plt.figure(figsize=(12, 12))

    if len(classes) > 2:
        # Convert labels to one-hot-encoding
        labels = label_binarize(labels, classes = classes)

        ### Individual class AUC ###
        for i in classes:
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        cl , cu =get_error(predictions,label,classes)

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
        
        #other metrics
        precision = precision_score(label, np.argmax(predictions,axis=1),average='macro')
        recall = recall_score(label, np.argmax(predictions,axis=1), average='macro')
        cohenskappa = cohen_kappa_score(label, np.argmax(predictions,axis=1))
        jaccard = jaccard_similarity_score(label, np.argmax(predictions,axis=1))
        logloss = log_loss(label, predictions, labels=classes)
         
        roc_auc['precision']=precision
        roc_auc['recall']=recall
        roc_auc['cohenskappa']=cohenskappa
        roc_auc['jaccard']=jaccard
        roc_auc['logloss']=logloss
        
        print('AUC:')
        print(roc_auc)
        print('CU:')
        print(cu)
        print('CL:')
        print(cl)
        
        ### Make plot ###

        plt.figure(figsize=(12, 12))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average: AUC = {0:0.2f} \n CI [{1:0.2f}, {2:0.2f}]'
                       ''.format(roc_auc['micro'], cl['micro'], cu['micro']),
                 color='darkorange', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average: AUC = {0:0.2f} \n CI [{1:0.2f}, {2:0.2f}]'
                       ''.format(roc_auc['macro'], cl['macro'], cu['macro']),
                 color='forestgreen', linestyle=':', linewidth=4)

        colors = cycle(['deeppink','navy','aqua','darkorange', 'cornflowerblue'])
        for i, color in zip(classes, colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='{0}: AUC = {1:0.2f} \n CI [{2:0.2f}, {3:0.2f}]'
                     ''.format(class_names[i], roc_auc[i], cl[i], cu[i]))

    else:
        fpr, tpr, _ = roc_curve(labels, predictions[:,1])
        auc_result = auc(fpr, tpr)

        for i in list(classes) + ['macro', 'micro']:
            roc_auc[i] = auc_result
            
        cl , cu =get_error(predictions,label,classes)
        
        #other metrics
        precision = precision_score(label, np.argmax(predictions,axis=1),average='macro')
        recall = recall_score(label, np.argmax(predictions,axis=1), average='macro')
        cohenskappa = cohen_kappa_score(label, np.argmax(predictions,axis=1))
        jaccard = jaccard_similarity_score(label, np.argmax(predictions,axis=1))
        logloss = log_loss(label, predictions, labels=classes)
         
        roc_auc['precision']=precision
        roc_auc['recall']=recall
        roc_auc['cohenskappa']=cohenskappa
        roc_auc['jaccard']=jaccard
        roc_auc['logloss']=logloss
        
        print(roc_auc)
        print('CU:')
        print(cu)
        print('CL')
        print(cl)
        
        color='navy'
        plt.figure(figsize=(12, 12))
        plt.plot(fpr, tpr, color=color, lw=2,
                     label='AUC = {0:0.2f} \n CI [{1:0.2f}, {2:0.2f}]'
                     ''.format(roc_auc[0], cl[0], cu[0]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=60)
    plt.ylabel('True Positive Rate', fontsize=60)
    plt.title('ROC Curve', fontsize=60)
    plt.legend(loc="lower right", fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.rcParams['axes.linewidth'] = 4
    ax.patch.set_edgecolor('black') 

    return fpr, tpr, roc_auc, cu, cl


#get the train log file
def get_class_coding(lf):
    auc_new = []
    phrase = "Class encoding:"

    with open(lf, 'r+') as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            #print(line)
            if phrase in line:
                class_encoding = lines[i + 1] # you may want to check that i < len(lines)
                break
                
    class_encoding = class_encoding.strip('\n').strip('{').strip('}')
    #print(class_encoding)
            
    class_names = []
    class_codes = []

    for c in class_encoding.split(','):
        #print(c)
        class_names.append(c.split(':')[0].replace("'", "").replace(" ", "").split('-')[-1])
        class_codes.append(int(c.split(':')[1]))


    class_coding = {}
    for i in range(len(class_names)):
        class_coding[class_codes[i]] = class_names[i]

    return class_names, class_codes, class_coding