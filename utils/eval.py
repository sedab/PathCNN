import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, cohen_kappa_score, jaccard_similarity_score, log_loss,recall_score, precision_score
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
import os


def get_tpf_fpr(predictions, labels, class_codes):
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
    else:
        fpr, tpr, _ = roc_curve(labels, predictions[:,1])
        auc_result = auc(fpr, tpr)

        for i in list(classes) + ['macro', 'micro']:
            roc_auc[i] = auc_result 

    return fpr, tpr, roc_auc





#confidence interval
def get_error(pred, true, classes=[0, 1, 2]):
    num_class=len(classes)

    n_bootstraps = 1000
    rng_seed0 = 42  # control reproducibility
    rng_seed1 = 100 # control reproducibility
    rng_seed2 = 250  # control reproducibility
    rng_seed3 = 400  # control reproducibility
    rng_seed4 = 650  # control reproducibility

    bootstrapped_scores0 = []
    bootstrapped_scores1 = []
    bootstrapped_scores2 = []
    bootstrapped_scores3 = []
    bootstrapped_scores4 = []
    
    rng0 = np.random.RandomState(rng_seed0)
    rng1 = np.random.RandomState(rng_seed1)
    rng2 = np.random.RandomState(rng_seed2)
    rng3 = np.random.RandomState(rng_seed3)
    rng4 = np.random.RandomState(rng_seed4)
    
    true_all=true.ravel()
    pred_all=pred.ravel()
   

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices0 = rng0.random_integers(0, len(pred[:,0]) - 1, len(pred[:,0]))
        #if len(np.unique(true[indices0,0])) < 2:
        #    continue
            
        indices1 = rng1.random_integers(0, len(pred[:,1]) - 1, len(pred[:,1]))
        if len(np.unique(true[indices1,1])) < 2:
             continue
                
      
        indices2 = rng2.random_integers(0, len(pred[:,2]) - 1, len(pred[:,2]))
        if len(np.unique(true[indices2,2])) < 2:
            continue
    
        indices4 = rng4.random_integers(0, len(pred_all) - 1, len(pred_all))
        if len(np.unique(true_all[indices4])) < 2:
            continue
       

        score0 = roc_auc_score(true[indices0,0], pred[indices0,0])
        score1 = roc_auc_score(true[indices1,1], pred[indices1,1])
        if num_class>2:
            score2 = roc_auc_score(true[indices2,2], pred[indices2,2])
            
        score4 = roc_auc_score(true_all[indices4], pred_all[indices4])
       
        bootstrapped_scores0.append(score0)
        bootstrapped_scores1.append(score1)
        if num_class>2:
            bootstrapped_scores2.append(score2)
       
        bootstrapped_scores4.append(score4)

    sorted_scores0 = np.array(bootstrapped_scores0)
    sorted_scores0.sort()
    confidence_lower0 = sorted_scores0[int(0.05 * len(sorted_scores0))]
    confidence_upper0 = sorted_scores0[int(0.95 * len(sorted_scores0))]

    sorted_scores1 = np.array(bootstrapped_scores1)
    sorted_scores1.sort()
    confidence_lower1 = sorted_scores1[int(0.05 * len(sorted_scores1))]
    confidence_upper1 = sorted_scores1[int(0.95 * len(sorted_scores1))]


    sorted_scores2 = np.array(bootstrapped_scores2)
    sorted_scores2.sort()
    confidence_lower2 = sorted_scores2[int(0.05 * len(sorted_scores2))]
    confidence_upper2 = sorted_scores2[int(0.95 * len(sorted_scores2))]

   
    #micro
    sorted_scores4=np.array(bootstrapped_scores4)                                      
    sorted_scores4.sort()
                                      
    confidence_lower_micro = sorted_scores4[int(0.05 * len(sorted_scores4))]
    confidence_upper_micro = sorted_scores4[int(0.95 * len(sorted_scores4))]                                    


    #macro
    
    sorted_scores5=np.array(np.concatenate((bootstrapped_scores1,bootstrapped_scores2), axis=0))

    
    sorted_scores5.sort()
                                      
    confidence_lower_macro = sorted_scores5[int(0.05 * len(sorted_scores5))]
    confidence_upper_macro = sorted_scores5[int(0.95 * len(sorted_scores5))]                                    

    if num_class==3:
        all= [confidence_lower_macro, confidence_upper_macro,confidence_lower_micro, confidence_upper_micro, confidence_lower0, confidence_upper0,confidence_lower1, confidence_upper1,confidence_lower2, confidence_upper2]
    return all









def get_auc(predictions, labels, class_names, classes=[0, 1, 2]):

    """
    Given predictions and labels, return the AUCs for all classes
    and micro, macro AUCs. Also saves a plot of the ROC curve to the
    path.
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
        for i in [0,1,2]:
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        errors=get_error(predictions,labels,classes)

        ### Micro AUC ###
        fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        roc_auc["micro_cl"]=errors[2]
        roc_auc["micro_cu"]=errors[3]


        ### Macro AUC ###
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(1,3)]))
        #print(all_fpr)
        mean_tpr = np.zeros_like(all_fpr)
        #SB
        for i in range(1,3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 2

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"]) 
        roc_auc["macro_cl"]=errors[0]      
        roc_auc["macro_cu"]=errors[1]
 
 
        cu.append(errors[5])
        cu.append(errors[7])
        cu.append(errors[9])
        cl.append(errors[4])
        cl.append(errors[6])
        cl.append(errors[8])

        
        #new metrics
        #precision = precision_score(label, np.argmax(predictions,axis=1),average='macro')
        #recall = recall_score(label, np.argmax(predictions,axis=1), average='macro')
        #cohenskappa = cohen_kappa_score(label, np.argmax(predictions,axis=1))
        #jaccard = jaccard_similarity_score(label, np.argmax(predictions,axis=1))
        #logloss = log_loss(label, predictions, labels=classes)
         
        #roc_auc['precision']=precision
        #roc_auc['recall']=recall
        #roc_auc['cohenskappa']=cohenskappa
        #roc_auc['jaccard']=jaccard
        #roc_auc['logloss']=logloss

        ### Make plot ###
        
        #plt.plot(fpr["micro"], tpr["micro"],
        #         label='micro-average: AUC = {0:0.3f}'
        #               ''.format(roc_auc["micro"]),
        #         color='deeppink', linestyle=':', linewidth=4)

        #plt.plot(fpr["macro"], tpr["macro"],
        #         label='macro-average: AUC = {0:0.3f}'
        #               ''.format(roc_auc["macro"]),
        #         color='navy', linestyle=':', linewidth=4)

        #colors = cycle(['aqua', 'darkorange', 'cornflowerblue','forestgreen','sienna','darkslateblue','limegreen','cadetblue','firebrick'])
        colors = cycle(['deeppink','navy','aqua'])
        for i, color in zip([0,1,2], colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='{0}: AUC = {1:0.2f} \n CI [{2:0.2f}, {3:0.2f}]'
                     ''.format(class_names[i], roc_auc[i], cl[i], cu[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=60)
    plt.ylabel('True Positive Rate', fontsize=60)
    plt.title('NYU ffpe Slides ROC Curve', fontsize=60)
    plt.legend(loc="lower right", fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.rcParams['axes.linewidth'] = 4
    ax.patch.set_edgecolor('black')  


    #print(class_encoding)
    print(roc_auc)
    return 


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