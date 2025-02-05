#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
import warnings as wr
wr.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

import xgboost as xgb
from xgboost import XGBClassifier


# # Model evaluation metrics: calculating recall, fpr, roc-auc, confusion matrix

# In[2]:


from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_performance3(model, X_test, y_test, threshold=0.5, plot_roc_curve=False, plot_confusion_matrix=False):
    dtest = xgb.DMatrix(X_test)
    preds = model.predict(dtest)
    auc_score = roc_auc_score(y_test, preds)
    # Classify based on the threshold
    pred_bool = (preds >= threshold).astype(int)
    
    # Compute confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_test, pred_bool).ravel()
    cm = confusion_matrix(y_test, pred_bool)
    tpr = tp / (tp + fn)  # True Positive Rate
    fpr = fp / (fp + tn)  # False Positive Rate
    
    # Compute ROC curve
    fpr_curve, tpr_curve, _ = roc_curve(y_test, preds)
    
    # Print the results
    print(f"Recall (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f'ROC AUC: {auc_score:.5f}')
    
    # Plot ROC curve if enabled
    if plot_roc_curve:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_curve, tpr_curve, color='blue', label=f'AUC = {auc_score:.4f}')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Guess')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
    
    # Plot confusion matrix if enabled
    if plot_confusion_matrix:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    return {
        "PREDS": preds, 
        "PREDS_BOOL": pred_bool, 
        "TN": tn, 
        "FP": fp, 
        "FN": fn, 
        "TP": tp, 
        "TPR": tpr,
        "FPR": fpr,
        "ROC AUC": auc_score,
        "Confusion Matrix": cm,
        "ROC Curve": (fpr_curve, tpr_curve),
    }
# # Objective function 

# In[3]:


def objective(hyperparameters, dtrain, iteration):
    """Objective function for grid and random search. Returns
       the cross validation score from a set of hyperparameters."""
    
    # Perform n_folds cross validation
    cv_results = xgb.cv(params = hyperparameters, 
                        dtrain = dtrain, num_boost_round = 10000, nfold = 10, 
                        early_stopping_rounds = 5, metrics = 'auc', seed = 42,
                        verbose_eval=2, maximize=True)
    
    # results to retun
    score = cv_results['test-auc-mean'].max()
    # estimators = len(cv_results['test-auc-mean'])
    # hyperparameters['n_estimators'] = estimators 
    
    return [cv_results, score, hyperparameters, iteration]


# # Random Search

# In[6]:


import csv
def write_csv_test(outfile):
    results = pd.DataFrame(columns = ['score', 'params', 'iterations'])
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([1,2,3])
    of_connection.close()
    
def random_search(dtrain, param_grid, out_file, iteration=1):
    best_auc = 0
    best_params = {}
    results = pd.DataFrame(columns = ['score', 'params', 'iterations'], 
                          index = list(range(iteration)))
    for i in range(iteration): 
        param_grid = param_grid
        eval_results = objective(param_grid, dtrain, i)
        results.loc[i, :] = {
            "score": eval_results[1], 
            "params": eval_results[2], 
            "iterations": eval_results[3]
        }

        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([eval_results[1], eval_results[2], eval_results[3]])

        of_connection.close()
        
        auc_score = eval_results[1]
        
        if auc_score > best_auc:
            best_auc = auc_score
            best_params = eval_results[2]

    results.sort_values('score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
        
    return best_params, best_auc

def train_model(best_params, dtrain, dtest):
    # Train the final model with the best hyperparameters
    model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=10000,
        early_stopping_rounds=5,
        evals=[(dtrain, 'training'), (dtest, 'testing')],
        maximize=True,
        verbose_eval=2
    )
    return model


# In[ ]:




