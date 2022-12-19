import joblib
import numpy as np
import pandas as pd
import gc
import time
import os
import sys
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from itertools import chain, product 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def model_lgb(train,test,params,stratified,num_folds,drop_features,seed_num,TARGET,vi):
    
    # start log 
    print('-'*50)
    print('>> seed_num:',seed_num)   
    print('>> drop_features:',len(drop_features))
    
    seed_everything(1)
    
    # Divide in training/validation and test data
    train_df = train.copy()
    test_df = test.copy()
    
    # ----
    
    # label encoder 
    class_le = preprocessing.LabelEncoder()
    snp_le = preprocessing.LabelEncoder()
    snp_col = [f'SNP_{str(x).zfill(2)}' for x in range(1,16)]
    snp_col = [i for i in train.select_dtypes(include=[object]).columns.tolist() if i not in ['id','class']]

    snp_data = []
    for col in snp_col:
        snp_data += list(train_df[col].values)

    # encoding - y 
    train_df[TARGET] = class_le.fit_transform(train_df[TARGET])
    snp_le.fit(snp_data)

    # encoding - x
    for col in train_df.columns:
        if col in snp_col:
            train_df[col] = snp_le.transform(train_df[col])
            test_df[col] = snp_le.transform(test_df[col])
            
    # ----
        
    # set training options
    stratified = stratified
    num_folds = num_folds

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1)

    # Create arrays and dataframes to store results
    oof_preds_lgb = np.zeros(train_df.shape[0])
    sub_preds_lgb = np.zeros((test_df.shape[0]))
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in [TARGET,'id']+drop_features]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[TARGET])):
        
        train_x, train_y = train_df[feats].iloc[train_idx], train_df[TARGET].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[TARGET].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            
            learning_rate = params['learning_rate'],
            num_leaves = int(round(params['num_leaves'])),
            colsample_bytree = params['colsample_bytree'],
            subsample = params['subsample'],
            max_depth = int(round(params['max_depth'])),
            reg_alpha = params['reg_alpha'],
            reg_lambda = params['reg_lambda'],
            min_split_gain = params['min_split_gain'],
            min_child_weight = params['min_child_weight'],
            min_child_samples = int(round(params['min_child_samples'])),    
            
            n_jobs = -1,
            n_estimators = 10000,            
            random_state = seed_num,
            objective = 'multiclass',
            silent=-1,
            deterministic=True,
            verbose=-1
        )
        
        with warnings.catch_warnings():
            
            warnings.filterwarnings('ignore')

            clf.fit(
                  train_x
                , train_y
                , eval_set=[(train_x, train_y), (valid_x, valid_y)]
                , eval_metric= 'multi_logloss'
                , verbose= -1
                , early_stopping_rounds= 100
            )
 
        oof_preds_lgb[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)
        sub_preds_lgb = clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_) / folds.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        # print('Fold %2d Accuracy : %.6f' % (n_fold + 1, accuracy_score(valid_y, oof_preds_lgb[valid_idx])))

    # Write submission file and plot feature importance
    train_df['oof'] = oof_preds_lgb   
    test_df['prob'] = pd.DataFrame(sub_preds_lgb).apply(lambda x: ''.join(str(x)),axis=1)
    test_df['pred'] = pd.Series(np.argmax(sub_preds_lgb, axis = 1))
    test_df['class'] = class_le.inverse_transform(test_df['pred'].astype(int))

    # eval 
    oof_eval = accuracy_score(train_df['class'], train_df['oof'])    
    oof_f1 = f1_score(train_df['class'], train_df['oof'], average='macro')    
    print('>> Full Accuracy score %.6f' % oof_eval)
    print('>> Full F1 score %.6f' % oof_f1)    
        
    # confusion matrix
    display(pd.crosstab(train_df['class'],train_df['oof']))

    # vi
    if vi==True:
        display(feature_importance_df.groupby(['feature'])['importance'].sum().sort_values(ascending=False).head(10))

    # save 
    test_df[['id','class']].to_csv('sub_'+str(np.round(oof_eval,8))+'.csv',index=False)
    
    return train_df,test_df[['id','class','pred','prob']]
