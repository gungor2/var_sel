# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:30:50 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:25:38 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:14:25 2019

@author: gungor2
"""
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor 
from itertools import combinations
from sklearn.neural_network import MLPRegressor


np.random.seed(10)
data_org = pd.read_csv("housing.data.csv",header = -1)
data = data_org.dropna()


features = data.iloc[:,0:13]
labels = data.iloc[:,13]


def model_fitting(trn_data_i,trn_label,type_ml):
    #print(trn_data_i)
    if type_ml=="linear":
        ml =  LinearRegression().fit(trn_data_i, trn_label)
    elif type_ml=="svm_rbf":
        ml = svm.SVR()
        ml.fit(trn_data_i, trn_label)
    elif type_ml=="svm_ln":
        ml = svm.SVR(kernel='linear')
        ml.fit(trn_data_i, trn_label)
    elif type_ml=="svm_sigmoid":
        ml = svm.SVR(kernel='sigmoid')
        ml.fit(trn_data_i, trn_label)
    elif type_ml=="decision_tree":
        ml = DecisionTreeRegressor(random_state = 0) 
        ml.fit(trn_data_i, trn_label)
    elif type_ml=="ANN":
        ml = MLPRegressor()
        ml.fit(trn_data_i, trn_label)
        
        
    return(ml)
        
        
    
    
MLs = ["linear","svm_rbf","decision_tree","ANN"]
#MLs = ["linear","svm_rbf"]

acc_all = []


for mll in MLs:
    
    
    folds = KFold(n_splits=5, shuffle=True)
    oof_pre = []
    RMSE_val = []
    RMSE_tr = []
    cols = features.columns.values
    n = len(features.columns.values)
    acc = []
    n_var = []
    for i in range(n):
        
        combs = list(combinations(cols, i+1))
        counter = 1
        for j in combs:
            
            f_j = list(j)
            
            features_c = features.iloc[:,f_j ]
           
            
            oof_pre = []
            RMSE_val = []
            RMSE_tr = []
        
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(features_c.values, labels.values)):
                
                #print("Fold {}".format(fold_))
            
                trn_data = features_c.iloc[trn_idx,:]
                trn_label=labels.iloc[trn_idx]
                val_data = features_c.iloc[val_idx,:]
                val_label=labels.iloc[val_idx]
                
                scaler = StandardScaler().fit(trn_data)
                               
                feature_tr_tr = scaler.transform(trn_data)
                
                val_data_tr = scaler.transform(val_data)
                
                model = model_fitting(feature_tr_tr,trn_label,mll)
                
                pre_tr = model.predict(feature_tr_tr)
                pre = model.predict(val_data_tr)
                oof_pre.extend(pre)
            
                diff_val = abs(pre - val_label.values)
                
                diff_tr = abs(pre_tr - trn_label.values)
                
                RMSE_val.append(np.mean(diff_val**2))
                RMSE_tr.append(np.mean(diff_tr**2))
            
            errs = oof_pre-labels.values
            diff = abs(oof_pre-labels.values)    
            oof_er = np.sqrt(np.mean(diff**2))
            
            acc.append(oof_er)
            n_var.append(len(f_j))
            
            print('ML is ' + mll + 
                    ' i equals to ' + str(i) + ' out of ' + str(n) + ' j equals to '+  str(counter) + ' out of ' +  str(len(combs)))
            
            counter = counter + 1
            #plt.plot(oof_pre,labels.values,'*')
    acc_all.append(acc)
 
    #plt.plot(labels.values,errs,'*')
    
    
    
    
    
    


    