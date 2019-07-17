# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:17:20 2019

@author: gungor2
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:41:21 2019

@author: gungor2
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor 
from itertools import combinations
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


np.random.seed(10)

data_type = "pima"


if data_type=="housing":
    data_org = pd.read_csv("housing.data.csv",header = -1)
    data = data_org.dropna()
    features = data.iloc[:,0:13]
    labels = data.iloc[:,13]
    pre_type = "reg"
    
elif data_type=="pima":
    data = pd.read_csv("pima-indians-diabetes.data.csv",header = -1)

    features = data.iloc[:,0:8]
    labels = data.iloc[:,8]
    pre_type = "class"


    





def model_fitting(trn_data_i,trn_label,type_ml,pre_type):
    #print(trn_data_i)
    if type_ml=="linear":
        if pre_type=="reg":
            ml =  LinearRegression().fit(trn_data_i, trn_label)
        elif pre_type == "class":
            ml =  LogisticRegression().fit(trn_data_i, trn_label)
    elif type_ml=="svm_rbf":
        if pre_type=="reg":
            ml = svm.SVR()
            ml.fit(trn_data_i, trn_label)
        elif pre_type == "class":
            ml = svm.SVC()
            ml.fit(trn_data_i, trn_label)
        
    elif type_ml=="svm_ln":
        if pre_type=="reg":
            ml = svm.SVR(kernel='linear')
            ml.fit(trn_data_i, trn_label)
        elif pre_type == "class":
            ml = svm.SVC(kernel='linear')
            ml.fit(trn_data_i, trn_label)
    elif type_ml=="svm_sigmoid":
        if pre_type=="reg":
            ml = svm.SVR(kernel='sigmoid')
            ml.fit(trn_data_i, trn_label)
        elif pre_type == "class":
            ml = svm.SVC(kernel='sigmoid')
            ml.fit(trn_data_i, trn_label)
    elif type_ml=="decision_tree":
        if pre_type=="reg":
            ml = DecisionTreeRegressor(random_state = 0) 
            ml.fit(trn_data_i, trn_label)
        elif pre_type == "class":
            ml = DecisionTreeClassifier(random_state = 0) 
            ml.fit(trn_data_i, trn_label)

    elif type_ml=="ANN":
        if pre_type=="reg":
            ml = MLPRegressor()
            ml.fit(trn_data_i, trn_label)
        elif pre_type == "class":
            ml = MLPClassifier()
            ml.fit(trn_data_i, trn_label)

        
        
    return(ml)
        
def model_eval(predicted,given,pre_type):
    if pre_type=="reg":
        diff = abs(predicted - given)
        er_model = np.sqrt(np.mean(diff**2))
    elif pre_type=="class":
        diff = predicted == given
        er_model = sum(diff)/len(diff)
        er_model = 1 - er_model
    
    return(er_model)
    
    
MLs = ["linear","svm_rbf","decision_tree","ANN"]
#MLs = ["linear","svm_rbf"]

acc_all = []
acc_all_i = []

var_all = []


folds = KFold(n_splits=5)
cols = features.columns.values
n = len(features.columns.values)

for mll in MLs:
    
    

    acc = []
    var_best = []
    acc_best_i = []
    
    #index_opt=-1
    combs = cols #combs remaining features
    for i in range(n):

        combs = list(combs)
        
        min_error = 1000000000000000000000
        index_opt=-1
       
        for j in range(len(combs)):
            
            f_j = var_best + [combs[j]]
            
            features_c = features.iloc[:,f_j]
           
            
            
            
            oof_pre = labels.copy()
            oof_pre.values[:] = -1
            
            
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(features_c.values, labels.values)):
                
                #print("Fold {}".format(fold_))
            
                trn_data = features_c.iloc[trn_idx,:]

                trn_label=labels.iloc[trn_idx]
                val_data = features_c.iloc[val_idx,:]
                val_label=labels.iloc[val_idx]
                
                scaler = StandardScaler().fit(trn_data)
                               
                feature_tr_tr = scaler.transform(trn_data)
                
                val_data_tr = scaler.transform(val_data)
                
                model = model_fitting(feature_tr_tr,trn_label,mll,pre_type)
                
                pre = model.predict(val_data_tr)
                
                
                #pre = np.array(pre)
                oof_pre.loc[val_data.index] = pre
            
            
            
            oof_err = model_eval(oof_pre.values,labels.values,pre_type)

            
            if oof_err<min_error:
                min_error = oof_err
                best_var_i = f_j
                index_opt = j
            
            acc.append(oof_err)
        
            
            print('ML is ' + mll + 
                    ' i equals to ' + str(i) + ' out of ' + str(n) + ' j equals to '+  str(j) + ' out of ' +  str(len(combs)))
            
        var_best.append(combs[index_opt])
        temp = []
        temp.extend(combs[0:index_opt])
        temp.extend(combs[index_opt+1:])
        
        combs = temp[:]
        
            
            
            #plt.plot(oof_pre,labels.values,'*')
        acc_best_i.append(min_error)
    acc_all.append(acc)
    acc_all_i.append(acc_best_i)
    var_all.append(var_best)
 
    #plt.plot(labels.values,errs,'*')
    
    
    
    
    
    


    