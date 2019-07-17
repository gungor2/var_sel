# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:14:25 2019

@author: gungor2
"""
import pandas as pd



data = pd.read_csv("pima-indians-diabetes.data.csv",header = -1)

features = data.iloc[:,0:8]
labels = data.iloc[:,8]

