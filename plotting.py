# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:09:43 2019

@author: gungor2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import matplotlib.ticker as mtick
import importlib.machinery
path_code = 'C:\\Users\\SupErman\\Dropbox\\research_misccalenours\\code\\plot_editor_py.py'

path_code = 'D:\\Dropbox\\research_misccalenours\\code\\plot_editor_py.py'

dirr_save = "D:\Dropbox\kaggle\PIMA_Blog"




loader = importlib.machinery.SourceFileLoader('report', path_code)
handle = loader.load_module('report')

Algos = ['Linear','SVM_rbf','Dec. Tree','ANN']

for i in range(4):
    
    
    fig2 = plt.figure(1)
    ax1 = fig2.subplots()

#    
    if pre_type=="reg":
        ax1.plot(n_var,np.array(acc_all[i]),'ro',markersize = 7,label = Algos[i])
        ax1.plot(list(range(1,n_var[len(n_var)-1]+1)),np.array(acc_all_i[i]),'b')
        
    
        min_x = min(n_var)*0.5
        max_x = max(n_var)*1.02
        min_y = min(acc_all[i])*0.95
        max_y = max(acc_all[i])*1.03
        fig2= handle.plot_editor('ggplot',fig2,ax1,'','Number of Variables',r'RMSE',min_x,max_x,min_y,max_y,1,'small',"blog",'reg')
        fig2.savefig(dirr_save   + "\\reg_" + Algos[i] + ".png", bbox_inches="tight",dpi = 700)
        fig2.clear()
    
#    
    elif pre_type=="class":
        ax1.plot(n_var,1-np.array(acc_all[i]),'ro',markersize = 7,label = Algos[i])
        ax1.plot(list(range(1,n_var[len(n_var)-1]+1)),1-np.array(acc_all_i[i]),'b')
    
        min_x = min(n_var)
        max_x = max(n_var)
        min_y = min(acc_all[i])
        max_y = max(acc_all[i])
        fig2= handle.plot_editor('ggplot',fig2,ax1,'','Number of Variables',r'Accuracy (%)',min_x,max_x,min_y,max_y,1,'small',"blogs",'reg')
        fig2.savefig(dirr_save   + "\\class_" + Algos[i] + ".png", bbox_inches="tight",dpi = 700)
        fig2.clear()