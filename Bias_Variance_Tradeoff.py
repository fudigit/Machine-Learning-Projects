'''error'''
# Bias = E[(f(x)_hat) - f(x)^2]

'''Bias, where is it from?'''
# Bias = (  E[f(x)_hat] - f(x)  )^2
# f(x)_hat has a range of predictions for a given point x, since different underlying dataset can give different f(x)_hat target functions
# higher bias: more assumptions on the traget function, but less flexible to adapt the shape of data

# Variance = E[(f(x)_hat - E[f(x)_hat])^2]
# how much the predictions for a given point x vary, between different f(x)_hat trained from different underlying datasets
# higher variance:

#https://nbviewer.jupyter.org/github/justmarkham/DAT7/blob/master/notebooks/08_bias_variance.ipynb#Brain-and-body-weight

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:38:40 2019

@author: Di
"""

'''demo of bias and variance trade off'''

import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_table('http://people.sc.fsu.edu/~jburkardt/datasets/regression/x01.txt', 
                   sep='\s+', skiprows=33, names=['id','brain','body'], index_col='id')

df.head()
df.describe()

# get rid of the large 'outliers'
sns.lmplot(x='body', y='brain', data=df, ci=None, fit_reg=False)
df = df[df.body < 200]


# relationship between body and brain weight
sns.lmplot(x='body', y='brain', data=df, ci=None, fit_reg=False)
sns.plt.xlim(-10, 200)
sns.plt.ylim(-10, 250)

# fit a linear regression
sns.lmplot(x='body', y='brain', data=df, ci=None)
sns.plt.xlim(-10, 200)
sns.plt.ylim(-10, 250)

# set a random seed for reproducibility
np.random.seed(12345)

# randomly assign every row to either sample 1 or sample 2
df['sample'] = np.random.randint(1, 3, len(df))
df.head()

# col='sample' subsets the data by sample and creates two separate plots
sns.lmplot(x='body', y='brain', data=df, ci=None, col='sample')
sns.plt.xlim(-10, 200)
sns.plt.ylim(-10, 250)


# hue='sample' subsets the data by sample and creates a single plot
sns.lmplot(x='body', y='brain', data=df, ci=None, hue='sample')
sns.plt.xlim(-10, 200)
sns.plt.ylim(-10, 250)


sns.lmplot(x='body', y='brain', data=df, ci=None, col='sample', order=8)
sns.plt.xlim(-10, 200)
sns.plt.ylim(-10, 250)


sns.lmplot(x='body', y='brain', data=df, ci=None, col='sample', order=2)
sns.plt.xlim(-10, 200)
sns.plt.ylim(-10, 250)



