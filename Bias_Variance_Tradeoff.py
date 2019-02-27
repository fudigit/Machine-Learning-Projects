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


import pandas as pd
import seaborn as sns

df = pd.read_csv('http://people.sc.fsu.edu/~jburkardt/datasets/regression/x01.txt', sep='\s+', skiprows=33, names=['id','brain','body'], index_col='id')

print(df.head())
print(df.describe())

#only focus a smaller subset of data

sns.lmplot(x = 'body', y = 'brain', data =df, ci = None, fit_reg = False)
sns.
