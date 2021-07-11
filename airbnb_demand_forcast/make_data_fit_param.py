# -*- coding: utf-8 -*-
# +
'''
we have groud-truth y of 196 category demand
we have a statistical model Gamma*cos

Leads : fit the model using maximum likelihood method.
1. expand numpy as features in to dataframe, using Logistic Regression
2. fit distribution by scipy (M.L.E.)
    分成兩種 
        - parametric fit - 有確定統計模型以及假設
        - non parametric fit - KDE fitting

3. train them all with ML model! might separated with simialry category
check the optimize documentation https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
'''
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats
from math import log2
# from sklearn.metrics import log_loss

# %matplotlib inline
# -

# # Data

# +
def np_exp(magnitude, k : float, x : 'np.array'):
    '''
    exp func
    leading time major component
    magnitude 初始強度 - based on 房間種類
    k 衰減速率
    一開始最多，後來逐漸變小，收斂時希望還能看到seasonality
    '''
    return magnitude * np.exp(k * x)

def np_cos(magnitude : 1.0, peroid : float, x : 'np.array'):
    '''
    cos function
    weekly seasonality component
    prroid不變維持7，
    magmitude隨著linear variable exponentail decay 但希望到最後(~ 100)還是有可見的變化
    '''
    return magnitude * np.cos(2 * np.pi / peroid * x) + 1



def construct_demend(category_intensity, decay, noise=False):
    peroid = 7
    x = np.arange(100)
    y = np_exp(magnitude = category_intensity, k = decay, x = x) *\
     np_cos(magnitude = category_intensity, peroid = peroid, x = x)
    if noise:
        ϵ = np.random.normal(loc = 0.0, scale = 0.03, size = 100)
        y += ϵ
    return y / y.sum()



np.random.seed(2)
N = 14
decay_list = np.random.randint(5, 20, size=N) / -100 # 先決定decay
multiple_factor = 2 * (np.random.random(size= N)) + 2 # 決定比例
category_intensity_list = abs(decay_list * multiple_factor) # 決定房型強度



d = {'leading_probability_array':[]}
for cat_idx in range(len(category_intensity_list)):
    for dec_idx in range(len(decay_list)):
        x_data = np.arange(100)
        cat = category_intensity_list[cat_idx]
        dec = decay_list[dec_idx]
        dist = construct_demend(cat, dec)
        d['leading_probability_array'].append(dist)


# -

len(d['leading_probability_array'])
rdn_idx_pick = np.random.choice(len(d['leading_probability_array']), size=10)

# +
# sample of data, plot
fig, ax = plt.subplots(nrows=2,
                       ncols=5, figsize=(12,8), sharey=True)

for cat_idx in range(2):
    for dec_idx in range(5):
        x_data = np.arange(100)
        dist_idx = 5 * cat_idx + dec_idx
        dist = d['leading_probability_array'][rdn_idx_pick[dist_idx]]
        ax[cat_idx][dec_idx].plot(x_data, dist, alpha=0.7)
        ax[cat_idx][dec_idx].set_title(f"data idx{dist_idx}")
        
plt.tight_layout()
# -

# # MLE

# define likelihood function
from scipy.special import gamma, beta
def MLERegression(params):
    alpha, beta, rho, omega, phi, sd = params[0], params[1], params[2], params[3], params[4], params[5] # inputs
    f_gamma = (beta ** alpha) * (x ** (alpha - 1)) * np.exp(-beta * x) / gamma(alpha)
    f_sin = rho * np.sin(omega * x + phi)
    yhat = f_gamma * (1 + f_sin)
        
    # next, we flip the Bayesian question
    # compute PDF of observed values normally distributed around mean (yhat)
    # with a standard deviation of sd
    negLL = -np.sum(stats.norm.logpdf(y, loc=yhat, scale=sd))
    # return negative LL
    return(negLL)


import pickle
# pickle.dump(param_result, open( "param.pkl", "wb" ) )
param_result = pickle.load( open( "param.pkl", "rb" ) )

if not param_result:
    print('training process....')
    param_result = {}

    for data_idx in range(0, 196):
        sample_data = d['leading_probability_array'][data_idx]

        x = np.arange(sample_data.shape[0])
        y = sample_data

        # let’s start with some random coefficient guesses and optimize
        guess = np.array([3,0.5,2,2,np.pi, 2])
        # linear method, should failed...
        results = minimize(MLERegression, guess, method = 'Nelder-Mead', options={'disp': True}) 
        if results['success'] == True:
            param_result[data_idx] = results['x']
        else:
            param_result[data_idx] = None
        print(results['message'], results['x'])
else:
    print('Skip training, loading model params from pkl')

# +
success = []
for k, v in param_result.items():
    if not v is None:
        success.append(k)

print('total : ', len(param_result.keys()),
      'successed : ', len(success), sep='\n\n')
# -

param_result

# +
# Minor Issue
# TODO 
# 目前的優化都是失敗的，使用傳統的graidnet descents，先前的t-SNE可以copy過來用 ...(1)
# 關注那些失敗的優化，如果是梯度爆炸，則進入scipy.optimized該方法中，加上gradient clipping ...(2)
# -

# # Result / Error Analysis
# * Cross Entropy Loss
# * Plot

to_show_data_idx = np.random.choice(success, size=10)
print(to_show_data_idx)

# +
# numpy 計算一下 Dist and Cross Entropy
# 原本result裡面就有? - 有的，但是沒存下來，所以回到上面的解法，用numpy重建
# scipy OPtimizedResult https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult

x = np.arange(100)

def Pred_Demand(x, params):
    alpha, beta, rho, omega, phi = params[0], params[1], params[2], params[3], params[4]
    # the sd term gives us the uncertainty of prediction, which is nice
    # but for simplification, we just pick the mean value
    print(alpha, beta, rho, omega, phi)
    f_gamma = (beta ** alpha) * (x ** (alpha - 1)) * np.exp(-beta * x) / gamma(alpha)
    f_sin = rho * np.sin(omega * x + phi)
    yhat = f_gamma * (1 + f_sin)
    return yhat


def cross_entropy(p, q):
    return -sum([p[i]*np.log2(q[i]) for i in range(len(p))])

def plot_result_and_ground_truth(x, data_idx, groued_truth_dict, pred_param_dict):
    ground_truth = groued_truth_dict['leading_probability_array'][data_idx]
    pred_param = pred_param_dict[data_idx]
    print(data_idx,pred_param)
    pred = Pred_Demand(x=x, params=pred_param)
    # TODO tiny number cannnot be caculate,...
    cross_entropy_loss = np.round(cross_entropy(ground_truth, pred), 5)
    
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x, ground_truth,'-b', lw=3, label='ground truth')
    ax.plot(x, pred ,'--g', lw=3, label='pred')
    ax.set_xlabel("Leading time (days)")
    ax.set_ylabel("Demand")
    ax.set_title(f'data_idx {data_idx} cross entropy loss {cross_entropy_loss}')
    ax.legend()


# print(param_result.keys())
# print(param_result[0])
# Pred_Demand(param_result[0])
for data_idx in to_show_data_idx:
    print(data_idx)
    plot_result_and_ground_truth(x=x,
                             data_idx=data_idx,
                             groued_truth_dict=d,
                             pred_param_dict=param_result)
# -

# # Kolmogorov-Smirnov test

# +
# TODO
# based on numpy, 不需建立scipy.stats的rv.continous的物件之上(那個物件太難用)
# google key word
# https://www.google.com/search?q=Kolmogorov-Smirnov+test+numpy+approach&oq=Kolmogorov-Smirnov+test+numpy+approach&aqs=chrome..69i57.4565j0j1&sourceid=chrome&ie=UTF-8
# so far scipy document
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
# -


