# -*- coding: utf-8 -*-
# +
# the matirals
# A Gentle Introduction to Maximum Likelihood Estimation
# https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-9fbff27ea12f

import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats
# import pymc3 as pm3
# import numdifftools as ndt
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
# %matplotlib inline
# -

# # Data

N = 100
x = np.linspace(0,20,N)
ϵ = np.random.normal(loc = 0.0, scale = 5.0, size = N)
y = 3*x + ϵ
df = pd.DataFrame({'y':y, 'x':x})
df['constant'] = 1

sns.regplot(df.x, df.y)

# # OLS Approach

# split features and target
X = df[['constant', 'x']]
# fit model and summarize
sm.OLS(y,X).fit().summary()


# # Maximizing Log Likilihood
# * min - log likelihood
# * [`stats.norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html) 一個normal distribution的object，
# $$
# f(x) = \frac{exp(-x^{2}/2)}{\sqrt{2 \pi}}
# $$
# * `stats.norm.logpdf` : Log of the probability density function

# define likelihood function
def MLERegression(params):
    intercept, beta, sd = params[0], params[1], params[2] # inputs are guesses at our parameters
    yhat = intercept + beta*x # predictions
    # next, we flip the Bayesian question
    # compute PDF of observed values normally distributed around mean (yhat)
    # with a standard deviation of sd
    negLL = -np.sum(stats.norm.logpdf(y, loc=yhat, scale=sd) )
    # return negative LL
    return(negLL)



# let’s start with some random coefficient guesses and optimize
guess = np.array([5,5,2])
results = minimize(MLERegression, guess, method = 'Nelder-Mead', 
 options={'disp': True})

results

resultsdf = pd.DataFrame({'coef':results['x']})
resultsdf.index=['constant','x','sigma']   
np.round(resultsdf.head(2), 4)

resultsdf.loc['x','coef']

fig, ax = plt.subplots()
plt.scatter(df.x, df.y, label='data', c='blue')
plt.plot(df.x, 
     df.x * resultsdf.loc['x','coef'] + resultsdf.loc['constant','coef'],
    label='prediction',c='green')
plt.legend()


# # Fit the demand curve

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
# -

d = {'leading_probability_array':[]}
for cat_idx in range(len(category_intensity_list)):
    for dec_idx in range(len(decay_list)):
        x_data = np.arange(100)
        cat = category_intensity_list[cat_idx]
        dec = decay_list[dec_idx]
        dist = construct_demend(cat, dec)
        d['leading_probability_array'].append(dist)

sample_data = d['leading_probability_array'][0]
x = np.arange(sample_data.shape[0])
y = sample_data
plt.scatter(x, y)

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
    negLL = -np.sum(stats.norm.logpdf(y, loc=yhat, scale=sd) )
    # return negative LL
    return(negLL)


# let’s start with some random coefficient guesses and optimize
guess = np.array([3,0.5,2,2,np.pi, 2])
results = minimize(MLERegression, guess, method = 'Nelder-Mead', 
 options={'disp': True})

results

resultsdf = pd.DataFrame({'coef':results['x']})
resultsdf.index=['alpha','beta','rho','omega','phi','sigma']
np.round(resultsdf, 4)

params = np.round(resultsdf, 4)['coef'].tolist()
params[5]


# +
def hypothesis(params, x):
    alpha, beta, rho, omega, phi, sd = params[0], params[1], params[2], params[3], params[4], params[5] # inputs
    f_gamma = (beta ** alpha) * (x ** (alpha - 1)) * np.exp(-beta * x) / gamma(alpha)
#     if f_gamma == np.inf:
#         f_gamma = np.exp(500)
#     elif f_gamma == -np.inf:
#         f_gamma =  - np.exp(500)
    f_sin = rho * np.sin(omega * x + phi)
    yhat = f_gamma * (1 + f_sin)
    return yhat
    
params = np.round(resultsdf, 4)['coef'].tolist()
y_pred = hypothesis(params=params, x=np.arange(1, 101))
# -

fig, ax = plt.subplots()
plt.scatter(x, y, label='data', c='blue')
plt.plot(x, y,label='prediction',c='green')
plt.legend()

# # Points of knowledge

# * **MLE can be seen as a special case of the maximum a posteriori estimation (MAP) that assumes a uniform prior distribution of the parameters, or as a variant of the MAP that ignores the prior and which therefore is unregularized.**
# * `posterior = likelihood x prior / evidence` special case comes from prior dist = 1 (uniform) and evidence dist = 1 
#
# * The distinction between probability and likelihood is fundamentally important: **Probability attaches to possible results; likelihood attaches to hypotheses.**


