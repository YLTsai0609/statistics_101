# -*- coding: utf-8 -*-
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Ref" data-toc-modified-id="Ref-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ref</a></span></li><li><span><a href="#Sampling-from-gaussian-populations" data-toc-modified-id="Sampling-from-gaussian-populations-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Sampling from gaussian populations</a></span></li><li><span><a href="#Summary-:" data-toc-modified-id="Summary-:-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Summary :</a></span></li></ul></div>

# # Ref
#
# [wiki](https://zh.wikipedia.org/wiki/%E4%B8%AD%E5%BF%83%E6%9E%81%E9%99%90%E5%AE%9A%E7%90%86)
#
# [post](https://medium.com/qiubingcheng/%E4%B8%AD%E5%A4%AE%E6%A5%B5%E9%99%90%E5%AE%9A%E7%90%86-central-limit-theorem-clt-c5e47d091865)
#
#
# 在適當條件下，從任何母體隨機抽取大量獨立的隨機變數，其**平均值的分佈**會趨近於常態分佈

# %config Completer.use_jedi = False


# +
import numpy as np
import pandas as pd
import random


from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import cufflinks as cf
import plotly.express as px
import cufflinks
cf.go_offline(connected=True)
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# %matplotlib inline
# -

# # Sampling from gaussian populations

# +
# 高斯母體, mean = 10, sigma = 5, population_size = 20000

gaussian_population = [random.gauss(10, 5) for _ in range(20000)]

print(gaussian_population[:20], np.mean(gaussian_population), sep='\n\n')
# -

# approach :
#
# 1. 每次抽樣的樣本數(10, 50, 100, 200)
# 2. 對抽出樣本取平均
# 3. 每一輪的重複抽樣次數 : 10次
# 4. 畫圖
#
# findings:
#
# 隨著抽樣樣本數的增加，樣本平均會趨近於常態分佈，且變異數會越來越小
#
# 重抽樣次數 $n$, 母體標準差 $\sigma$
#
# 樣本平均數的標準差 $\frac{\sigma}{\sqrt{n}}$
#

# +
sample_size_list = [10,50,100,200]

result = {}
for sample_size in sample_size_list:
    l = []
    for _ in range(10):
        x_bar = sum(random.sample(gaussian_population,sample_size))/sample_size
        l.append(x_bar)
    result[sample_size] = l

for i,sample_size in enumerate(result):
    plt.figure()
    p = sns.distplot(result[sample_size])
    p.set(xlim=(8, 12))

# +
# unirform 母體, 1~10皆有，完全同樣的比重, population size 1000

x = np.random.randint(1,10+1,size=1000)
print(x[:20], np.mean(x), sep='\n\n')
# -

print(sample_size)
sum(np.random.randint(1, 10 + 1,sample_size))

result

# +
sample_size_list = [10,50,100,200]

result = {}
for sample_size in sample_size_list:
    l = []
    for _ in range(10):
        x_bar = sum(np.random.randint(1, 10 + 1,sample_size))/sample_size
        l.append(x_bar)
    result[sample_size] = l

for i,sample_size in enumerate(result):
    plt.figure()
    p = sns.distplot(result[sample_size])
    p.set(xlim=(1, 9))
# -

# # Summary :
# 1. 
#
#     進行抽樣估計時(例如Precision, Recall 在小樣本上的表現)
#
#     若能夠相當掌握資料變異度(量化以 $\sigma$ 表示)
#
#     那麼抽樣的大小越大，估計越準，且符合，抽樣平均值的標準差
#
#     重抽樣次數 $n$, 母體標準差 $\sigma$
#
#     $\frac{\sigma}{\sqrt{n}}$
#
# <br>
#
# 2. 
#     進行抽樣估計時，若考慮的是某個在意的平均值(流失率均值、媒合率等)
#     
#     無論原本分佈是什麼分佈，該**均值分佈**會符合高斯分佈
#     
#     這也就是 Central Limit Theorem(CLT)的威力

# # Additional
#
# * exponential dist / bi-norm dist

# +
# exp, population_size = 20000


exp_population = [random.expovariate(10) for _ in range(20000)]


def make_n_peak_population(params : list):
    dist = []
    for p in params:
        mu,sigma,pop = p['mu'], p['sigma'], p['pop']
        dist.extend(
            [
                random.gauss(mu, sigma)
                for _ 
                in range(pop)
            ]
        )
    return dist

bi_norm_dist = make_n_peak_population(params=[
    {'mu':10,'sigma':5,'pop':20000},
    {'mu':200,'sigma':20,'pop':20000},
])


skew_bi_norm_dist = make_n_peak_population(params=[
    {'mu':10,'sigma':5,'pop':50},
    {'mu':200,'sigma':20,'pop':3000}, # factor = 6
])

print(exp_population[:20], np.mean(exp_population), sep='\n\n')
print()
print(bi_norm_dist[:20],np.mean(bi_norm_dist),sep='\n\n')
print()
print(skew_bi_norm_dist[:20],np.mean(skew_bi_norm_dist),sep='\n\n')


# +
def boostraping_plot(
    dist : np.array,
    mark : str,
    sample_size_list : list,
    n_trails : int = 10) -> None:
    result = {}
    for sample_size in sample_size_list:
        l = []
        for _ in range(n_trails):
            x_bar = sum(random.sample(dist,sample_size))/sample_size
            l.append(x_bar)
        result[sample_size] = l

    for i,sample_size in enumerate(result):
        plt.figure()
        p = sns.distplot(result[sample_size],)
        p.set(title=f'dist : {mark}, sample_size : {sample_size}, n_trails : {n_trails}')
    
    
sample_size_list = [10,50,100,200,500]
n_trails = [3,5,10]

for dist,mark in zip(
    [
        exp_population,
        bi_norm_dist,
        skew_bi_norm_dist
    ],
    [
        'exp',
        'bi_norm',
        'skew_bi_norm'
    ]
    ):
    for t in n_trails:
        boostraping_plot(dist=dist,
                         mark=mark,
                         sample_size_list=sample_size_list,
                         n_trails=t)
# -


