# -*- coding: utf-8 -*-
'''
0221 Fri
按照各種房型分類，有下訂時間，按照距離入房日及訂房日拉出leading time，並在各種房型中取histogram
Q : bins間隔?
確認過資料，calendar, reviews, listing 三者無法滿足所需，決定先使用模擬的dataset

0221 Fri scipy.stats的rv_continous很難用.....
透過scipy建立自訂的pdf，再進行放大
based on "how to build a custom distribution on stackoverflow"
https://stackoverflow.com/questions/46055690/python-how-to-define-customized-distributions

st.rv_continous用來建立自訂分佈的隨機亂數
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html

0221 直接用numpy吧!

0222
numpy random sampling
https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html

exopnential + cos 
共用震幅強度, peroid = 7, 參數控制 震幅強度, decay
1. cat對於expoenetal沒有影響，伸縮的部份最後會被約掉 - 每一col的y ticks都一致沒有變動
2. cat對於cos有影響，需考量是否會大於1，大於一是不合理的，越接近1，seasonality的效應越大
   取值應該在0~1之間，合理的seasonality應小於0.2，取值在0.1~0.2之間
3. decay越小，long tail越長，表示越往後面仍然有訂房，合理取值在-0.05 ~ -0.2之間
4. decay和cat有交互作用，比例取絕對值，abs (cat / dec) 越大，seasonality效應越強，
   合理情況下，seasonality應該要小於dec效應，合理取值在2倍~4倍之間

ground turth 設計
共計200種房型
|房型編號|leading_probability_array (shape=100,)|
'''
import pandas as pd
import numpy as np
from math import gamma
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(2)
# rd_variable = np.random.random(1000) # 0 ~ 1 random variable
linear_variable = np.arange(100)
gamma_dis = np.random.gamma(1, 2, 100)
def np_exp(magnitude, k : float, x : 'np.array'):
    '''
    exp func
    leading time major component
    magnitude 初始強度 - based on 房間種類
    k 衰減速率
    一開始最多，後來逐漸變小，收斂時希望還能看到seasonality
    '''
    return magnitude * np.exp(k * x)

def np_gamma(shape : float, scale : float, x:'np.array'):
    '''
    Test Case
    gamma(0.5) ~ 1.77 ~ 3.13 ** 0.5
    分子(x=0, shape=1, scale=2) = 1 * 2 * 1
    https://zh.wikipedia.org/wiki/%E4%BC%BD%E7%8E%9B%E5%88%86%E5%B8%83
    '''
    return (x ** (shape - 1)) * (scale ** shape) * (np.exp(-scale * x)) / gamma(shape)
def np_cos(magnitude : 1.0, peroid : float, x : 'np.array'):
    '''
    cos function
    weekly seasonality component
    prroid不變維持7，
    magmitude隨著linear variable exponentail decay 但希望到最後(~ 100)還是有可見的變化
    '''
    return magnitude * np.cos(2 * np.pi / peroid * x) + 1
def noise_level(level : float, shape : int):
    return level * np.random.random(shape) 


N = 100

Y1 = np_exp(magnitude = 0.2, k = -0.03, x = linear_variable)
Y2 = np_cos(magnitude = 0.2, peroid = 7, x = linear_variable)
Y3 = Y1*Y2 
Y3 = Y3 / Y3.sum() # normolization become density distribution
Y4 = np_exp(magnitude = 0.2, k = -0.07, x = linear_variable) *\
     np_cos(magnitude = 0.2, peroid = 7, x = linear_variable)
Y4 = Y4 / Y4.sum()

Y5 = np_exp(magnitude = 0.2, k = -0.01, x = linear_variable) *\
     np_cos(magnitude = 0.2, peroid = 7, x = linear_variable)
Y5 = Y5 / Y5.sum()

def construct_demend(category_intensity, decay):
    peroid = 7
    x = np.arange(100)
    y = np_exp(magnitude = category_intensity, k = decay, x = x) *\
     np_cos(magnitude = category_intensity, peroid = peroid, x = x)
    return y / y.sum()

# category_intensity_list = [0.1, 0.13, 0.30, 0.42]
decay_list = np.random.randint(5, 20, size=(4)) / -100 # 先決定decay
multiple_factor = 2 * (np.random.random(size=4)) + 2 # 決定比例
category_intensity_list = abs(decay_list * multiple_factor) # 決定房型強度


fig, ax = plt.subplots(nrows=len(category_intensity_list),
                       ncols=len(decay_list))
for cat_idx in range(len(category_intensity_list)):
    for dec_idx in range(len(decay_list)):
        x_data = np.arange(100)
        cat = category_intensity_list[cat_idx]
        dec = decay_list[dec_idx]
        dist = construct_demend(cat, dec)
        print(dist.sum())
        ax[cat_idx][dec_idx].plot(x_data, dist, label = f"cat = {np.round(cat, 2)}\n dec={dec}", alpha=0.6)
        ax[cat_idx][dec_idx].legend(fontsize=12)
# ax[2,1].set_xlabel('$\\theta$', fontsize=14)
# ax[1,0].set_ylabel('$p(y|\\theta)$', fontsize=14)
# n, bins, patches = ax[0].hist(X1, 50, normed=1, facecolor='green', alpha=0.75)
# ax[0].plot(linear_variable, Y3, 50, color='green', alpha=0.75)
# ax[1].plot(linear_variable, Y4, 50, color='blue', alpha=0.75)
# ax[2].plot(linear_variable, Y5, 50, color='red', alpha=0.75)
# # ax[3].plot(linear_variable, Y6, 50, color='black', alpha=0.75)
# ax[0].set_ylim(0, 0.1)
# ax[1].set_ylim(0, 0.1)
# ax[2].set_ylim(0, 0.1)
# ax[3].set_ylim(0, 0.04)
# ax[2].set_yscale('log')
plt.tight_layout()
plt.show()



