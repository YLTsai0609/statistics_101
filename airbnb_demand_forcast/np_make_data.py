'''
ground turth 設計
共計200種房型
|房型編號|leading_probability_array (shape=100,)|
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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



def construct_demend(category_intensity, decay):
    peroid = 7
    x = np.arange(100)
    y = np_exp(magnitude = category_intensity, k = decay, x = x) *\
     np_cos(magnitude = category_intensity, peroid = peroid, x = x)
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
df = pd.DataFrame(d)
df.to_csv('data/ground_truth.csv', index=False)
print('Done')

# Check data
# fig, ax = plt.subplots(nrows=len(category_intensity_list),
#                        ncols=len(decay_list))
# for cat_idx in range(len(category_intensity_list)):
#     for dec_idx in range(len(decay_list)):
#         x_data = np.arange(100)
#         cat = category_intensity_list[cat_idx]
#         dec = decay_list[dec_idx]
#         dist = construct_demend(cat, dec)
#         print(dist.sum())
#         ax[cat_idx][dec_idx].plot(x_data, dist, label = f"cat = {np.round(cat, 2)}\n dec={dec}", alpha=0.6)
#         ax[cat_idx][dec_idx].legend(fontsize=12)

# plt.tight_layout()
# plt.show()