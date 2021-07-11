# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import statsmodels.datasets

# %matplotlib inline
data = statsmodels.datasets.heart.load_pandas().data

# heart transplant
data.tail()

# survival
data = data[data.censors == 1]
survival = data.survival
survival.tail()

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(sorted(survival)[::-1], "o")
ax1.set_xlabel("Patient")
ax1.set_ylabel("Survival time (days)")

# our distribution
ax2.hist(survival, bins=15)
ax2.set_xlabel("Survival time (days)")
ax2.set_ylabel("Number of patients")

# -

# # Statistical model fitting
# * [exponential](https://zh.wikipedia.org/wiki/%E6%8C%87%E6%95%B0%E5%88%86%E5%B8%83) $f(x ;\lambda) = \lambda e^{-\lambda x},~~x \geq 0$
# * 假設 : 現有資料皆$s_{i}$獨立地從指數分佈中抽取出來，每筆抽樣資料對應到一個機率，所有事件一起發生的可能性為每筆資料發生的機率相乘， likelihood function   $ L$
#
#
# $$
# \begin{align}
# & L(\lambda, \{s_{i}\})\\
# &= \Pi_{i=1}^{n}P(s_{i}|\lambda)~~~~~~~~(by~independence~~of~~the~~s_{i}) \\
# &= \Pi_{i=1}^{n} \lambda exp(- \lambda s_{i}) \\
# &= \lambda^{n} exp(-\lambda \sum_{i=1}^{n}s_{i}) \\
# &= \lambda^{n} exp(-\lambda n \bar{s})
# \end{align}
# $$
# where $\bar{s}$ is the sample mean
# * 一次微分=0找參數
# $$
# \begin{align}
# \frac{dL(\lambda, \{s_{i}\})}{d\lambda} = \lambda^{n-1} exp(\lambda n \bar{s})(n - n \lambda \bar{s})
# \end{align}
# $$
#
# $$
# \begin{align}
# \frac{dL(\lambda), \{s_{i}\}}{d\lambda} = 0
# \end{align}
# $$
# $$
# \begin{align}
# \lambda = \frac{1}{\sqrt{s}}
# \end{align}
# $$
# * 其實就可以用gradient descent來解
# * scipy有numberical solution來解，也不一定要gradient descent

# ## Analytical solution

smean = survival.mean()
rate = 1.0 / smean

smax = survival.max()
days = np.linspace(0.0, smax, 1000)
# bin size: interval between two
# consecutive values in `days`
dt = smax / 999.0

# lambda
print(1.0 / rate)

dist_exp = st.expon.pdf(days, scale=1.0 / rate)
print(dist_exp[:5], type(dist_exp), dist_exp.shape, sep="\n\n")

nbins = 30
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(survival, nbins, label="data")
ax.plot(
    days,
    dist_exp * len(survival) * smax / nbins,
    "-r",
    lw=3,
    label="hypothsis after MLE",
)
ax.set_xlabel("Survival time (days)")
ax.set_ylabel("Number of patients")
plt.legend()

# ## Numerical Solution using Scipy

# +
dist = st.expon
args = dist.fit(survival)

print(dist, dir(dist), args)
# -

# ## Compare analytical solution and numerical solution

# +
# 幾乎重疊再一起的解析解與數值解
dist_from_exp_num = dist.pdf(days, *args)

nbins = 30
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(survival, nbins, label="data")
ax.plot(
    days,
    dist_exp * len(survival) * smax / nbins,
    "-r",
    lw=1,
    alpha=0.6,
    label="MLE analytical solution",
)

ax.plot(
    days,
    dist_from_exp_num * len(survival) * smax / nbins,
    "-g",
    lw=1,
    alpha=0.6,
    label="MLE numerical solution",
)

ax.set_xlabel("Survival time (days)")
ax.set_ylabel("Number of patients")
ax.set_ylim(0, 10)
plt.legend()

# -

# ## Kolmogorov-Smirnov test
# * 檢驗一組所觀測到的資料是否來自於我們的假設分佈，ruturn `p-value`表示可能的機率

# ## TODO
# * 一般來說 KS test的機率多少稱為significant?

# trival solution
# 自己分佈產生一組random variable然後放進自己的statistical model
sample_data = dist.rvs(size=1000, *args)
result = st.kstest(sample_data, dist.cdf, args)
print(args, result, sep="\n\n")

result = st.kstest(survival, dist.cdf, args)
print(survival[:5], dist.cdf, args, result, sep="\n\n")

# * null hypothesis : 所觀測到的資料是來自我們所提供的分佈以及參數的機率 `p=8.6e-06` 機率爆炸低，不太可能

# ## Birnbaum-Sanders distribution

dist = st.fatiguelife
args = dist.fit(survival)
st.kstest(survival, dist.cdf, args)

# * null hypothesis : 所觀測到的資料是來自我們所提供的分佈以及參數的機率 `p=7.32%` 機率稍為高，depends on 怎樣的信心水準是有效的建模

dist_fl = dist.pdf(days, *args)
nbins = 30
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(survival, nbins)
ax.plot(days, dist_exp * len(survival) * smax / nbins, "-r", lw=3, label="exp")
ax.plot(days, dist_fl * len(survival) * smax / nbins, "--g", lw=3, label="BS")
ax.set_xlabel("Survival time (days)")
ax.set_ylabel("Number of patients")
ax.legend()
