# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Ref" data-toc-modified-id="Ref-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ref</a></span></li><li><span><a href="#likelihood" data-toc-modified-id="likelihood-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>likelihood</a></span></li><li><span><a href="#Maximum-likelihood" data-toc-modified-id="Maximum-likelihood-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Maximum likelihood</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Summary</a></span></li><li><span><a href="#Scipy" data-toc-modified-id="Scipy-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Scipy</a></span></li><li><span><a href="#Tensorflow-probability" data-toc-modified-id="Tensorflow-probability-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Tensorflow-probability</a></span></li></ul></div>
# -

# # Ref
#
# * 世界第一簡單貝氏統計學
# * 林軒田 - L10 Logistic Regression - the likelihood

# # likelihood
#
# 又翻譯做 概度 or 可能性
#
# 舉一個極簡單的例子，有一個抽獎箱，裡面有三種球，分別代表三種獎項
#
# 舉例一個已知的生成機制(通常在真實場景下，我們是不知道母體分佈的)
#
# 抽到 A 球的機率是 $\frac{1}{4}$
# 抽到 B 球的機率是 $\frac{2}{4}$
# 抽到 C 球的機率是 $\frac{1}{4}$
#
# 小明抽了6次，抽到的獎依序為
#
# BCABAA
#
# 如果想要從小明抽到的獎推論A, B, C 球的抽中機率，該如何估計?
#
# Naive way(直接平均就對了?) :
#
# A : $\frac{3}{6}$
#
# B : $\frac{2}{6}$
#
# C : $\frac{1}{6}$
#
# likelihood :
#
# 數學上可以做一些假設，例如
#
# 1. 每次抽球之間的行為是不相關的
# 2. 球每次抽出來之後，同樣的球抽出的機率還是維持一定(例如我抽了A球，但因為箱子裡面A球太多，所以下一次抽到A球的機率還是一樣的)
#
# 那麼小明抽到獎項 
#
# BCABAA的機率就稱作為 likelihood (或稱概度、可能性)
#
# 抽中A球的機率 : $q_{A}$
# 抽中B球的機率 : $q_{B}$
# 抽中C球的機率 : $1 - q_{A} - q_{B}$
#
# 而背後的生成機制如何形成這個可能性?
#
# likelihood :
#
# $$
# ll(q_{A}, q_{B}) = q_{B}^{2}q_{A}^{3}q_{C}
# $$
#
# # Maximum likelihood
#
# 如果想要反推生成機制($q_{A}, q_{B}$)，其中一個觀點就是
#
# 怎麼樣的$q_{A}, q_{B}$，最有可能產生小明這個抽獎結果? - 這個思路就是 Maximum likelihood
#
#
# 用數學語言描述 : 你有一個多項式，有兩個變數，你想知道哪一組參數可以使得你的函數值最大
#
# $$
# max_{q_{A}, q_{B}} ll(q_{A}, q_{B})
# $$
#
# $$
# \frac{dll}{dq_{A}} = 0
# $$
#
# $$
# \frac{dll}{dq_{B}} = 0
# $$
#
#
# 由於 log 是單調函數
#
# 因此找maximum，取log還是找得到(不會變成找minimum)
#
# $$
# max_{q_{A}, q_{B}} ll(q_{A}, q_{B}) ～ max log~ll(q_{A}, q_{B})
# $$
#
# $\frac{dll}{dq_{A}} = \frac{d (3log~q_{A} + 2log~q_{B} + log~(1 - q_{A} - q_{B}))}{q_{A}} = 0$
#
# $\frac{dll}{dq_{A}} = \frac{d (3log~q_{A} + 2log~q_{B} + log~(1 - q_{A} - q_{B}))}{q_{B}} = 0$
#
#
# 連立方程式
#
# 解出
#
# $q_{A} = \frac{3}{6}$
#
# $q_{B} = \frac{2}{6}$
#
# $q_{C} = \frac{1}{6}$
#
# 和人類直覺相符
#
# 你可能會想說為何數字和真實生成機制
#
# 抽到 A 球的機率是 $\frac{1}{4}$
# 抽到 B 球的機率是 $\frac{2}{4}$
# 抽到 C 球的機率是 $\frac{1}{4}$
#
#
# 不同?
#
# 只是因為試驗的次數太少 =)
#
# # Summary
#
# likelihood : 從生成機制經過 i.i.d. 假設，把每個資料點的行程的機率都用符號表示，然後全部連乘，表示了生成機制(machenism)產生你看到的資料(data) 的機率，這條方程式就稱為 likelihood (或稱概率、可能性)
#
# maximum likelihood : 生成機制會是某些參數的函數(這會取決於我們建立模型的角度)，這些參數個別是多少的情況下，生成機制的概率(生成我手中資料地機率會最大?)，就是從這個角度去找參數
#
# 1. 微分 - 當likelihood的數學形式具有解析解
# 2. Autograd(Gradient-Descent-based) - 當likelihood的數學形式不具有解析解，就用數值解的方式
# 3. Maximum Expectation - 其中一種 optimization，用在離散變數很好用
#
# ...
#
# 最佳化方法很多，看你的library support 到什麼程度

# # Scipy

# # Tensorflow-probability


