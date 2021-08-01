# -*- coding: utf-8 -*-
# # Ref
#
# https://flag-editors.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%8B%95%E6%89%8B%E5%81%9Alesson-8-%E8%88%87%E8%81%B7%E5%A0%B4%E6%81%AF%E6%81%AF%E7%9B%B8%E9%97%9C%E7%9A%84pearson-spearman-kendall%E7%9B%B8%E9%97%9C%E4%BF%82%E6%95%B8-%E4%B8%8A%E7%AF%87-87c93d38f27e
#
# # Summary
#
# 1. Pearson 相關係數是共變藝術的標準化 (標準化之後就不考慮單位)
# 2. Pearson 背後的假設是資料是線性分佈(a.k.a 分線性的相關捕捉能力較差，甚至壞掉)，可以說是有母數計算
# 3. Spearnman 採用排名來計算相關係數，意即非線性也可以，不藏有對母體分佈的假設，為無母數計算
#

# +
import pandas as pd
import matplotlib.pyplot as plt
X=pd.Series([1, 2, 3, 4, 5])
Y=pd.Series([6, 7, 10, 13, 21])
plt.plot(X,Y)
plt.show()

import matplotlib.pyplot as plt
X=pd.Series([1, 2, 3, 4, 5])
Y=pd.Series([6, 7, 10, 13, 21])
plt.plot(X,Y)
plt.show()
print("Pearson套件相關係數:"+str(round(X.corr(Y,method="pearson"),2)))
print("==========================================================")
print("使用Pearson公式")
print("X的平均數:"+str(round(X.mean(),2))+" , "+"Y的平均數:"+str(round(Y.mean(),2)))
print("X的變異數:"+str(round(X.var(),2))+" , "+"Y的變異數:"+str(round(Y.var(),2)))
print("X的標準差:"+str(round(X.std(),2))+" , "+"Y的標準差:"+str(round(X.std(),2)))
print("X和Y的共變異數:"+str(round(X.cov(Y),2)))
print("Pearson公式相關係數:"+str(round(X.cov(Y)/(X.std()*Y.std()),2)))Pearson套件相關係數:"+str(round(X.corr(Y,method=’pearson’),2)))
print("==========================================================")
print("使用Pearson公式")
print("X的平均數:"+str(round(X.mean(),2))+" , "+"Y的平均數:"+str(round(Y.mean(),2)))
print("X的變異數:"+str(round(X.var(),2))+" , "+"Y的變異數:"+str(round(Y.var(),2)))
print("X的標準差:"+str(round(X.std(),2))+" , "+"Y的標準差:"+str(round(X.std(),2)))
print("X和Y的共變異數:"+str(round(X.cov(Y),2)))
print("Pearson公式相關係數:"+str(round(X.cov(Y)/(X.std()*Y.std()),2)))
# -


