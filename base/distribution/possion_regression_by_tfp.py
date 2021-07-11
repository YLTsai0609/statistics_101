# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="J33av_vA1Y3m"
# # Modelling count data with Tensoflow Probability
#
# **Goal:** In this notebook you will again work with TFP. You will set up regression models that are able to output different conditional probability distributions to model count data. You will define different models with Keras, sklearn and the Tensorflow probability framework and optimize the negative log likelihood (NLL).
# You compare the performace of the Poisson regression vs. the linear regression on a test dataset. Finally, you will extend the Poisson model to the zero-inflated Poisson model and compare the NLL of all models.
#
# **Usage:** The idea of the notebook is that you try to understand the provided code by running it, checking the output and playing with it by slightly changing the code and rerunning it.
#
# **Dataset:** 
# You work with a camper dataset form https://stats.idre.ucla.edu/r/dae/zip/. The dataset contains data on 250 groups that went to a park. Each group was questioned about how many fish they caught (count), how many children were in the group (child), how many people were in the group (persons), if they used a live bait  and whether or not they brought a camper to the park (camper).
# You split the data into train and test dataset.
#
# **Content:**
# * Work with different distributions in TFP: Normal, Poisson and zero-inflated Poisson
# * Load and split the camper dataset 
# * Fit different regression models to the camper train dataset: linar regression, Poisson regression and zero-inflated Poisson regression
# * Plot the predicted probability distributions (CPD) for two specific datapoints along with their likelihood
# * Plot the testdata along with the predicted mean and the 2.5% and 97.5% percentiles of the predicted CPD
# * Compare the different models based on the test NLL 
#
# | [open in colab](https://colab.research.google.com/github/tensorchiefs/dl_book/blob/master/chapter_05/nb_ch05_02.ipynb)
#

# + id="Fv5u8wSX1Y3p"
try: #If running in colab 
    import google.colab
    IN_COLAB = True 
    %tensorflow_version 2.x
except:
    IN_COLAB = False

# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="ZQM8G3oPd9GE" outputId="b5921d61-3436-498b-e657-15103112bfe1"
import tensorflow as tf
if (not tf.__version__.startswith('2')): #Checking if tf 2.0 is installed
    print('Please install tensorflow 2.0 to run this notebook')
print('Tensorflow version: ',tf.__version__, ' running in colab?: ', IN_COLAB)

# + colab={"base_uri": "https://localhost:8080/", "height": 124} id="bPlT84Mqd9GZ" outputId="2ead57fd-8645-48ea-ea76-1b5bdb452611"
# !pip install tensorflow_probability==0.8.0

# +
# The tensorflow probability

# https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html
# https://www.tensorflow.org/probability?hl=zh-tw

# + [markdown] id="DOhO_5Pt-N9E"
# #### Imports

# + colab={"base_uri": "https://localhost:8080/", "height": 52} id="WLP-37UY1Y31" outputId="ca4cfe36-0828-4391-d4e9-a6cd882b59e1"
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_probability as tfp

# %matplotlib inline
plt.style.use('default')

tfd = tfp.distributions
tfb = tfp.bijectors
print("TFP Version", tfp.__version__)
print("TF  Version",tf.__version__)
np.random.seed(42)
tf.random.set_seed(42)

# + [markdown] id="ecQD6xNF1Y38"
# ### Working with a TFP Poisson distribution
#
# Here you can see a small example how to work with a Poisson distribution in TFP. The Poisson distribution has only one parameter, often called $\lambda$ or rate, which defines the mean and the variance of the distribution. We set the rate $\lambda$ to 2, and plot the probability distribution for the values 0 to 10. Below in the notebook you will define a model to learn this parameter.
#
# #### Listing 5.5: The Poisson Distribution in TFP 
#

# + colab={"base_uri": "https://localhost:8080/", "height": 540} id="NMzlJ8mO1Y39" outputId="2afe9a8b-012f-4d7c-bd2a-6ccab2dfa27d"
dist = tfd.poisson.Poisson(rate = 2) #A
vals = np.linspace(0,10,11) #B
p = dist.prob(vals) #C
print(dist.mean().numpy())  #D
print(dist.stddev().numpy())   #E

plt.xticks(vals)
plt.stem(vals, p)
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.show()

#A Poisson distribution with parameter rate = 2
#B Integer values from 0 to 10 for the x-axis 
#C Computes the probability for the values
#D The mean value yielding 2.0
#E The standard deviation yielding sqrt(2.0) = 1.41...


# + [markdown] id="18dtcjMG1Y4E"
# ### Loading real count data
#
# Here you load the camper data from: https://stats.idre.ucla.edu/r/dae/zip/. The traget variable is the number of fish caught, during a state park visit by a group. You have data of 250 groups that went to the park. Each group was questioned about how many fish they caught (count), how many children were in the group (child), how many people were in the group (persons), if they used a live bait (livebait) and whether or not they brought a camper to the park (camper). This will be the features.
# You randomly split the data into train and test dataset (80% train and 20% test).
# -

# save the data to avoid broken resource
import pandas as pd
(
    pd.read_csv('https://raw.githubusercontent.com/tensorchiefs/dl_book/master/data/fish.csv')
    .to_csv('data/fish.csv',index=False)
)

# + id="eQuwzZUj1Y4L"
# The Fish Data Set
# See example 2 from https://stats.idre.ucla.edu/r/dae/zip/ 
#"nofish","livebait","camper","persons","child","xb","zg","count"
dat = np.loadtxt('https://raw.githubusercontent.com/tensorchiefs/dl_book/master/data/fish.csv',delimiter=',', skiprows=1)
X = dat[...,1:5] #"livebait","camper","persons","child
y = dat[...,7]
X=np.array(X,dtype="float32")
y=np.array(y,dtype="float32")

# + id="_58wisLu1Y4S"
# Uncomment the next line, to enhance the ZIP model (see below why you would like to do it)
# n = len(y)
# idx = np.random.permutation(n)[0:int(n*0.3)] 
# y[idx] = 0

# + [markdown] id="ILaYEVVnBN3F"
# Let's split the data and look at the counts (how many fish each group caught).
#

# + colab={"base_uri": "https://localhost:8080/", "height": 141} id="RCdA9LKL1Y4Y" outputId="0d76977e-3255-484b-c9d3-4ee71ca9c990"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)
d = X_train.shape[1]
X_train.shape, y_train.shape, X_test.shape, y_test.shape,dat.shape, y_test[0:10], y_train[0:10]

# + [markdown] id="vTcHrKJVByqM"
# In the following we will look at the number of fish each group caught. 

# + colab={"base_uri": "https://localhost:8080/", "height": 595} id="RVqCbgQZ1Y4e" outputId="b124b6ef-235a-402a-a34d-4cd7be80f22c"
plt.figure(figsize=(14,5))

vals, counts = np.unique(y_train, return_counts=True)
plt.subplot(1,2,1)
plt.stem(vals, counts)
plt.xlabel('Count: number of fish caught')
plt.ylabel('Frequency')
plt.title('Distribution of number of fish caught in training')

plt.subplot(1,2,2)
plt.stem(vals, counts)
plt.xlabel('Count: number of fish caught')
plt.ylabel('Frequency')
plt.xlim(-1,10)
plt.title('Zoomed distribution of number of fish caught in training')
plt.show()

np.max(y_train)

# + [markdown] id="fWdfVAeiPhMP"
# You see that most of the groups didn't catch any fish at all. Most of the groups were not very successful, but there is one group that was very successful and caught 149 fish!

# + [markdown] id="T2LZ1O7SPyZe"
# Lets pick the two test observations 31 and 33, which you will investigate in the following. 

# + colab={"base_uri": "https://localhost:8080/", "height": 88} id="lNnxCICNwElp" outputId="0fb4380a-91d2-4b4f-b581-94a17bb18a08"
print(X_test[31])#"livebait","camper","persons","child
print(X_test[33])#"livebait","camper","persons","child
print(y_test[31])#"number of caught fish
print(y_test[33])#"number of caught fish

# + [markdown] id="MAzV_poQQLqe"
# Group 31 used livebait, had a camper and were 4 persons with one child. They caught 5 fish.  
# Group 33 used livebait, didn't have a camper and were 4 persons with two childern. They caught 0 fish.

# + [markdown] id="QeOx8Gk91Y4l"
# ## Linear regression with constant variance
#
# In the next few cells you will ignore the fact that you are dealing with count data here and just fit a linear regression model with **constant variance** to the data. You will fist do this with sklearn and then with keras. You will use the standart MSE loss and calculate the optimal standart deviation to minimize the NLL. Finally, you predict the test data and compare the performance with the RSME, MAE and the NLL. 

# + [markdown] id="Y2jVnNH71Y4m"
# ### Linear regression with sklearn 
#  
# Let's fist fit the linear regression with sklean on the training data.

# + id="kl72JqqH1Y4o"
# The linear regression using non deep learning methods
# These methods have no problem with convergence 
from sklearn.linear_model import LinearRegression
model_skl = LinearRegression()
res = model_skl.fit(X_train, y_train)

# + [markdown] id="43AlkxLU1Y4w"
# In linear regression, we assuming that the $\sigma$ is constant. To calculate the NLL, we need to estimate this quantity from the training data. The prediction is of course done on the testdata. Note that we calculate the mean  test NLL.

# + colab={"base_uri": "https://localhost:8080/", "height": 132} id="aZrTcLWA1Y4z" outputId="f4c8ad47-f3d4-4440-d7fd-23c841b86ccd"
import pandas as pd
# Calculation of the the optimal sigma 
y_hat_train = model_skl.predict(X_train)
n = len(y_hat_train)
sigma_hat_2 = (n-1.)/(n-2.) * np.var(y_train - y_hat_train.flatten(),ddof=1)
print('Estimated variance ', sigma_hat_2)
print('Estimated standart deviation ', np.sqrt(sigma_hat_2))

y_hat = model_skl.predict(X_test) #Prediction on the testset
RMSE_skl = np.sqrt(np.mean((y_test - y_hat.flatten())**2))
MAE_skl = np.mean(np.abs(y_test- y_hat.flatten())) 

NLL_skl =  0.5*np.log(2 * np.pi * sigma_hat_2) + 0.5*np.mean((y_test - y_hat.flatten())**2)/sigma_hat_2
print('NLL on training:', 0.5*np.log(2 * np.pi * sigma_hat_2) + 0.5*np.mean((y_train - y_hat_train.flatten())**2)/sigma_hat_2)

df1 = pd.DataFrame(
          {'RMSE' : RMSE_skl, 'MAE' : MAE_skl, 'NLL (mean)' : NLL_skl}, index=['Linear Regression (sklearn)']
)
df1

# + [markdown] id="AxwxHcV_1Y4_"
# ### Linear regression with Keras 
#  
# Let's do the same as before with sklearn, but this time you fit a linear regression  model with keras.
# You have 4 inputs (child , persons livebait, camper) and 1 output (count). Note that you'll use the standart MSE loss.

# + id="J8oUGFcK1Y5C"
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input 
from tensorflow.keras.optimizers import Adam

model_lr = Sequential() 
model_lr.add(Dense(1,input_dim=d, activation='linear')) 
model_lr.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.01))

# + id="7bZZMADo1Y5K"
hist_lr = model_lr.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=5000, verbose=0, batch_size=len(y_train))

# + colab={"base_uri": "https://localhost:8080/", "height": 449} id="Eo4VNK9J1Y5R" outputId="a153beab-d98c-4a5d-e947-97c4e2d452bb"
plt.plot(hist_lr.history['loss']) #Note this is the MSE and not the RMSE
plt.plot(hist_lr.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.show()

# + [markdown] id="kzvS6_dEV7K2"
# #### Evaluation of the Performance 
#

# + colab={"base_uri": "https://localhost:8080/", "height": 162} id="Qyzmr9jLKJiK" outputId="d05a2a71-d842-4348-aa46-d8b35b797c6f"
# Calculation of the the optimal sigma 
y_hat_train = model_lr.predict(X_train)
sigma_hat_2 = (n-1.)/(n-2.) * np.var(y_train - y_hat_train.flatten(),ddof=1)
print('Estimated variance ', sigma_hat_2)
print('Estimated standart deviation ', np.sqrt(sigma_hat_2))

y_hat = model_lr.predict(X_test) #Prediction on the testset
RMSE_lr = np.sqrt(np.mean((y_test - y_hat.flatten())**2))
MAE_lr = np.mean(np.abs(y_test - y_hat.flatten())) 

NLL_lr =  0.5*np.log(2 * np.pi * sigma_hat_2) + 0.5*np.mean((y_test - y_hat.flatten())**2)/sigma_hat_2
print('NLL on training:', 0.5*np.log(2 * np.pi * sigma_hat_2) + 0.5*np.mean((y_train - y_hat_train.flatten())**2)/sigma_hat_2)

df2 = pd.DataFrame(
          {'RMSE' : RMSE_lr, 'MAE' : MAE_lr, 'NLL (mean)' : NLL_lr}, index=['Linear Regression (MSE Keras)']
)
pd.concat([df1,df2])

# + [markdown] id="OTrEASJJ1Y5f"
# In the pandas dataframe above you see that the RMSE, MAE and the NLL are same. In the next cell you are comparing the coefficients of the keras and sklearn linear regression models. As you can see you get the same results! 

# + colab={"base_uri": "https://localhost:8080/", "height": 88} id="O_KZ6MI81Y5h" outputId="c03b1a58-7e9e-4e98-b62c-9bd13db24f69"
print('weights using deep learning:          ',model_lr.get_weights()[0][:,0])
print('weights from sklearn:                 ',res.coef_)
print('Intercept (bias) using deep learning: ',model_lr.get_weights()[1][0])
print('Intercept (bias) using sklearn:       ',res.intercept_)

# + [markdown] id="scvDQ-mg1Y5r"
# Let's plot the observed values vs the predicted mean of caught fish on the test dataset. To inicate the CPD you also plot  the 2.5% and 97.5% percentiles of the predicted CPD. You highlight the observations 31 and 33.

# + colab={"base_uri": "https://localhost:8080/", "height": 472} id="FlzxWOQT1Y5u" outputId="f4b85a59-aefa-463b-8aa3-493d731ca4fa"
y_hat_test=model_lr.predict(X_test)
plt.scatter(y_hat_test, y_test,alpha=0.3)
plt.scatter(y_hat_test[33], y_test[33],c="orange",marker='o',edgecolors= "black")
plt.scatter(y_hat_test[31], y_test[31],c="orange",marker='o',edgecolors= "black")
sort_idx=np.argsort(y_hat_test,axis=0)
plt.plot(y_hat_test[sort_idx].flatten(), y_hat_test[sort_idx].flatten()+2*np.sqrt(sigma_hat_2),linestyle='dashed',c="black")
plt.plot(y_hat_test[sort_idx].flatten(), y_hat_test[sort_idx].flatten()-2*np.sqrt(sigma_hat_2),linestyle='dashed',c="black")
plt.plot(y_hat_test, y_hat_test, c="black")
plt.title('Comparison on the testset')
plt.xlabel('predicted average of caught fish')
plt.ylabel('observed number of caught fish')
plt.show()
#plt.savefig("camper_lr.pdf")
#from google.colab import files
#files.download('camper_lr.pdf') 

# + colab={"base_uri": "https://localhost:8080/", "height": 88} id="shnjM81c1pEf" outputId="71e8995e-ce0f-44ab-f359-978f24105484"
# Let's check the mean of the predicted CPDs for the obeservations nr 31 and 33
print(y_hat_test[31])
print(y_hat_test[33])
# Remember the observed nr of caught fish for the obeservations nr 31 and 33
print(y_test[31])
print(y_test[33])

# + [markdown] id="Y9R790TuTw-9"
# Lets check the predicted outcome distribution for the observations 31 and 33.

# + colab={"base_uri": "https://localhost:8080/", "height": 564} id="bUzQHcEh2fy2" outputId="a312e8ed-605a-4342-e0a1-f84287aed95a"
dist = tfd.Normal(loc=y_hat_test,scale=np.sqrt(sigma_hat_2,dtype="float32"))
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.plot(np.arange(-40,40,0.1),dist.prob(np.arange(-40,40,0.1))[31])
plt.vlines(y_hat_test[31], ymin=0, ymax=dist.prob(y_hat_test)[31],linestyle='dashed')
plt.vlines(np.expand_dims(y_test,axis=1)[31], ymin=0, ymax=dist.prob(np.expand_dims(y_test,axis=1))[31],linestyle='dotted',color="purple",linewidth=2)
plt.xlabel('Number of Events')
plt.ylabel('Probability density')
plt.title('Test observation 31, observed fish=5')

plt.subplot(1,2,2)
plt.plot(np.arange(-40,40,0.1),dist.prob(np.arange(-40,40,0.1))[33])
plt.vlines(y_hat_test[33], ymin=0, ymax=dist.prob(y_hat_test)[33],linestyle='dashed')
plt.vlines(np.expand_dims(y_test,axis=1)[33], ymin=0, ymax=dist.prob(np.expand_dims(y_test,axis=1))[33],linestyle='dotted',color="purple",linewidth=2)
plt.xlabel('Number of Events')
plt.ylabel('Probability density')
plt.title('Test observation 33, observed fish=0')
plt.show()
#plt.savefig("5.gauss.dist.pdf")
#from google.colab import files
#files.download('5.gauss.dist.pdf')

# + [markdown] id="UQQlDYPt1Y5_"
# You can see that the liklihood of the observed values are quite high under the predicted CPDs (dotted line). However, note that the linear model predicts also negative values, which is obviously wrong. 

# + [markdown] id="RC5yQ6Na1Y6u"
# ## Poisson Regression 
#
# Now you use  the TFP framework and the Poission distribution to model the output of the network as a Poissonian CPD. You will not use any hidden layers in between and the loss will be the NLL. After the fitting, you predict the test data and compare the performance with the linear regression model.
# $$
#     Y \thicksim \tt{Pois}(exp(w^{T} \cdot x + b))
# $$
#
# The output of your network is the parameter which gives $\lambda = exp(w^{T} \cdot x + b)$
#
# and we will using optimizer to estimate $\lambda$ by 
#
# `y_true` = label data of counts
#
# `y_pred` = Possion($\lambda$), $\lambda$ is the output of your network
#
#
# #### Listing 5.6: Simple Poisson regression for the number of fish caught 
#

# + colab={"base_uri": "https://localhost:8080/", "height": 266} id="c1D6mfOl1Y6w" outputId="619fe60f-6194-4067-a95b-a9267ce0d47f"
inputs = Input(shape=(X_train.shape[1],))  
rate = Dense(1, 
         activation=tf.exp)(inputs) #A
p_y = tfp.layers.DistributionLambda(tfd.Poisson)(rate) #B 

model_p = Model(inputs=inputs, outputs=p_y) #C


def NLL(y_true, y_hat): #D
  return -y_hat.log_prob(y_true)

# it should be tf.tensor
# print(type(y_true), type(y_hat))

model_p.compile(Adam(learning_rate=0.01), loss=NLL)
model_p.summary()

#A Definition of a single layer with one output
#B We use exponential of the output to model the rate
#C Glueing the NN and the output layer together. Note that output p_y is a tf.distribution
#D The second argument is the output of the model and thus a tfp-distribution. It's as simple as calling log_prob to calculate the log-probability of the observation that is needed to calculate the NLL.

# + id="kheyBbO_1Y6-"
hist_p = model_p.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=2000, verbose=0)

# + colab={"base_uri": "https://localhost:8080/", "height": 449} id="kSO3GedG1Y7I" outputId="8437e0b6-e691-48fc-a0eb-4d4b83fa1601"
plt.plot(hist_p.history['loss'])
plt.plot(hist_p.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.show()

# + [markdown] id="3koDxrd5V_KJ"
# #### Evaluation of the Performance 
#

# + colab={"base_uri": "https://localhost:8080/", "height": 177} id="oVjw2n-C1Y7P" outputId="ffdb0d96-7922-4864-f605-fff43a2bb50c"
model = Model(inputs=inputs, outputs=p_y.mean()) 
y_hat_test = model.predict(X_test).flatten()


rmse=np.sqrt(np.mean((y_test - y_hat_test)**2))
mae=np.mean(np.abs(y_test - y_hat_test)) 

NLL = model_p.evaluate(X_test, y_test) #returns the NLL 

df3 = pd.DataFrame(
         { 'RMSE' : rmse, 'MAE' : mae, 'NLL (mean)' : NLL}, index=['Poisson Regression (TFP)']
)
pd.concat([df1,df2,df3])

# + [markdown] id="vWa4RKhaVVAX"
# In the pandas dataframe above you see that the RMSE, MAE and the NLL of the diferent models. You see that the Poisson regression outperform the linear regression because of the lower NLL.

# + [markdown] id="mmdYk1eLWFaN"
# Let's plot the observed values vs the predicted mean of caught fish on the test dataset. To inicate the CPD you also plot  the 2.5% and 97.5% percentiles of the predicted CPD. You highlight the observations 31 and 33.

# + colab={"base_uri": "https://localhost:8080/", "height": 564} id="ddqrjcSb1Y7T" outputId="aa65a562-7d4f-49eb-e193-590e398b2182"
from scipy.stats import poisson
lower=poisson.ppf(0.025, y_hat_test)
upper=poisson.ppf(0.975, y_hat_test)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(y_hat_test, y_test, alpha=0.3)
plt.scatter(y_hat_test[33], y_test[33],c="orange",marker='o',edgecolors= "black")
plt.scatter(y_hat_test[31], y_test[31],c="orange",marker='o',edgecolors= "black")
plt.title('Comparison on the testset')
plt.xlabel('predicted average of caught fish')
plt.ylabel('observed number of caught fish')
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), lower[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), upper[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test, y_hat_test, c="black")


plt.subplot(1,2,2)
plt.scatter(y_hat_test, y_test, alpha=0.3)
plt.scatter(y_hat_test[33], y_test[33],c="orange",marker='o',edgecolors= "black")
plt.scatter(y_hat_test[31], y_test[31],c="orange",marker='o',edgecolors= "black")
plt.title('Zoomed comparison on the testset')
plt.xlabel('predicted average of caught fish')
plt.ylabel('observed number of caught fish')
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), lower[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), upper[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test, y_hat_test, c="black")
plt.xlim([-0.5,6])
plt.ylim([-0.5,6])
#plt.savefig("camper_pois.pdf")
#from google.colab import files
#files.download('camper_pois.pdf')
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 88} id="T_5JwyCQWdWm" outputId="9cf8eeaa-c6e2-44c3-894f-8844f3f32f18"
# Let's check the mean of the predicted CPDs for the obeservations nr 31 and 33
print(y_hat_test[31])
print(y_hat_test[33])
# Remember the observed nr of caught fish for the obeservations nr 31 and 33
print(y_test[31])
print(y_test[33])

# + [markdown] id="gHT2pj1kWp6c"
# Lets check the predicted outcome distribution for the observations 31 and 33.

# + colab={"base_uri": "https://localhost:8080/", "height": 655} id="3KdtKSlq6CDI" outputId="c6ea642c-1110-4a10-afa8-6eb3cd1f4800"
probs=model_p(X_test).prob(np.arange(0,20,1)).numpy()
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.stem(np.arange(0,20,1),probs[31,:])
plt.xticks(np.arange(0,20,1))
plt.vlines(np.expand_dims(y_test,axis=1)[31], ymin=0, ymax=probs[31,np.int(y_test[31])],linestyle='dotted',color="purple",linewidth=4)

plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title('Test observation 31, observed fish=5')

plt.subplot(1,2,2)
plt.stem(np.arange(0,20,1),probs[33,:])
plt.xticks(np.arange(0,20,1))
plt.vlines(np.expand_dims(y_test,axis=1)[33], ymin=0, ymax=probs[33,np.int(y_test[33])],linestyle='dotted',color="purple",linewidth=4)
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title('Test observation 33, observed fish=0')
plt.show()
#plt.savefig("pois_dist_31_33.pdf")
#from google.colab import files
#files.download('pois_dist_31_33.pdf')

# + [markdown] id="RP8y4gdz1Y7a"
# You can see that the liklihood of the observed values are quite high under the predicted CPDs (dotted line). Note that the Poisson CPD does only predict non-negative integer values which is a quite nice property for count data.
#
#
#  

# + [markdown] id="qQY2rCxQuHfG"
# ### Return to the book 
# <img src="https://raw.githubusercontent.com/tensorchiefs/dl_book/master/imgs/Page_turn_icon_A.png" width="120" align="left" />  
# Return to the book to learn about zero inflated Poisson Regression.

# + [markdown] id="PJWH_iYP1Y7a"
# ## Zero inflated Poisson Regression 
#
# You saw that there are a lot of unlucky groups that did not catch any fish at all. You will now define a model with two outputs, one for the poisson mean and one for the probability that zero fish were caught. This is the so called zero-inflated Poisson distribution. You use the TFP framework to create a mixture two processes: a Poission process and a zero generating process. You will not use any hidden layers in between and the loss will be the NLL. After the fitting, you predict the test data and compare the performance with the other models.
#
#

# + [markdown] id="dWyekfiA9lxG"
# The ZIP distribution needs two parameters:
# * rate: which defines the rate $\lambda$ of a Poisson process
# * s: the probability to pick Poisson process (accordingly the zero-generating process is picked with probability 1-s)
#
# #### Listing 5.7: Custom Distribution for a zero inflated Poisson distribution 
#

# + id="s9tYvDbj1Y7b"
def zero_inf(out): 
    rate = tf.squeeze(tf.math.exp(out[:,0:1])) #A 
    s = tf.math.sigmoid(out[:,1:2]) #B  
    probs = tf.concat([1-s, s], axis=1) #C 
    return tfd.Mixture(
          cat=tfd.Categorical(probs=probs),#D
          components=[
          tfd.Deterministic(loc=tf.zeros_like(rate)), #E
          tfd.Poisson(rate=rate), #F 
        ])

#A The first component codes for the rate. We used exponential to guaranty values > 0. We also used the squeeze function to flatten the tensor.
#B The second component codes for zero inflation; using the sigmoid squeezes the value between 0 and 1.
#C The two probabilities for 0’s or Poissonian distribution 
#D tfd.Categorical allows creating a mixture of two components. 
#E Zero as a deterministic value 
#F Value drawn from a Poissonian distribution


# + [markdown] id="2dxIHn1UkwHp"
# In the next cell you can check if the ZIP distribution is working. As you can see in the code above, the zero_inf distribution takes two values as input. The first value controls the rate of the Poisson distribution and the second value controls the probability to pick the Poisson process. Both values can be negative or positive. To guarantee that the rate is a positive number, we transform the first argument with the exp() function.To guarantee that the probability s is a number between zero and one, we transform the second argument with the sigmoid() function.  
#
# If the first argument is 1 then the rate of the Poisson process is exp(1) ~ 2.7. If the second argument is 10 then the probability to pick the Poisson process is sigmoid(10) ~ 0.9999. Accordingly, if the input to the zero_inf() distribution is 1 and 10, we would expect that we almost always take the Poisson process which has a rate parameter of ~ 2.7.  
#
# If the input to the zero_inf() distribution is 1 and -10, we would expect that we almost always pick the zero-generating process. 
#   
# In the following cell you can check that the zero_inf function works as expected. It is also possible to sample from the distribution or calculate the mean.

# + colab={"base_uri": "https://localhost:8080/", "height": 159} id="0cObilkk1Y7g" outputId="e410ada6-33dd-43f1-b186-6521a6c9bf58"
## testinging the distribution, we evalute some data 

print("rate of the poissonian :", tf.exp(1.0).numpy())
print("probability to pick the poisson process :" ,tf.math.sigmoid(10.0).numpy())
print("probability to pick the poisson process :" ,tf.math.sigmoid(-10.0).numpy())


t = np.ones((2,2), dtype=np.float32)
t[0,0] = 1
t[0,1] = 10#almost always take pois 
t[1,0] = 1
t[1,1] = -10# almost always take zero
#t = tf.cast(t, dtype="float32")
print('Input Tensor : ')
print(t)
print('Output Mean  : ',zero_inf(t).mean().numpy())
print('Output Sample  : ',zero_inf(t).sample().numpy())

# + [markdown] id="iNVLvmqF4jgL"
# Here you define the network and use the zero_inf distribution.
#
# #### Listing 5.8: NN in front of a zero inflated Poisson distribution 
#

# + colab={"base_uri": "https://localhost:8080/", "height": 266} id="IhkJUDNW1Y7m" outputId="5e6283d5-77d1-424b-9b53-262dbf03f077"
## Definition of the custom parametrized distribution
inputs = tf.keras.layers.Input(shape=(X_train.shape[1],))  
out = Dense(2)(inputs) #A
p_y_zi = tfp.layers.DistributionLambda(zero_inf)(out)
model_zi = Model(inputs=inputs, outputs=p_y_zi)

#A A dense layer is used without activation. The transformation is done inside the zero_inf function
model_zi.summary()

# + [markdown] id="xVLRbGNg1Y7q"
# ### Training of the model (by hand)with GradientTape [optional]
#
# The following code trains the NN using a evaluation loop by hand. This can help to find instabilities, See also: https://www.tensorflow.org/beta/guide/keras/training_and_evaluation#part_ii_writing_your_own_training_evaluation_loops_from_scratch
#

# + colab={"base_uri": "https://localhost:8080/", "height": 195} id="FQGr8xl-1Y7r" outputId="4fcc7875-8c23-49d7-9740-a6951e22508d"
optimizer=tf.optimizers.Adam(learning_rate=0.05)
steps=10
loss_values = np.zeros((steps))
for e in range(steps):
    with tf.GradientTape() as tape:
        y_hat = model_zi(X_train)
        loss_value = -tf.reduce_mean(y_hat.log_prob(y_train))
        loss_values[e] = loss_value
        grads = tape.gradient(loss_value, model_zi.trainable_weights)
        weights =  model_zi.trainable_weights       
        optimizer.apply_gradients(zip(grads,weights))
        print(loss_value)


# + [markdown] id="LjpdKTYq1Y7t"
# ### Training using keras

# + id="QbmF_D-p1Y7u"
def NLL(y_true, y_hat):
    return -y_hat.log_prob(tf.reshape(y_true,(-1,)))

model_zi.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=NLL)
hist_zi = model_zi.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=2000, verbose=0)

# + colab={"base_uri": "https://localhost:8080/", "height": 542} id="BTrC-2dzW8_-" outputId="f097f15f-d27f-483d-a2a0-c093f383bb9e"
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(hist_zi.history['loss'])
plt.plot(hist_zi.history['val_loss'])
plt.legend(['ZI loss','ZI val_loss'])
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.subplot(1,2,2)
plt.plot(hist_p.history['loss'],linestyle='-.')
plt.plot(hist_p.history['val_loss'])
plt.plot(hist_zi.history['loss'])
plt.plot(hist_zi.history['val_loss'])
plt.legend(['Poisson loss','Poisson val_loss','ZI loss','ZI val_loss'])
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.show()

# + [markdown] id="YLquO2M5Y_gX"
# #### Evaluation of the Performance 
#

# + colab={"base_uri": "https://localhost:8080/", "height": 207} id="dwM1es891Y74" outputId="2716dab9-dc09-40da-a51f-3499355da3ec"
model = Model(inputs=inputs, outputs=p_y_zi.mean()) 
y_hat_test = model.predict(X_test).flatten()


mse=np.sqrt(np.mean((y_test - y_hat_test)**2))
mae=np.mean(np.abs(y_test - y_hat_test)) 

NLL = model_zi.evaluate(X_test, y_test) #returns the NLL 


df4 = pd.DataFrame(
         { 'RMSE' : mse, 'MAE' : mae, 'NLL (mean)' : NLL}, index=['ZIP (TFP)']
)
pd.concat([df1,df2,df3,df4])

# + [markdown] id="LzEnO51Pm2BI"
# In the pandas dataframe above you see that the RMSE, MAE and the NLL of the diferent models. You see that the ZIP regression outperforms the Poisson and the Linear regression models  because of the lower NLL.

# + [markdown] id="vgdfkpZynPwm"
# Let's plot the observed values vs the predicted mean of caught fish on the test dataset. To inicate the CPD you also plot  the 2.5% and 97.5% percentiles of the predicted CPD. You highlight the observations 31 and 33.

# + colab={"base_uri": "https://localhost:8080/", "height": 564} id="3Sc175du1Y79" outputId="fc18e959-b944-45c5-d6cb-c742878ffa7f"
samples=model_zi(X_test).sample(5000).numpy()
lower=np.quantile(samples,0.025,axis=0)
upper=np.quantile(samples,0.975,axis=0)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(y_hat_test, y_test, alpha=0.3)
plt.scatter(y_hat_test[33], y_test[33],c="orange",marker='o',edgecolors= "black")
plt.scatter(y_hat_test[31], y_test[31],c="orange",marker='o',edgecolors= "black")

plt.title('Comparison on the testset')
plt.xlabel('predicted average of caught fish')
plt.ylabel('observed number of caught fish')
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), lower[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), upper[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test, y_hat_test, c="black")


plt.subplot(1,2,2)
plt.scatter(y_hat_test, y_test, alpha=0.3)
plt.scatter(y_hat_test[33], y_test[33],c="orange",marker='o',edgecolors= "black")
plt.scatter(y_hat_test[31], y_test[31],c="orange",marker='o',edgecolors= "black")

plt.title('Zoomed comparison on the testset')
plt.xlabel('predicted average of caught fish')
plt.ylabel('observed number of caught fish')
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), lower[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test[np.argsort(y_hat_test,axis=0)].flatten(), upper[np.argsort(y_hat_test,axis=0)],linestyle='dashed',c="black")
plt.plot(y_hat_test, y_hat_test, c="black")
plt.xlim([-0.5,6])
plt.ylim([-0.5,6])


#plt.savefig("camper_zipois.pdf")
#from google.colab import files
#files.download('camper_zipois.pdf')
plt.show()

# + [markdown] id="GMCbYE29nroP"
# Compared to the Poisson model it is striking that the 2.5% percentile is zero over the whole range. This is due the zero-inflated process modeling a higher amount of zeros compared to the Poisson process.

# + colab={"base_uri": "https://localhost:8080/", "height": 88} id="0PwBywhEnaEc" outputId="74ba81cb-fe41-407b-cc60-b9ae6f570504"
# Let's check the mean of the predicted CPDs for the obeservations nr 31 and 33
print(y_hat_test[31])
print(y_hat_test[33])
# Remember the observed nr of caught fish for the obeservations nr 31 and 33
print(y_test[31])
print(y_test[33])

# + [markdown] id="L-gOH43vnj1G"
# Lets check the predicted outcome distribution for the observations 31 and 33.

# + colab={"base_uri": "https://localhost:8080/", "height": 655} id="2M1VM1bnwE5j" outputId="2cc589f7-361f-4941-edb4-cf79a17ce2ea"
probs=model_zi(X_test).prob(np.arange(0,20,1).reshape(20,1)).numpy()
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.stem(np.arange(0,20,1),probs[:,31])
plt.vlines(np.expand_dims(y_test,axis=1)[31], ymin=0, ymax=probs[np.int(y_test[31]),31],linestyle='dotted',color="purple",linewidth=4)
plt.xticks(np.arange(0,20,1))
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title('Test observation 31, observed fish=5')

plt.subplot(1,2,2)
plt.stem(np.arange(0,20,1),probs[:,33])
plt.vlines(np.expand_dims(y_test,axis=1)[33], ymin=0, ymax=probs[np.int(y_test[33]),33],linestyle='dotted',color="purple",linewidth=4)
plt.xticks(np.arange(0,20,1))
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title('Test observation 33, observed fish=0')
plt.show()
#plt.savefig("zip_dist_31_33.pdf")
#from google.colab import files
#files.download('zip_dist_31_33.pdf')

# + [markdown] id="Cb_VU8L3oZe1"
# You can see that the  predicted CPDs has a large peak at zero. This is due the zero-inflated process modeling a higher amount of zeros compared to the Poisson process.  
# You can see that the liklihood of the observed values are quite high under the predicted CPDs (dotted line). Note that the ZIP CPD does only predict non-negative integer values which is a quite nice property for count data.
#

# + [markdown] id="vkWMUK151Y8A"
# Let's see what happens if you make more fisherman unlucky and remove randomly remove some catched. You can uncomment the lines in cell 6.

# + id="qwxUFHulpE_l"

