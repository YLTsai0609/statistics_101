import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

class LevyPDF(st.rv_continuous):
    def _pdf(self,r):
        r0 = 100
        k = 1500
        beta = 1.6
        return (r + r0)**(-beta)*np.exp(-r/k)

class gaussian_gen(st.rv_continuous):
    "Gaussian distribution"
    # def __init__(self, x):
    #     super(gaussian_gen, self).__init__()
    #     self.x = x
    def _pdf(self, x, sigma):
        return np.exp(-x**2 / 2.) / np.sqrt(2.0 * sigma * np.pi)

gaussian = gaussian_gen(name='gaussian')
gaussian.pdf(x=1,sigma=0.02)
gaussian_2 = gaussian_gen(name='gaussian2')
gaussian_2.pdf(x=1,sigma=400)
nmin = 0
nmax = 50
my_cv = LevyPDF(a=nmin, b=nmax, name='LevyPDF')

N = 100
X1= gaussian.rvs(size=N, random_state=1)
X2= gaussian_2.rvs(size=N, random_state=2)

fig, axe = plt.subplots(1, 2)
n, bins, patches = axe[0].hist(X1, 50, normed=1, facecolor='green', alpha=0.75)
n, bins, patches = axe[1].hist(X2, 50, normed=1, facecolor='blue', alpha=0.75)
plt.show()
