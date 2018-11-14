import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
from scipy.stats import lognorm


def funk1(e1, v1):
    mi1 = np.log(e1**2/np.sqrt(v1 + e1**2))
    sigma21 = np.log(1+v1/e1**2)
    return mi1, sigma21

m, s2 = funk1(3, 5)
s = np.sqrt(s2)

samples = np.random.lognormal(mean=m, sigma=s, size=10000)
x = np.sort(samples)
y = np.arange(len(x))/float(len(x))

plt.scatter(np.where(samples), samples, color='g', label='sample')
plt.legend(loc='upper right')
plt.title('Sample')
plt.xlabel('Indices')
plt.ylabel('X Data')
plt.show()


plt.hist(samples, bins=100, density=True, alpha=0.6, color='g', label='Sample histogram')
pdf = scipy.stats.lognorm.pdf(x, s, scale=np.exp(m))
plt.plot(x, pdf, color='r', alpha=0.6, label='Exact PDF')
plt.title('PDF')
plt.legend(loc='upper right')
plt.xlabel('X Data')
plt.ylabel('PDF')
plt.show()

plt.plot(x, lognorm.cdf(x, s, scale=np.exp(m)), 'r-', label='Exact CDF')  #exact cdf
plt.plot(x, y, 'g', label='Empiric CDF')
plt.title('CDF')
plt.legend(loc='upper right')
plt.xlabel('X data')
plt.ylabel('CDF')
plt.show()
