import matplotlib.pyplot as plt
from sys import argv
import re
import numpy as np
from numpy import array, genfromtxt
from math import sqrt


E = 1.E-6
E_prime = np.linspace(0.05,10,num=1000)
mu = [sqrt(E/i) for i in E_prime]
print mu
f = [2.*i for i in mu]

# plot temperatures
plt.plot(mu,f,"-b")

# annotations
plt.xlabel('$\mu_0$')
plt.ylabel('$P(\mu_0)$')
plt.grid()
plt.savefig("scat_kernel.pdf",bbox_inches='tight')

mu = [sqrt(E/i) for i in E_prime]
print mu
f = [2.*i for i in mu]

# plot temperatures
plt.plot(E_prime,f,"-b")

# annotations
plt.xlabel('$E\'$ MeV')
plt.ylabel('$P(\mu_0)$')
plt.grid()
plt.savefig("scat_kernel_E.pdf",bbox_inches='tight')


