import matplotlib.pyplot as plt
from sys import argv
import re
import numpy as np
from numpy import array, genfromtxt
from math import sqrt


E_prime = .05
E_max = 1.e-06
mu_max = sqrt(E_max/E_prime)
mu = np.linspace(0,mu_max,num=1000)
E_range = [E_prime*i*i*10.**6 for i in mu]
f = [2.*i for i in mu]

# plot temperatures
plt.plot(mu,f,"-b")

# annotations
plt.xlabel('$\mu_0$')
plt.ylabel(r'$P(\mu_0,0\leq E \leq \mathrm{1 eV})$')
plt.grid()
plt.savefig("scat_kernel_05"+".pdf",bbox_inches='tight')

plt.figure()
# plot temperatures
plt.plot(E_range,f,"-b")

# annotations
plt.xlabel('$E$ eV')
plt.ylabel(r'$P(\mu_0,0\leq E \leq \mathrm{1 eV})$')
plt.grid()
plt.savefig("scat_kernel_E_05"+".pdf",bbox_inches='tight')

print sqrt(1.e-06/0.05), sqrt(1.e-06/10.), sqrt(1.e-06/0.05)/sqrt(1.e-06/10.)



