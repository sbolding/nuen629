import matplotlib.pyplot as plt
from sys import argv
import re
import numpy as np
from numpy import array, genfromtxt
from math import sqrt
from sympy import *
from math import sqrt

x, Al, Ar, z = symbols('X Al Ar z')
Bl, Br, Q, D = symbols('Bl Br Q D')
alpha = symbols('alpha')


K1 = -1.*(Al*Q*x**2/(8.*D) + Bl*Q*x/2. - (Al/Ar)*(Ar*Q*x**2/(8.*D) + Br*Q*x/2.)) \
        / (Al*x/2. + Bl*D + (Al/Ar)*(Ar*x/2. + Br*D))

K2 = (Al*Q*x**2/(8.*D) + Bl*Q*x/2. + ((Al*x/2. + Bl*D)/(Ar*x/2.+Br*D))*(Ar*Q*x**2/(8.*D) + Br*Q*x/2.)) \
        / (Al + ((Al*x/2. + Bl*D)/(Ar*x/2.+Br*D))*Ar)


phi = Q*z**2/(2*D) + K1*z + K2
Ddphidx   = Q*z + K1*D


#marshak vacuum
subs = dict()
A = 0.25
B = 0.5
subs['Al']=subs['Ar']=A
subs['Bl']=subs['Br']=B
subs['z']=0.5*x

print "Marshak"
print "K1: ", K1, "K2: ", K2


valr = A*phi + B*Ddphidx
vall = A*phi - B*Ddphidx
print "Phi at right edge: ", phi.subs(subs)
print "Value at right edge is: ", valr.subs(subs)

subs['z'] = -0.5*x
print "\nPhi at left edge: ", phi.subs(subs)
print "Value at left edge is: ", vall.subs(subs)

subs = dict()
A_l = 1
B_l = 0.0
subs['Al']=A_l
subs['Bl']=B_l
A_r = (1.-alpha)/(2.*(1.+alpha))
B_r = 1.
subs['Ar']=A_r
subs['Br']=B_r
subs['Q']=1.0
subs['D']=1.0
subs['X']=1.0
subs['z']=0.5*x

print "ALBEDO"
print "K1: ", K1, "K2: ", K2
print "1 everywhere: K1", K1.subs(subs), "K2: ", K2.subs(subs)

phi = Q*z**2/(2.*D) + Q*z*x*(0.5/D-(1+A_r*x/(2*D))/(A_r*x + D)) + Q*x**2/2.*(0.25/D -
        (1+A_r*(x/(2*D)))/(A_r*x+D))
Ddphidx = D*diff(phi,z)

valr = A*phi + B*Ddphidx
vall = A*phi - B*Ddphidx
subs['z']=0.5*1
subs['alpha'] = 0.5
print "Phi at right edge: ", phi.subs(subs)
print "Value at right edge is: ", valr.subs(subs)

subs['z'] = -0.5
subs['X'] = 1.0
print "\nPhi at left edge: ", phi.subs(subs)
print "Value at left edge is: ", vall.subs(subs)

Q = 1.0
D = 0.1
alpha = 0.5
X = 1.0
A_r = (1.-alpha)/(2.*(1.+alpha))
alpha1 = 1.0
A_r1    = (1.-alpha1)/(2.*(1.+alpha1))


phi1 = lambda x: -1.*Q*(x*x/(2.*D) - X**2/(8.*D) - X)
phi2 = lambda x: -1.*Q*(x*x/(2.*D) - X**2/(8.*D) - sqrt(3)/2*X)
phi3 = lambda x: -1.*Q/(2.*D)*(x**2 - X**2/4)
phi4 = lambda x:  -1.*Q*x**2/(2.*D) - Q*x*X*(0.5/D-(1+A_r*X/(2*D))/(A_r*X + D)) - Q*X**2/2.*(0.25/D -
        (1+A_r*(x/(2*D)))/(A_r*x+D))
phi5 = lambda x:  Q/(2.*D)*(-1.*x**2 + x*X + 3.*X*X/4.)


phi6 = lambda x:  -1.*Q*x**2/(2.*D) - Q*x*X*(0.5/D-(1+A_r1*X/(2*D))/(A_r1*X + D)) - Q*X**2/2.*(0.25/D -
        (1+A_r1*(x/(2*D)))/(A_r1*x+D))

x = np.linspace(-X/2,X/2,num=100)

plt.figure()
# plot temperatures
plt.plot(x,phi1(x),"-b",label="Marshak")
plt.plot(x,phi2(x),"-r",label="Mark")
plt.plot(x,phi3(x),"-g",label="Vacuum")
plt.plot(x,phi4(x),"--b",label=r"Albedo, $\alpha=0.5$")
plt.plot(x,phi5(x),"--r",label="Reflecting")
plt.plot(x,phi6(x),"or",label=r"Albedo, $\alpha=1$")

# annotations
plt.xlabel('$E$ eV')
plt.ylabel(r'$P(\mu_0,0\leq E \leq \mathrm{1 eV})$')
plt.grid()
plt.legend(loc='best')
plt.show()
#plt.savefig("scat_kernel_E_05"+".pdf",bbox_inches='tight')

plt.figure()
# plot temperatures
plt.plot(x,phi4(x),"--b",label=r"Albedo, $\alpha=0.5$")
#plt.plot(x,phi5(x),"--r",label="Reflecting")
#plt.plot(x,phi6(x),"or",label=r"Albedo, $\alpha=1$")

# annotations
plt.xlabel('$E$ eV')
plt.ylabel(r'$P(\mu_0,0\leq E \leq \mathrm{1 eV})$')
plt.grid()
plt.legend(loc='best')
plt.show()
plt.savefig("scat_kernel_E_05"+".pdf",bbox_inches='tight')










