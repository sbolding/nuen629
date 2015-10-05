import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import math
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
from scipy import interpolate
from scipy.integrate import quadrature
from copy import deepcopy

#create the fission spectrum 
chi = lambda E:  0.4865*np.sinh(np.sqrt(2*E))*np.exp(-E)

#read in cross sections
sigma_t_235 = np.genfromtxt('u235_total.csv', delimiter=",",skip_header=1)
sigma_s_235 = np.genfromtxt('u235_elastic.csv', delimiter=",", skip_header=1)
sigma_t_12  = np.genfromtxt('carbon_total.csv', delimiter=",", skip_header=1)
sigma_s_12  = np.genfromtxt('carbon_elastic.csv', delimiter=",", skip_header=1)
#read in 238-U data
#open total cross-section
sigma_t_238 = np.genfromtxt('u238_total.csv', delimiter=",",skip_header=1)
#open total cross-section
sigma_s_238 = np.genfromtxt('u238_elastic.csv', delimiter=",",skip_header=1)

#Convert energies to MeV and apply a fixup
def fix_energies(cx_2d):
    for i in xrange(len(cx_2d)):
        cx_2d[i,0] *= 1.E-6
        if cx_2d[i,0] == cx_2d[i-1,0]:
           cx_2d[i,0] *= 1.0000001

#apply to all energies
fix_energies(sigma_t_235)
fix_energies(sigma_s_235)
fix_energies(sigma_t_238)
fix_energies(sigma_s_238)
fix_energies(sigma_t_12)
fix_energies(sigma_s_12)

#make interpolation functions
sig_t_235_interp = interpolate.interp1d(sigma_t_235[:,0],
        sigma_t_235[:,1],bounds_error=False, fill_value=sigma_t_235[-1,1])
sig_s_235_interp = interpolate.interp1d(sigma_s_235[:,0],
        sigma_s_235[:,1],bounds_error=False, fill_value=sigma_s_235[-1,1])
sig_t_238_interp = interpolate.interp1d(sigma_t_238[:,0],
        sigma_t_238[:,1],bounds_error=False, fill_value=sigma_t_238[-1,1])
sig_s_238_interp = interpolate.interp1d(sigma_s_238[:,0],
        sigma_s_238[:,1],bounds_error=False, fill_value=sigma_s_238[-1,1])
sig_t_12_interp = interpolate.interp1d(sigma_t_12[:,0],
        sigma_t_12[:,1],bounds_error=False, fill_value=sigma_t_12[-1,1])
sig_s_12_interp = interpolate.interp1d(sigma_s_12[:,0],
        sigma_s_12[:,1],bounds_error=False, fill_value=sigma_s_12[-1,1])

energies = np.union1d(sigma_t_235[:,0], sigma_t_238[:,0])
energies = np.union1d(energies,sigma_t_12[:,0])


#let's make some plots
fig = plt.figure()
plt.loglog(energies, sig_t_238_interp(energies),label=r"$\sigma^{238}_t$")
plt.loglog(energies, sig_s_238_interp(energies),label=r"$\sigma^{238}_s")
plt.loglog(energies, sig_t_12_interp(energies), label="$\sigma^{12}_\mathrm{t}$")
plt.loglog(energies, sig_s_12_interp(energies), label="$\sigma^{12}_\mathrm{s}$")
plt.legend(loc=3) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel("$\sigma$ (barns)")
plt.xlabel("E (MeV)")
plt.title("Elemental Carbon")
plt.savefig("carb_cx.pdf",bbox_inches='tight')

#atom ratios, keys are atomic mass numbers
gam = dict()
gam[235] = 0.0072
gam[238] = 0.9928
gam[12.0107]  = 150.
isotopes = gam.keys()

#energy change factors
exc_func = lambda A: (A+1.)**2/(A*A + 1.)

#Put cross sections in a dict
sig_t = dict()
sig_s = dict()
sig_t[12.0107] = sig_t_12_interp
sig_t[235] = sig_t_235_interp
sig_t[238] = sig_t_238_interp
sig_s[12.0107] = sig_s_12_interp
sig_s[235] = sig_s_235_interp
sig_s[238] = sig_s_238_interp

#Initialize phi to 0
phi_prev = interpolate.interp1d(energies,np.zeros(len(energies)),fill_value=0,bounds_error=False)
phi1 = None

converged = 0
tolerance = 1.0e-6
iteration = 0
max_iterations = 1000


#Function for evaluating a new phi
phi_new = lambda E: E*0.0
while not(converged):

    phi_prev = interpolate.interp1d(energies,phi_new(energies),fill_value=0,bounds_error=False)

    #make some lambdas to simplify things
    scat_src_k = lambda E: sum([gam[i]*sig_s[i](exc_func(i)*E)*phi_prev(exc_func(i)*E) for i in isotopes])
    sig_t_k = lambda E: sum([gam[i]*sig_t[i](E) for i in isotopes])
    phi_new = lambda E: (chi(E) + scat_src_k(E))/sig_t_k(E)

    rel_err = np.linalg.norm(phi_prev(energies) - phi_new(energies))/ \
                 np.linalg.norm(phi_new(energies))
    converged = rel_err < tolerance or (iteration >= max_iterations)
    iteration += 1

    #if first iteration save it for plotting
    if iteration == 2:
        phi1 = deepcopy(phi_prev)

    print "Completed iteration:", iteration, "Relative change:", rel_err
    

#plot the first iteration and last iteration, normalized to have an integral of 1
plt.figure()
plt.loglog(energies,phi_new(energies)/np.sum(phi_new(energies)),label=r"$\phi(E)$")
plt.loglog(energies,phi1(energies)/np.sum(phi_new(energies)),label=r"$\phi^{(1)}(E)$")

#plt.loglog(energies,phi(energies)/np.linalg.norm(phi(energies)),label="U metal")
plt.xlabel("E (MeV)")
plt.ylabel("$\phi(E)/\|phi(E)\|_2$ (MeV$^{-1}$)")
plt.legend(loc='best')
plt.savefig("../uc_spect.pdf",bbox_inches='tight')


#Collapse the cross sections
cx_t = []
cx_s = []
int_phi_sig_t = 0.0
int_phi_sig_s = 0.0
int_phi = 0.0
bounds = [1.E-06, 0.1, 19.999999999]
count = 0

for Ei in xrange(len(energies)-1):

    E = (energies[Ei]+energies[Ei+1])/2.
    dE = energies[Ei+1] - energies[Ei]
    
    #get cross sections at this energy
    sig_t_tot = sum([gam[i]*sig_t[i](E) for i in isotopes])
    sig_s_tot = sum([gam[i]*sig_s[i](E) for i in isotopes])
    phi_i = phi_new(E)

    #Use left point quadrature at this energy
    int_phi_sig_t += phi_i*sig_t_tot*dE
    int_phi_sig_s += phi_i*sig_s_tot*dE
    int_phi += phi_i*dE

    #check if hit bound, make CX
    if E > bounds[count] or Ei == len(energies)-2:
        print Ei
        print "Done with group", count
        cx_t.append(int_phi_sig_t/int_phi)
        cx_s.append(int_phi_sig_s/int_phi)
        int_phi_sig_t = 0.0
        int_phi_sig_s = 0.0
        int_phi = 0.0
        count += 1

print "Final cross sections: "
print "Scattering: ", [i for i in reversed(cx_s)]
print "Total: ", [i for i in reversed(cx_t)]

plt.show()










