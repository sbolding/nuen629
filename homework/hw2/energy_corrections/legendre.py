import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import math
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
from scipy import interpolate
from copy import deepcopy
from scipy.integrate import quadrature


def main():

    group_edges, sig_t, sig_el, scat_mat = proc_cx_file('hydrogen.cx')
    group_dE = [group_edges[i] - group_edges[i+1] for i in range(len(group_edges)-1)]

    #convert all cross sections to
    atom_dens = 0.00149

    sig_t = [i*atom_dens for i in sig_t]
    for key in scat_mat.keys():
        scat_mat[key] = [[i*atom_dens for i in x] for x in scat_mat[key]]
    sig_el = [i*atom_dens for i in sig_el]

    #Check 
    print "These two numbers should be equal", sum([scat_mat[0][i][4] for i in range(5)]), sig_el[4]

    #create the fission spectrum for U235
    chi = lambda E:  0.4865*np.sinh(np.sqrt(2.*E))*np.exp(-E)

    #Find group averaged chi's
    chi_groups = []
    for i in range(len(group_edges)-1):
        chi_g = quadrature(chi,group_edges[i+1],group_edges[i])[0]
        chi_groups.append(chi_g)
    chi_groups = np.array(chi_groups)

    #Build a matrix
    A = np.zeros((len(chi_groups),len(chi_groups)))

    #Same matrix for all methods
    S = scat_mat[0]
    for i in range(len(A)):
        A[i,i] = sig_t[i]
        for j in range(len(A[i])):
            A[i,j] -= S[i][j]

    #Solve system
    phi = np.linalg.solve(A,chi_groups)

    #normalize
    intphi = sum([group_dE[i]*phi[i] for i in range(len(phi))])
    phi = [i/intphi for i in phi]
    
    #let's make some plots
    fig = plt.figure()
    log_x = np.linspace(math.log10(group_edges[0]-0.000001),math.log10(group_edges[-1]),num=1000)
    x = [10.**i for i in log_x]
    y = []
    for i in x:
        for g in range(len(phi)):
            if i <= group_edges[g]:
                if i >= group_edges[g+1]:
                    y.append(phi[g])
    print len(x), len(y)
    plt.semilogx(x,y)
    plt.legend(loc=3) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel("$\phi(E)/|\phi(E)|_1$ MeV$^{-1}$")
    plt.xlabel("E (MeV)")
    plt.savefig("../method_compare.pdf",bbox_inches='tight')

def proc_cx_file(fname):

    groups = []
    sig_t = None
    sig_el = None
    scat_mat = dict()

    with open(fname) as f:
        lines = f.readlines()
        for line in lines:

            if re.search("Group boun", line):
                idx = lines.index(line)
                while True:
                    if re.search("^\s*$",lines[idx]):
                        break
                    else:
                        idx+=1
                        groups += [float(i)*1.E-06 for i in
                            lines[idx].split()]

            elif re.search("MT 1\s*$", line):
                sig_t = [float(i) for i in lines[lines.index(line)+1].split()]
            elif re.search("MT 2\s*$",line):
                sig_el =[float(i) for i in lines[lines.index(line)+1].split()]
            elif re.search(", Moment \d+",line):
                mom = float(re.search(", Moment (\d+)",line).group(1))
                idx = lines.index(line)+1
                mat = dict()
                    
                #Get all the scatering moments til we are done
                while True:
                    if idx >= len(lines) or re.search(", Moment \d+",lines[idx]) or re.search("^\s+$",lines[idx]):
                        scat_mat[mom] = mat
                        break
                    else:
                        (sink,first,last) = lines[idx].split()[3:6]
                        idx += 1
                        mat[int(sink)] = [float(i) for i in lines[idx].split()]
                        idx += 1

    #Convert scat matrix to an actual matrix
    for mom in scat_mat.keys():
        dic = scat_mat[mom]
        rows = max(dic.keys())+1
        mat = [None for j in range(rows)]
        for i in range(rows):
            new_row = [0.0 for j in range(rows)]
            for j in range(len(dic[i])):
                new_row[j] = dic[i][j]
            mat[i] = new_row

        scat_mat[mom] = np.array(mat)
            

    return groups, sig_t, sig_el, scat_mat


if __name__ == "__main__":
    main()








