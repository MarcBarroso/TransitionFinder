"""
Transition 2 fields. This program explores the parameter space (as specified in FindCouplings()) and print out the possible physical points in field space.
Authors: Daan Verweij, Marc Barroso.
"""

from potentials import *
import numpy as np
import multiprocessing
import numdifftools as nd
import warnings
import time

file_name = '2fields.txt' #name of the file where the points are going to be saved

l1min = 0
l1max = 0.2

l2min = 1e-2
l2max = 1e-10

l3min = -0.02
l3max = 0

gxmin = 0.2
gxmax = 1

def allMinima(m, n, Tmin, Thigh):
    temp = np.linspace(Tmin, Thigh, num=n)
    pos = np.ndarray((n, 2))
    for T in temp:
        minimum = m.findMinimum(T=T)
        pos = np.concatenate((pos, [minimum]), axis=0)

    print pos
    plt.figure()
    plt.axis([-20, 300, -500, 3500])
    plt.scatter(*zip(*pos))
    plt.show()

def CheckCouplings(params, verbose=False):
    """
    Checks whether the parameters sent in the array param correspond to a physical model (i.e. if one of the minima are close to the Higgs v (experimentally verified) and if one of the masses correspond to the Higgs mass (exp. verified as well).
    Input: params - array of l1, l2, l3 and gx. verbose - if True, it outputs all the information
    Output: the function will append the point to a file (file_name) if it corresponds to a physical model. If verbose=True, it outputs all the information through console regardless.
    """
    l1 = params[0]
    l2 = params[1]
    l3 = params[2]
    gx = params[3]
    m = model_2f(l1, l2, l3, y_t_interpol(np.log(v/mz)), gx)
    minima, success = m.findMinimum() #the boolean success is added because we cannot trust the minima if numpy.optimize.minimize has failed
    if not verbose:
        tolvevh = 2.0
        tolmh = 2.0
        condition0 = abs(minima-v) < tolvevh
        if condition0.any() and success:
            ddVtot = nd.Hessian(m.Vtot_0T)
            hess = ddVtot(minima)
            masses = np.linalg.eigvalsh(hess) #computes masses...
            positive_condition = masses > 0
            if(positive_condition.all()): #we will only check them IF they are positive
                masses = np.sqrt(np.abs(masses))
                condition1 = abs(masses-mh) < tolmh
                if condition1.any():
                    stability = m.CheckStability() #we check the stability of the model
                    f = open(file_name, 'a')
                    line0 = str(l1)+' '+str(l2)+' '+str(l3)+' '+str(gx)+' '+str(minima[0])+' '+str(minima[1])+' '+str(masses[0])+' '+str(masses[1]) #we print everything
                    line0 = line0 + ' '+str(stability)
                    f.write(line0+'\n')
                    f.write('-'*90+'\n')
                    f.close()
    else:
        """
        Just checks the minima of the model m, the masses of the particles and whether it is stable or not
        Output: prints the information
        """
        print "Minimum at T = 0.0: ", minima, success
        print "Masses: "
        ddVtot = nd.Hessian(m.Vtot_0T)
        hess = ddVtot(minima)
        print np.sqrt(np.linalg.eigvalsh(hess))
        print 'Stable: ', m.CheckStability()==1

def RandomFindCouplings():
    points = 10
    # Total points: points*multiprocessing.cpu_count()
    p = multiprocessing.Pool()
    f = open(file_name, 'w+')
    line = '|l1--l2--l3--gx--minima--mass1--mass2--stable|'
    f.write(line+'\n')
    f.write('-'*90+'\n')
    f.close()
    for i in range(points):
        start_time_loop = time.time()
        params = np.array([[0,0,0,0]])
        for j in range(multiprocessing.cpu_count()):
            l1 = np.random.uniform(l1min,l1max)
            l2 = np.random.uniform(l2min,l2max)
            l3 = np.random.uniform(l3min,l3max)
            gx = np.random.uniform(gxmin,gxmax)
            params1 = np.array([[l1, l2, l3, gx]])
            params = np.concatenate((params, params1), axis=0)
        params = np.delete(params, (0), axis=0)
        print params.shape
        p.map(CheckCouplings, params)
        print("--- Loop has taken: %s seconds ---" % (time.time() - start_time_loop))

def FindCouplings():
    """
    We get points to check, and parallelize the checking of the points. This is the function that does all the work in terms of exploring the parameter space.
    Input : -
    Output: -
    In order to optimize this function, the multiplication of all 'num' should be proportional to multiprocessing.cpu_count()
    """
    l1v = np.linspace(l1min, l1max, num=48)
    l2v = np.logspace(np.log10(l2min), np.log10(l2max), num=48)
    l3v = np.linspace(l3min, l3max, num=48)
    gxv = np.linspace(gxmin, gxmax, num=48)
    p = multiprocessing.Pool()
    f = open(file_name, 'w+')
    line = '|l1--l2--l3--gx--minima--mass1--mass2--stable|'
    f.write(line+'\n')
    f.write('-'*90+'\n')
    f.close()
    for l1 in l1v:
        for l2 in l2v:
            start_time_loop = time.time()
            params = cartesian((l1, -l2, l3v, gxv))
            print params.shape
            p.map(CheckCouplings, params)
            print("--- Loop has taken: %s seconds ---" % (time.time() - start_time_loop))

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    start_time = time.time()

    #FindCouplings()

    #RandomFindCouplings()

    CheckCouplings((0.175, -0.0049, -0.0038, 0.83))

    print("--- %s seconds ---" % (time.time() - start_time))
    #TO DO:
    # - Check with more points
