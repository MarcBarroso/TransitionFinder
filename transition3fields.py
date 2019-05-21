"""
Transition 3 fields. This program explores the parameter space (as specified in FindCouplings()) and print out the possible physical points in field space.
Authors: Daan Verweij, Marc Barroso.
"""

from potentials import *
import numpy as np
import multiprocessing
import numdifftools as nd
import warnings
import time

file_name = '3fields_random.txt' #name of the file where the points are going to be saved

l1min = 0
l1max = 0.2

l2min = 1e-2
l2max = 1e-10

l3min = -0.02
l3max = 0

l4min = -0.2
l4max = 0.2

l5min = -0.2
l5max = 0.2

l6min = -0.2
l6max = 0.2

gxmin = 0.2
gxmax = 1

def CheckCouplings(params, verbose=False):
    """
    Checks whether the parameters sent in the array param correspond to a physical model (i.e. if one of the minima are close to the Higgs v (experimentally verified) and if one of the masses correspond to the Higgs mass (exp. verified as well).
    Input: params - array of l1, l2, l3 and gx.
    Output: the function will append the point to a file (file_name) if it corresponds to a physical model.
    """
    l1 = params[0]
    l2 = params[1]
    l3 = params[2]
    l4 = params[3]
    l5 = params[4]
    l6 = params[5]
    gx = params[6]
    m = model_3f(l1, l2, l3, l4, l5, l6, y_t_interpol(np.log(v/mz)), gx)
    minima, success = m.findMinimum() #the boolean success is added because we cannot trust the minima if numpy.optimize.minimize has failed
    if not verbose:
        tolvevh = 2.0
        tolmh = 2.0
        condition0 = abs(minima-v) < tolvevh
        if condition0.any() and success:
            ddVtot = nd.Hessian(m.Vtot_0T)
            hess = ddVtot(minima)
            try:
                masses = np.linalg.eigvalsh(hess) #computes masses...
                positive_condition = masses > 0
            except LinAlgError:
                positive_condition = np.array([False])
            if(positive_condition.all()): #we will only check them IF they are positive
                masses = np.sqrt(np.abs(masses))
                condition1 = abs(masses-mh) < tolmh
                if condition1.any():
                    stability = m.CheckStability() #we check the stability of the model
                    f = open(file_name, 'a')
                    line0 = str(l1)+' '+str(l2)+' '+str(l3)+' '+str(l4)+' '+str(l5)+' '+str(l6)+' '+str(gx)+' '+str(minima[0])+' '+str(minima[1])+' '+str(minima[2])+' '+str(masses[0])+' '+str(masses[1])+' '+str(masses[2])+' '+str(stability) #we print everything
                    f.write(line0+'\n')
                    f.write('-'*90+'\n')
                    f.close()
    else:
        print "Minimum at T = 0.0: ", minima, success
        print "Masses: "
        ddVtot = nd.Hessian(m.Vtot_0T)
        hess = ddVtot(minima)
        print np.sqrt(np.linalg.eigvalsh(hess))
        print 'Stable: ', m.CheckStability()==1

def RandomFindCouplings():
    points = 1000000
    # Total points: points*multiprocessing.cpu_count()
    p = multiprocessing.Pool()
    f = open(file_name, 'w+')
    line = '|l1--l2--l3--l4--l5--l6--gx--minima--mass1--mass2--mass3--stable|'
    f.write(line+'\n')
    f.write('-'*90+'\n')
    f.close()
    for i in range(points):
        start_time_loop = time.time()
        params = np.array([[0,0,0,0,0,0,0]])
        for j in range(multiprocessing.cpu_count()):
            l1 = np.random.uniform(l1min,l1max)
            l2 = np.random.uniform(l2min,l2max)
            l3 = np.random.uniform(l3min,l3max)
            l4 = np.random.uniform(l4min,l4max)
            l5 = np.random.uniform(l5min,l5max)
            l6 = np.random.uniform(l6min,l6max)
            gx = np.random.uniform(gxmin,gxmax)
            params1 = np.array([[l1, l2, l3, l4, l5, l6, gx]])
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
    l4v = np.linspace(l4min, l4max, num=10)
    l5v = np.linspace(l5min, l5max, num=10)
    l6v = np.linspace(l6min, l6max, num=10)
    gxv = np.linspace(gxmin, gxmax, num=48)
    p = multiprocessing.Pool()
    f = open(file_name, 'w+')
    line = '|l1--l2--l3--l4--l5--l6--gx--minima--mass1--mass2--mass3--stable|'
    f.write(line+'\n')
    f.write('-'*90+'\n')
    f.close()
    for l1 in l1v:
        for l2 in l2v:
            for l3 in l3v:
                for l4 in l4v:
                    for l5 in l5v:
                        start_time_loop = time.time()
                        params = cartesian((l1, l2, l3, l4, l5, l6v, gxv))
                        print params.shape
                        p.map(CheckCouplings, params)
                        print("--- Loop has taken: %s seconds ---" % (time.time() - start_time_loop))

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    start_time = time.time()

    #FindCouplings()

    RandomFindCouplings()

    #CheckCouplings((0.1276, 0.0036, 0.004, 0.2257, 0.001, 0.06, 0.95), True)

    print("--- %s seconds ---" % (time.time() - start_time))
