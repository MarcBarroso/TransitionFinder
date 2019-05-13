from scipy.optimize import minimize
import numpy as np
from cosmoTransitions import generic_potential
import multiprocessing as mp
import numdifftools as nd
import warnings

import time

v2 = 246.**2

class model1(generic_potential.generic_potential):
    """
    A sample model which makes use of the *generic_potential* class.

    This model doesn't have any physical significance. Instead, it is chosen
    to highlight some of the features of the *generic_potential* class.
    It consists of two scalar fields labeled *phi1* and *phi2*, plus a mixing
    term and an extra boson whose mass depends on both fields.
    It has low-temperature, mid-temperature, and high-temperature phases, all
    of which are found from the *getPhases()* function.
    """
    def init(self, l1, l2, l3, yt, gx):
        """
          m1 - tree-level mass of first singlet when mu = 0.
          m2 - tree-level mass of second singlet when mu = 0.
          mu - mass coefficient for the mixing term.
          Y1 - Coupling of the extra boson to the two scalars individually
          Y2 - Coupling to the two scalars together: m^2 = Y2*s1*s2
          n - degrees of freedom of the boson that is coupling.
        """
        # The init method is called by the generic_potential class, after it
        # already does some of its own initialization in the default __init__()
        # method. This is necessary for all subclasses to implement.

        # This first line is absolutely essential in all subclasses.
        # It specifies the number of field-dimensions in the theory.
        self.Ndim = 2

        # self.renormScaleSq is the renormalization scale used in the
        # Coleman-Weinberg potential.
        self.renormScaleSq = v2

        # This next block sets all of the parameters that go into the potential
        # and the masses. This will obviously need to be changed for different
        # models.
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.gx = gx

        sOW2    = 0.23122
        sOW     = np.sqrt(sOW2)
        e       = 0.3028221208
        self.g  = e/sOW
        self.gp = e/(2*sOW*np.sqrt(1-sOW2))
        self.yt = yt
        self.n1 = 9
        self.n2 = 9
        self.n3 = 9
        self.n4 = 4

    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is useful to set if there is, for example, a Z2 symmetry
        in the theory and you don't want to double-count all of the phases. In
        this case, we're throwing away all phases whose zeroth (since python
        starts arrays at 0) field component of the vev goes below -5. Note that
        we don't want to set this to just going below zero, since we are
        interested in phases with vevs exactly at 0, and floating point numbers
        will never be accurate enough to ensure that these aren't slightly
        negative.
        """
        return (np.array([X])[...,0] < -5.0).any()

    def V0(self, X):
        """
        This method defines the tree-level potential. It should generally be
        subclassed. (You could also subclass Vtot() directly, and put in all of
        quantum corrections yourself).
        """
        # X is the input field array. It is helpful to ensure that it is a
        # numpy array before splitting it into its components.
        X = np.asanyarray(X)
        # x and y are the two fields that make up the input. The array should
        # always be defined such that the very last axis contains the different
        # fields, hence the ellipses.
        # (For example, X can be an array of N two dimensional points and have
        # shape (N,2), but it should NOT be a series of two arrays of length N
        # and have shape (2,N).)
        phi1,phi2 = X[...,0], X[...,1]
        r = 0.25 *(self.l1*phi1*phi1*phi1*phi1+self.l2*phi1*phi1*phi2*phi2+self.l3*phi2*phi2*phi2*phi2)
        return r

    def boson_massSq(self, X, T):
        XT = np.array(X)
        X = XT.reshape(XT.size/2, 2)
        m1 = np.empty([0,0])
        m2 = np.empty([0,0])
        mb1 = np.empty([0,0])
        mb2 = np.empty([0,0])
        mb3 = np.empty([0,0])
        #ddV0 = nd.Hessian(self.V0)
        for vec in X:
            phi1 = vec[0]
            phi2 = vec[1]

            #hess = ddV0([phi1,phi2])
            #print hess
            #eigen =  np.linalg.eigvalsh(hess)

            MSQ = np.array([[3*self.l1*phi1*phi1+0.5*self.l2*phi2*phi2, self.l2*phi1*phi2],
                           [self.l2*phi1*phi2, 3*self.l3*phi2*phi2+0.5*self.l2*phi1*phi1]])

            eigen = np.linalg.eigvalsh(MSQ)

            m1 = np.append(m1, eigen[0])
            m2 = np.append(m2, eigen[1])
            mb1t = (0.5*self.g*phi1)**2
            mb2t = (0.5*self.gp*phi1)**2 + (0.5*self.g*phi1)**2
            mb3t = (0.5*self.gx*phi2)**2
            mb1 = np.append(mb1, mb1t)
            mb2 = np.append(mb2, mb2t)
            mb3 = np.append(mb3, mb3t)

        if(X.shape[0] == 1):
            M = np.array([m1[0], m2[0], mb1[0], mb2[0], mb3[0]])
        else:
            M = np.array([m1, m2, mb1, mb2, mb3])

        # At this point, we have an array of boson masses, but each entry might
        # be an array itself. This happens if the input X is an array of points.
        # The generic_potential class requires that the output of this function
        # have the different masses lie along the last axis, just like the
        # different fields lie along the last axis of X, so we need to reorder
        # the axes. The next line does this, and should probably be included in
        # all subclasses.
        M = np.rollaxis(M, 0, len(M.shape))

        # The number of degrees of freedom for the masses. This should be a
        # one-dimensional array with the same number of entries as there are
        # masses.
        dof = np.array([1, 1, self.n1, self.n2, self.n3])

        # c is a constant for each particle used in the Coleman-Weinberg
        # potential using MS-bar renormalization. It equals 1.5 for all scalars
        # and the longitudinal polarizations of the gauge bosons, and 0.5 for
        # transverse gauge bosons.
        cnumb = 5.0/6.0
        cnums = 1.5
        c = np.array([cnums, cnums, cnumb, cnumb, cnumb])

        return M, dof, c

    def Vtot_0T(self, X):
        return self.Vtot(X, 0.0)

    def fermion_massSq(self, X):
        """
        Calculate the fermion particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.

        Returns
        -------
        massSq : array_like
            A list of the fermion particle masses at each input point `X`. The
            shape should be such that  ``massSq.shape == (X[...,0]).shape``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.

        Notes
        -----
        Unlike :func:`boson_massSq`, no constant `c` is needed since it is
        assumed to be `c = 3/2` for all fermions. Also, no thermal mass
        corrections are needed.
        """
        # The following is an example placeholder which has the correct output
        # shape. Since dof is zero, it does not contribute to the potential.
        Nfermions = 1
        phi1,phi2 = X[...,0], X[...,1]

        m1 = (self.yt*phi1/np.sqrt(2))**2
        massSq = np.empty(m1.shape + (Nfermions,))
        massSq[...,0] = m1
        dof = np.array([4])
        return massSq, dof

    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        return [np.array([v2**.5,2411])]


    def findMinimum(self, X=None, T=0.0):
        """
        Convenience function for finding the nearest minimum to `X` at
        temperature `T`.
        """
        if X is None:
            X = self.approxZeroTMin()[0]

        bnds = ((0, None), (0, None))
        min1 = minimize(self.Vtot, X, args=(T,), method='TNC', bounds=bnds, tol=1e-6)
        min0 = self.Vtot((0,0), T)
        if self.Vtot(min1.x, T) < min0:
            return min1.x, min1.success
        else:
            return np.empty([2]), min1.success

    def dV(self, mT=0.0):
        minimum = self.findMinimum((v2**.5,2411), T)
        dV =  self.Vtot(minima, 0.0) - self.Vtot([0.0, 0.0], 0.0)
        return dV

    def plot3d(self, box, T=0, offset=0,
               xaxis=0, yaxis=1, n=50, clevs=200, cfrac=.8, **contourParams):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        xmin,xmax,ymin,ymax = box
        X = np.linspace(xmin, xmax, n).reshape(n,1)*np.ones((1,n))
        Y = np.linspace(ymin, ymax, n).reshape(1,n)*np.ones((n,1))
        XY = np.zeros((n,n,self.Ndim))
        XY[...,xaxis], XY[...,yaxis] = X,Y
        XY += offset
        Z = self.Vtot(XY,T)
        minZ, maxZ = min(Z.ravel()), max(Z.ravel())
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

def makePlots(m, T, xlow, xhigh, ylow, yhigh):
        boxlowx = xlow
        boxhighx = xhigh
        boxlowy = ylow
        boxhighy = yhigh
        m.plot3d((boxlowx, boxhighx, boxlowy, boxhighy), T)
        #m.plot2d((boxlowx, boxhighx, boxlowy, boxhighy), T)

def allMinima(m, n, Tmin, Thigh):
    temp = np.linspace(Tmin, Thigh, num=n)
    pos = np.ndarray((n, 2))
    for T in temp:
        minimum = m.findMinimum((v2**.5,2411), T)
        pos = np.concatenate((pos, [minimum]), axis=0)

    print pos
    plt.figure()
    plt.axis([-20, 300, -500, 3500])
    plt.scatter(*zip(*pos))
    plt.show()

def VatSomePoint(m, X, T):
    bosons = m.boson_massSq(X,T)
    fermions = m.fermion_massSq(X)

    print "--------------------------------------------------------"
    print "Point: ", X, " Temperature: ", T
    print "Tree-level potential: ", m.V0(X)
    print "Tree-level + one-loop correction: ", m.V0(X)+m.V1(bosons, fermions)
    print "Sum of tree+one loop+thermal: ", m.Vtot(X, T)
    print "--------------------------------------------------------"

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def CheckCouplings(param):
    l1 = param[0]
    l2 = param[1]
    l3 = param[2]
    gx = param[3]
    m = model1(l1, l2, l3, 0.9, gx)
    minima, success = m.findMinimum((v2**.5,2000), 0.0)
    condition0 = (abs(minima[0]-246) < 10 or abs(minima[1]-246) < 10) and success
    if condition0:
    	ddVtot = nd.Hessian(m.Vtot_0T)
        hess = ddVtot(minima, 0.0)
        masses = np.linalg.eigvalsh(hess)
        positive_condition = masses > 0
        if(positive_condition.all()):
            masses = np.sqrt(np.abs(masses))
            condition1 = abs(masses[0]-125) < 5 or abs(masses[1]-125) < 5
            if condition1:
                f = open('2fields.txt', 'a')
                line0 = '| l1 = '+str(l1)+' l2 = '+str(l2)+' l3 = '+str(l3)+' gx = '+str(gx)+' minima: '+str(minima)+' masses: '+str(masses)
                print line0
                print '-'*90
                f.write(line0+'\n')
                f.write('-'*90+'\n')
                f.close()

def FindCouplings():
    l1v = np.linspace(-1, 1, num=10)
    l2v = np.linspace(-0.1, 0.1, num=10)
    l3v = np.linspace(-1, 1, num=10)
    gxv = np.linspace(-1, 1, num=10)
    p = mp.Pool()
    for l1 in l1v:
        for l2 in l2v:
            for l3 in l3v:
                params = cartesian((l1, l2, l3, gxv))
                p.map(CheckCouplings, params)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    start_time = time.time()
    m = model1(0.07692307692307693, -0.0836734693877551, 0.02564102564102564, 0.9, 1.0)
    T = 0.0

    #FindCouplings()

    minima, success = m.findMinimum((v2**.5, 2000), T)
    print "Minimum at T = ", T, ": ", minima, success

    print "Masses: "
    ddVtot = nd.Hessian(m.Vtot_0T)
    hess = ddVtot(minima)
    print np.sqrt(np.linalg.eigvalsh(hess))

    #VatSomePoint(m, minima, T)

    #allMinima(m, 100, 200, 400)
    #m.findAllTransitions()
    #m.phases.pop(3)
    #print m.TnTrans
    #plt.figure()
    #m.plotPhasesPhi()
    #plt.axis([0,700,-50,20000])
    #plt.title("Minima as a function of temperature")
    #plt.show()
    #m.calcTcTrans()
    print("--- %s seconds ---" % (time.time() - start_time))

# Add thermal mass
# Implement the Euclidean action.
