from cosmoTransitions import generic_potential
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import numpy as np

"""
We first define some physical constant that we will use later.
mh = Higgs mass
mz = Z gauge boson mass
mw = W gauge boson mass
mp = Planck mass
"""
mh = 125.18
mz = 91.1876
mw = 80.379
mp = 2.435*10**(18)
"""
We define the sin(theta_W)^2, where theta_W is the Weinberg angle
Alpha is the coupling constant of electromagnetism, and alpha_s the coupl. const. of the strong interaction. We will need them to compute the beta functions
"""
s2thetaW = 0.23122
alpha = 1.0/128.0
alpha_s = 0.1182
"""
We now define the couplings of the electro and weak parts, squared
"""
g2W = 4.0*np.pi*alpha/s2thetaW
gp2W = 4.0*np.pi*alpha/(1.0-s2thetaW)
"""
And the vev of the Higgs boson
"""
v = 2.0*mw/np.sqrt(g2W)
v2 = v*v

"""
We define the beta functions of some of the couplings, and solve for the gauge ones which can be done analytically.
As before, gs2 is the coupling of the strong interaction, squared. g2 for electromagnetism, g2p for the weak
Source for the gauge beta functions: arXiv: 1208.3357
"""
def gs2(x):
    return 4.0*np.pi*alpha_s/(1.0+7.0/2.0*alpha_s/np.pi*x)

def g2(x):
    return g2W/(1.0+19.0/(48.0*np.pi**2)*g2W*x)

def gp2(x):
    return gp2W/(1.0-41.0/(48.0*np.pi**2)*gp2W*x)

def beta_top(y, x):
    """
    This is the beta function of the top quark. Careful with the conventions used.
    Input: y - value of the Yukawa top coupling. x - log(energy) we are studying
    Output: value of the beta function at the point x
    Source: arXiv: 1212.6829 and private notes
    """
    dydx = 1/(16.0*np.pi**2)*y*(9.0/2.0*y**2-8.0*gs2(x)-9.0/4.0*g2(x)-17.0/12.0*gp2(x))
    return dydx

xmin = np.log(173.0/mz)
xmax = np.log(mp/mz) + 5.0
y0 = 0.9
x = np.linspace(xmin, xmax, 2000)
sol_y_t = odeint(beta_top, y0, x)
y_t_interpol = interp1d(x, np.asarray(sol_y_t).squeeze(), kind='cubic')

class model_2f(generic_potential.generic_potential):
    """
    Model for the case with the Higgs scalar + the SU(2)_X sector.
    """
    def init(self, l1, l2, l3, yt, gx):
        """
        l1, l2, l3 - couplings in the potential, defines at the energy v (VEV of the Higgs, around 246 GeV)
        gx - coupling of the X gauge boson
        yt - value of the Yukawa coupling of the top quark at v
        n - degrees of freedom of the boson that is coupling.
        """
        # It specifies the number of field-dimensions in the theory.
        self.Ndim = 2

        # self.renormScaleSq is the renormalization scale used in the
        # Coleman-Weinberg potential.
        self.renormScaleSq = v2

        # This next block sets all of the parameters that go into the potential
        # and the masses.
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.gx = gx
        self.yt = yt
        self.params = [l1, l2, l3, gx]

        self.g  = np.sqrt(g2(np.log(self.renormScaleSq/mz)))
        self.gp = np.sqrt(gp2(np.log(self.renormScaleSq/mz)))
        self.n1 = 6
        self.n2 = 3
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
        This method defines the tree-level potential.
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
        """
	Computation of the value of the boson masses squared.
	Input: X - point in field space. T - temperature.
	Output: M - array with the masses squared. c - renormalization constants. dof - degrees of freedom
	"""
	XT = np.array(X)
	X = XT.reshape(XT.size/2, 2) #Since we can overload our input by passing an array of points, we have to make sure to be dealing with it point by point.
	m1  = np.empty([0,0])
	m2  = np.empty([0,0])
	mb1 = np.empty([0,0])
	mb2 = np.empty([0,0])
	mb3 = np.empty([0,0])
	#ddV0 = nd.Hessian(self.V0) Uncomment this and the lines below if the hessian is to be computed numerically directly from V0
	for vec in X:
	    phi1 = vec[0]
	    phi2 = vec[1]

	    #hess = ddV0([phi1,phi2])
	    #print hess
	    #eigen =  np.linalg.eigvalsh(hess)

	    MSQ = np.array([[3*self.l1*phi1*phi1+0.5*self.l2*phi2*phi2, self.l2*phi1*phi2], #We can compute the matrix of second derivatives by hand in this case
		[self.l2*phi1*phi2, 3*self.l3*phi2*phi2+0.5*self.l2*phi1*phi1]])

	    eigen = np.linalg.eigvalsh(MSQ) #and just ask for the eigenvalues

	    m1 = np.append(m1, eigen[0])
	    m2 = np.append(m2, eigen[1])

	    mb1 = np.append(mb1, (0.5*self.g*phi1)**2) #here we compute the mass of the gauge bosons
	    mb2 = np.append(mb2, (0.5*self.gp*phi1)**2 + (0.5*self.g*phi1)**2)
	    mb3 = np.append(mb3, (0.5*self.gx*phi2)**2)

	if(X.shape[0] == 1): #PLACEHOLDER: just making sure that the end result is consistent
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
        """
        # The following is an example placeholder which has the correct output
        # shape. Since dof is zero, it does not contribute to the potential.
        Nfermions = 1
        phi1,phi2 = X[...,0], X[...,1]

        m1 = (self.yt*phi1/np.sqrt(2))**2
        massSq = np.empty(m1.shape + (Nfermions,))
        massSq[...,0] = m1
        dof = np.array([12])
        return massSq, dof

    def findMinimum(self, X=np.array([v, 2000]), T=0.0):
        """
        Convenience function for finding the nearest minimum to `X` at
        temperature `T`.
        """
        bnds = ((0, None), (0, None))
        min1 = minimize(self.Vtot, X, args=(T,), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        min0 = self.Vtot((0,0), T)
        if self.Vtot(min1.x, T) < min0:
            return min1.x, min1.success
        else:
            return np.empty([2]), min1.success

    def VatSomePoint(self, X, T):
        """
        Implementation of a sanity check.
        Input: m - model we are studying. X - point in field space. T - temperature.
        Output: print of the tree-level, one-loop and thermal correction of the potential of model m at point X and temperature T
        """
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        print "--------------------------------------------------------"
        print "Point: ", X, " Temperature: ", T
        print "Tree-level potential: ", self.V0(X)
        print "Tree-level + one-loop correction: ", self.V0(X)+self.V1(bosons, fermions)
        print "Sum of tree+one loop+thermal: ", self.Vtot(X, T)
        print "--------------------------------------------------------"

    def Beta_fun(self, y, x):
        """
        Beta function for l1, l2, l3 and gx, all at once.
        Input: y - list with the values of l1, l2, l3 and gx. x - log(energy) we are studying. y_t_interpol - interpolated function for the Yukawa top coupling.
        Output: list with the value of the beta functions at the point x
        Beta functions adopted from arXiv: 1805.09292.
        """
        l1, l2, l3, gx = y
        beta_l1 = (12.0*l1**2+l2**2+0.5*l1*(-9.0*g2(x)-3.0*gp2(x)+12.0*y_t_interpol(x)**2)+3.0/8.0*g2(x)**2+3.0*(g2(x)+gp2(x))**2/16.0-3.0*y_t_interpol(x)**4)
        beta_l1 = 1.0/(8.0*np.pi**2)*beta_l1
        beta_l2 = 1.0/(8.0*np.pi**2)*(6.0*l1*l2+2.0*l2**2+6.0*l2*l3+0.25*l2*(-9.0*g2(x)-3.0*gp2(x)+12.0*y_t_interpol(x)**2-9.0*gx**2))
        beta_l3 = 1.0/(8.0*np.pi**2)*(l2**2+12.0*l3**2-9.0/2.0*l3*gx**2+9.0/16.0*gx**4)
        beta_gx = 1.0/(16.0*np.pi**2)*(-43.0/6.0*gx**3-1.0/(4.0*np.pi)**2*256.0/6.0*gx**5)
        dydx = [beta_l1, beta_l2, beta_l3, beta_gx]
        return dydx

    def CheckStability(self):
        """
        Checks if the combination l1, l2, l3 and gx satisfy the positivity condition at the Planck scale.
        Output: 1 if the model is stable, 0 otherwise.
        """
        xmin = np.log(v/mz)
        xmax = np.log(mp/mz)
        N = 1000 #This changes the amount of points in our numerical solution of the beta functions
        x = np.linspace(xmin, xmax, N)
        sol_l = odeint(self.Beta_fun, self.params, x)
        l1mp = sol_l[N-1, 0]
        l2mp = sol_l[N-1, 1]
        l3mp = sol_l[N-1, 2]
        if l1mp >= 0 and l3mp >= 0:
            if l2mp >= -2.0*np.sqrt(l1mp*l3mp):
                return 1
        return 0

class model_3f(generic_potential.generic_potential):
    """
    Model for the case with the Higgs scalar + the SU(2)_X sector + singlet
    """
    def init(self, l1, l2, l3, l4, l5, l6, yt, gx):
        """
        l1, l2, l3, l4, l5, l6 - couplings in the potential, defines at the energy v (VEV of the Higgs, around 246 GeV)
        gx - coupling of the X gauge boson
        yt - value of the Yukawa coupling of the top quark at v
        n - degrees of freedom of the boson that is coupling.
        """
        # It specifies the number of field-dimensions in the theory.
        self.Ndim = 3

        # self.renormScaleSq is the renormalization scale used in the
        # Coleman-Weinberg potential.
        self.renormScaleSq = v2

        # This next block sets all of the parameters that go into the potential
        # and the masses.
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.l6 = l6
        self.gx = gx
        self.yt = yt
        self.params = [l1, l2, l3, l4, l5, l6, gx]

        self.g  = np.sqrt(g2(np.log(v/mz)))
        self.gp = np.sqrt(gp2(np.log(v/mz)))
        self.n1 = 6
        self.n2 = 3
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
        This method defines the tree-level potential.
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
        phi1,phi2, phi3 = X[...,0], X[...,1], X[...,2]
        phi1_2 = phi1*phi1
        phi2_2 = phi2*phi2
        phi3_2 = phi3*phi3
        r = 0.25 *(self.l1*phi1_2*phi1_2+self.l2*phi1_2*phi2_2+self.l3*phi2_2*phi2_2+self.l4*phi3_2*phi3_2+self.l5*phi1_2*phi3_2+self.l6*phi3_2*phi2_2)
        return r

    def boson_massSq(self, X, T):
        """
        Computation of the value of the boson masses squared.
        Input: X - point in field space. T - temperature.
        Output: M - array with the masses squared. c - renormalization constants. dof - degrees of freedom
        """
        XT = np.array(X)
        X = XT.reshape(XT.size/3, 3) #Since we can overload our input by passing an array of points, we have to make sure to be dealing with it point by point.
        # We need to define the field-dependent boson masses. This is obviously
        # model-dependent.
        # Note that these can also include temperature-dependent corrections.
        m1  = np.empty([0,0,0])
        m2  = np.empty([0,0,0])
	m3  = np.empty([0,0,0])
        mb1 = np.empty([0,0,0])
        mb2 = np.empty([0,0,0])
        mb3 = np.empty([0,0,0])
        #ddV0 = nd.Hessian(self.V0) Uncomment this and the lines below if the hessian is to be computed numerically directly from V0
        for vec in X:
            phi1 = vec[0]
            phi2 = vec[1]
	    phi3 = vec[2]
            phi1_2 = phi1*phi1
            phi2_2 = phi2*phi2
            phi3_2 = phi3*phi3
	    MSQ = np.array([[3*self.l1*phi1_2+0.5*self.l2*phi2_2+0.5*self.l5*phi3_2, self.l2*phi1*phi2, self.l5*phi1*phi3],
                        [self.l2*phi1*phi2, 3*self.l2*phi2_2+0.5*self.l2*phi1_2+0.5*self.l5*phi3_2, self.l6*phi3*phi2],
                        [self.l5*phi1*phi3, self.l6*phi3*phi2, 3*self.l4*phi3_2+0.5*self.l5*phi1_2+0.5*self.l6*phi2_2]])
            eigen = np.linalg.eigvalsh(MSQ) #and just ask for the eigenvalues

            m1 = np.append(m1, eigen[0])
            m2 = np.append(m2, eigen[1])
	    m3 = np.append(m3, eigen[2])

            mb1 = np.append(mb1, (0.5*self.g*phi1)**2) #here we compute the mass of the gauge bosons
            mb2 = np.append(mb2, (0.5*self.gp*phi1)**2 + (0.5*self.g*phi1)**2)
            mb3 = np.append(mb3, (0.5*self.gx*phi2)**2)

        if(X.shape[0] == 1): #PLACEHOLDER: just making sure that the end result is consistent
            M = np.array([m1[0], m2[0], m3[0], mb1[0], mb2[0], mb3[0]])
        else:
            M = np.array([m1, m2, m3, mb1, mb2, mb3])
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
        dof = np.array([1, 1, 1, self.n1, self.n2, self.n3])

        # c is a constant for each particle used in the Coleman-Weinberg
        # potential using MS-bar renormalization. It equals 1.5 for all scalars
        # and the longitudinal polarizations of the gauge bosons, and 0.5 for
        # transverse gauge bosons.
        cnumb = 5.0/6.0
        cnums = 1.5
        c = np.array([cnums, cnums, cnums, cnumb, cnumb, cnumb])

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
        Nfermions = 1
        phi1, phi2, phi3 = X[...,0], X[...,1], X[...,2]

        m1 = (self.yt*phi1/np.sqrt(2))**2

        massSq = np.empty(m1.shape + (Nfermions,))
        massSq[...,0] = m1
        dof = np.array([12])
        return massSq, dof

    def findMinimum(self, X=np.array([v, 2000, 2000]), T=0.0):
        """
        Convenience function for finding the nearest minimum to `X` at
        temperature `T`.
        """
        bnds = ((0, None), (0, None), (0, None))
        min1 = minimize(self.Vtot, X, args=(T,), method='L-BFGS-B', bounds=bnds, tol=1e-5)
        min0 = self.Vtot((0,0,0), T)
        if self.Vtot(min1.x, T) < min0:
            return min1.x, min1.success
        else:
            return np.empty([3]), min1.success

    def VatSomePoint(m, X, T):
        """
        Implementation of a sanity check.
        Input: m - model we are studying. X - point in field space. T - temperature.
        Output: print of the tree-level, one-loop and thermal correction of the potential of model m at point X and temperature T
        """
        bosons = m.boson_massSq(X,T)
        fermions = m.fermion_massSq(X)

        print "--------------------------------------------------------"
        print "Point: ", X, " Temperature: ", T
        print "Tree-level potential: ", m.V0(X)
        print "Tree-level + one-loop correction: ", m.V0(X)+m.V1(bosons, fermions)
        print "Sum of tree+one loop+thermal: ", m.Vtot(X, T)
        print "--------------------------------------------------------"

    def Beta_fun(self, y, x):
        """
        Beta function for l1, l2, l3, l4, l5, l6 and gx, all at once.
        Input: y - list with the values of li and gx. x - log(energy) we are studying.
        Output: list with the value of the beta functions at the point x
        Beta functions adopted from arXiv: 1508.03031 (they use SU(5) convention for g and g') and 1805.09292. We haven't included Yukawa of neutrinos yet.
        """
        l1, l2, l3, l4, l5, l6, gx = y
        beta_l1 = (12*l1**2+l2**2+0.25*l5**2+0.5*l1*(-9.0*g2(x)-3.0*gp2(x)+12.0*y_t_interpol(x)**2)+3.0/8.0*g2(x)**2+3.0*(g2(x)+gp2(x))**2/16.0-3.0*y_t_interpol(x)**4)
        beta_l1 = 1.0/(8.0*np.pi**2)*beta_l1
        beta_l2 = 1.0/(8.0*np.pi**2)*(6.0*l1*l2+2.0*l2**2+l5*l6+6.0*l2*l3+0.25*l2*(-9.0*g2(x)-3.0*gp2(x)+12.0*y_t_interpol(x)**2-9.0*gx**2))
        beta_l3 = 1.0/(8.0*np.pi**2)*(l2**2+12.0*l3**2+0.25*l6**2-9.0/2.0*l3*gx**2+9.0/16.0*gx**4)
        beta_l4 = 1.0/(8.0*np.pi**2)*(9.0*l4**2+l5**2+l6**2)
        beta_l5 = 1.0/(8.0*np.pi**2)*(l5*(3.0*y_t_interpol(x)**2+6.0*l1+3.0*l4+2.0*l5-9.0*g2(x)-3.0*gp2(x))+2.0*l2*l6)
        beta_l6 = 1.0/(8.0*np.pi**2)*(l6*(6.0*l3+3.0*l4-2.0*l6-9.0/2.0*gx**2)+2.0*l5*l2)
        beta_gx = 1.0/(16.0*np.pi**2)*(-43.0/6.0*gx**3-1.0/(4.0*np.pi)**2*256.0/6.0*gx**5)
        dydx = [beta_l1, beta_l2, beta_l3, beta_l4, beta_l5, beta_l6, beta_gx]
        return dydx

    def CheckStability(self):
        """
        Checks if the combination l1, l2, l3, l4, l5, l6 and gx satisfy the positivity condition at the Planck scale.
        Input: -
        Output: 1 if the model is stable, 0 otherwise.
        Source: arXiv: 1508.03031
        """
        xmin = np.log(v/mz)
        xmax = np.log(mp/mz)
        N = 1000 #This changes the amount of points in our numerical solution of the beta functions
        x = np.linspace(xmin, xmax, N)
        sol_l = odeint(self.Beta_fun, self.params, x)
        l1mp = sol_l[N-1, 0]
        l2mp = sol_l[N-1, 1]
        l3mp = sol_l[N-1, 2]
        l4mp = sol_l[N-1, 3]
        l5mp = sol_l[N-1, 4]
        l6mp = sol_l[N-1, 5]
        condition0 = l1mp >= 0 and l3mp >= 0 and l4mp >= 0 #equation 2.6 from the paper
        if condition0:
            q1 = -l2mp/(2.0*np.sqrt(l1mp*l3mp))
            q2 = -l5mp/(2.0*np.sqrt(l1mp*l4mp))
            q3 = -l6mp/(2.0*np.sqrt(l3mp*l4mp))
            condition1 = q1 <= 1 and q2 <= 1 and q3 <= 1 #2.7
            if condition1:
                condition2 = q1+q2+q3 <= 1 #2.9
                if condition2:
                    return 1
        return 0

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
