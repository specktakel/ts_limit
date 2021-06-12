import numpy as np
import sys
from scipy.integrate import simps


class structured_field():
    '''Class definition of structured magnetic field, see 1008.5353 and
    1908.03084 for details.'''

    '''alpha is lowest positive, non-zero root of
    tan(alpha)=3alpha/(3-alpha**2). Put F_0 and norm here as well,
    they do not depend on anything else.'''
    alpha = 5.7634591968
    F_0 = (alpha * np.cos(alpha) - np.sin(alpha)) * alpha**2
    norm = np.sqrt((3 * F_0 + alpha**5)**2) * 2 / (3 * alpha**2)

    def __init__(self, B_0, R, theta, radians=False, cell_num=100, **kwargs):
        '''Norm in micro gauss, theta in degrees, if not specified.'''
        if radians:
            self.theta = theta
        else:
            self.theta = np.radians(theta)
            self._theta_ = theta
        print(self.theta)
        self.B_0 = B_0
        # Normalisation from \lim_{r\to 0}
        self.B_0 = self.B_0
        self.R = R
        self._r_scale = 1 / self.R
        self.cell_num = cell_num
        self.dL = R / cell_num
        self.r = self._get_r_points()
        # TODO: need to implement some other way to calculate dL when manual
        # set of self.r is done.
        self.dL_vec = np.full(self.r.shape, self.dL)

    def _get_r_points(self):
        return np.linspace(self.dL / 2, self.R - self.dL / 2, self.cell_num)

    '''r needs to be rescaled in field strength expressions to be smaller
    than one. this is done here. calling b.r multiplies with R, setting divides
    by R.'''
    @property
    def r(self):
        return self._r * self.R

    @r.setter
    def r(self, val=None):
        if val is None:
            print('Resetting radial points and dL_vec')
            self._r = self._get_r_points() / self.R
            self.dL_vec = np.full(self.r.shape, self.dL)
        else:
            if np.max(val) > self.R:
                raise ValueError('You cannot choose r_i > R')
            else:
                # print('You need to manually set dL_vec!')
                self._r = val / self.R

    @property
    def dL_vec(self):
        return self._dL_vec

    @dL_vec.setter
    def dL_vec(self, val):
        self._dL_vec = val

    @property
    def angle(self):
        '''See docstring of _angle_b_trans().'''
        return self._angle_b_trans(self.b_phi, self.b_theta)

    '''@property (ies?...) return normalised and correctly scaled field values.
    Dividing by normalisation at r=0 is done in private classmethods,
    scaling done in property definitions.'''
    @property
    def b_r(self):
        return self.B_0 * self._b_r(self._r, self.theta)

    @property
    def b_phi(self):
        return self.B_0 * self._b_phi(self._r, self.theta)

    @property
    def b_theta(self):
        return self.B_0 * self._b_theta(self._r, self.theta)

    @property
    def b_par(self):
        return self.B_0 * self._b_par(self._r, self.theta)

    @property
    def b_trans(self):
        return self.B_0 * self._b_trans(self._r, self.theta)

    @classmethod
    def _b_par(cls, r, theta):
        return cls._b_r(r, theta)

    @classmethod
    def _b_trans(cls, r, theta):
        b_phi = cls._b_phi(r, theta)
        b_theta = cls._b_theta(r, theta)
        return np.sqrt(b_phi**2 + b_theta**2)

    @staticmethod
    def _angle_b_trans(b_phi, b_theta):
        '''Calculates angle of transversal field component (psi).
        Conforms to psi definition of gammaALPs. See definition of
        Bs, Bt, Bu in GMF environs and trafo.py.
        In this case, B_b = -B_theta, B_l = -B_phi,
        b=galactic latitude, l=galactic longitude.
        To rotate jet axis around LOS, add or subtract some angle.
        Using the value of 1908 paper (PA=147° in galactic coordinates)
        subtract np.radians(147) from resulting angle.
        TODO: Include visualisation somewhere.'''
        return np.arctan2(-b_theta, -b_phi)

    '''Magnetic field expressions, see 1008.5353 and 1908.03084 for details.
    Field values at r=0 are evaluated seperately due the impossibility of
    dividing by zero.
    Can be used for array, as well as floats.
    Carefull when calling from class, as there will be no normalisation done.
    '''
    @classmethod
    def _b_r(cls, r, theta):
        zero_val = - np.cos(theta) * (6 * cls.F_0 + 2 * cls.alpha**5) \
                   / (3 * cls.alpha**2)
        if np.any(np.isclose(r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(r, 0))
                val = 2 * np.cos(theta) * cls._f(r) / r**2
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = 2 * np.cos(theta) * cls._f(r) / r**2
        return val / cls.norm

    @classmethod
    def _b_theta(cls, r, theta):
        zero_val = np.sin(theta) * (6 * cls.F_0 + 2 * cls.alpha**5) \
                   / (3 * cls.alpha**2)
        if np.any(np.isclose(r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(r, 0))
                val = - np.sin(theta) * cls._f_prime(r) / r
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = - np.sin(theta) * cls._f_prime(r) / r
        return val / cls.norm

    @classmethod
    def _b_phi(cls, r, theta):
        zero_val = 0
        if np.any(np.isclose(r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(r, 0))
                val = cls.alpha * np.sin(theta) * cls._f(r) / r
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = cls.alpha * np.sin(theta) * cls._f(r) / r
        return val / cls.norm

    '''Unscaled versions of f, df/dr of referenced paper. Scaling done in
    definitions of field values (properties and classmethods).'''
    @classmethod
    def _f(cls, r):
        # should maybe include special case of r=0 here as well.
        # on the other hand, never used explicitely. same with df/dr
        return cls.alpha * np.cos(cls.alpha * r) - \
               np.sin(cls.alpha * r) / r \
               - cls.F_0 * r**2 / cls.alpha**2

    @classmethod
    def _f_prime(cls, r):
        return (- cls.alpha**2 * np.sin(cls.alpha * r)
                - cls.alpha * np.cos(cls.alpha * r) / r
                + np.sin(cls.alpha * r) / r**2) \
                - 2 * cls.F_0 * r / cls.alpha**2

    def rotation_measure(self, nel):
        '''Rotation measure (RM) = rad * m^-2 * 812 * integral nel * B dz,
        nel in 1/cm^3, B in muGauss, z in kpc.'''
        return 812 * simps(self.b_par * nel, self.r)


if __name__ == "__main__":
    sys.exit()