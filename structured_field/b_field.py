import numpy as np
import matplotlib.pyplot as plt


class structured_field():
    alpha = 5.7634591968
    F_0 = (alpha * np.cos(alpha) - np.sin(alpha)) * alpha**2

    def __init__(self, B_0, R, theta, radians=False, cell_num=100):
        '''Norm in micro gauss, theta in degrees, if not specified.'''
        if radians:
            self.theta = theta
        else:
            self.theta = np.radians(theta)
        print(self.theta)
        self.B_0 = B_0
        # Normalisation from \lim_{r\to 0}
        self.c = self.B_0 / (np.sqrt((3 * self.F_0 + self.alpha**5)**2) * 2
                             / (3 * self.alpha**2))
        self.R = R
        self.cell_num = cell_num
        self.dL = R / cell_num
        self.r = self._get_r_points()
        # self.get_dL()
        self.dL_vec = np.full(self.r.shape, self.dL)

    def _get_r_points(self):
        return np.linspace(self.dL / 2, self.R - self.dL / 2, self.cell_num)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, val=None):
        if val is None:
            self._r = self._get_r_points()
        else:
            self._r = val

    @property
    def angle(self):
        return self._angle_b_trans(self.b_phi, self.b_theta)

    @property
    def b_r(self):
        return self._b_r(self.r, self.theta)

    @property
    def b_phi(self):
        return self._b_phi(self.r, self.theta)

    @property
    def b_theta(self):
        return self._b_theta(self.r, self.theta)

    @property
    def b_par(self):
        return self._b_par(self.r, self.theta)

    @property
    def b_trans(self):
        return self._b_trans(self.r, self.theta)

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
        '''Calculates angle of transversal field component w.r.t.
        theta direction (arbitrarily chosen). Returns angle in radians.'''
        return np.arctan2(b_phi, b_theta)

    '''Magnetic field expressions, see 1008.5353 and 1908.03084 for details.
    Field values at r=0 are evaluated seperately due the impossibility of
    dividing by zero. Can be used for array, as well as integers.'''
    @classmethod
    def _b_r(cls, r, theta):
        zero_val = - np.cos(theta) * (6 * cls.F_0 + 2 * cls.alpha**5) \
                   / (3 * cls.alpha**2)
        if np.any(np.isclose(r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(r, 0))
                val = 2 * np.cos(theta) * cls._f() / r**2
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = 2 * np.cos(theta) * cls._f(r) / r**2
        return val

    @classmethod
    def _b_theta(cls, r, theta):
        zero_val = np.sin(theta) * (6 * cls.F_0 + 2 * cls.alpha**5) \
                   / (3 * cls.alpha**2)
        if np.any(np.isclose(r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(r, 0))
                val = - np.sin(theta) * cls._f_prime() / r
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = - np.sin(theta) * cls._f_prime(r) / r
        return val

    @classmethod
    def _b_phi(cls, r, theta):
        zero_val = 0
        if np.any(np.isclose(r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(r, 0))
                val = cls.alpha * np.sin(theta) * cls.f(r) / r
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = cls.alpha * np.sin(theta) * cls._f(r) / r
        return val

    @classmethod
    def _f(cls, r):
        return cls.alpha * np.cos(cls.alpha * r) - \
               np.sin(cls.alpha * r) / r \
               - cls.F_0 * r**2 / cls.alpha**2

    @classmethod
    def _f_prime(cls, r):
        return (- cls.alpha**2 * np.sin(cls.alpha * r)
                - cls.alpha * np.cos(cls.alpha * r) / r
                + np.sin(cls.alpha * r) / r**2) \
                - 2 * cls.F_0 * r / cls.alpha**2


b = structured_field(8.3, 1, 225)
plt.plot(b.r, b.c * b.b_r, label=r'$B_r$')
#plt.plot(b.r, b.c * b.b_theta(b.r, b.theta), label=r'$B_\theta$')
#plt.plot(b.r, b.c * b.b_phi(b.r, b.theta), label=r'$B_\phi$')
#plt.plot(b.r, b._angle_b_trans(b._b_phi(b.r, b.theta), b._b_theta(b.r, b.theta)),
#         label=r'$\psi=\frac{B_\phi}{B_\theta}$')
#plt.plot(b.r, b.c * b._b_trans(b.r, b.theta), label=r'$B_{\text{trans}}$')
# plt.plot(b.c * b._b_theta(b.r, b.theta), b.c * b._b_phi(b.r, b.theta))
plt.legend()
