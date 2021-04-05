import numpy as np
import matplotlib.pyplot as plt

class structured_field():
    alpha = 5.7634591968
    F_0 = (alpha * np.cos(alpha) - np.sin(alpha)) * alpha**2
    # print(F_0)
    def __init__(self, B_0, R, theta, radians=False, cell_num=100):
        
        '''Norm in micro gauss, theta in degrees, if not specified.'''
        if radians:
            self.theta = theta
        else:
            self.theta = np.radians(theta)
        print(self.theta)
        self.B_0 = B_0
        self.c = self.B_0 / (np.sqrt((3 * self.F_0 + self.alpha**5)**2) * 2 \
            / (3 * self.alpha**2))    # from r=0, sqrt(\vec{B}**2)
        print(self.c)
        # self.F_0 = (alpha * np.cos(alpha) - np.sin(alpha)) * alpha**2
        self.R = R
        self.cell_num = cell_num
        self.dL = R / cell_num
        self.get_r_points()
        # self.get_dL()
        self.dL_vec = np.full(self.r.shape, self.dL)

    def get_r_points(self):
        self.r = np.linspace(self.dL / 2, self.R - self.dL / 2, self.cell_num)

    def get_dL(self):
        self.dL = self.r[1:] - self.r[:-1]

    #@property
    #def r(self):
    #    return self._r


    #@r.setter
    #def r(self, vals):
        


    def calc_b(self):
        pass


    #@staticmethod
    def b_r(self):
        zero_val = - np.cos(self.theta) * (6 * self.F_0 + 2 * self.alpha**5) \
                   / (3 * self.alpha**2)
        if np.any(np.isclose(self.r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(self.r, 0))
                val = 2 * np.cos(self.theta) * self.f() / self.r**2
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = 2 * np.cos(self.theta) * self.f() / self.r**2
        return val
       
 
    # @staticmethod
    def b_theta(self):
        zero_val = np.sin(self.theta) * (6 * self.F_0 + 2 * self.alpha**5) / (3 * self.alpha**2)
        if np.any(np.isclose(self.r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(self.r, 0))
                val = - np.sin(self.theta) * self.f_prime() / self.r
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = - np.sin(self.theta) * self.f_prime() / self.r
        return val


    # @staticmethod
    def b_phi(self):
        zero_val = 0
        if np.any(np.isclose(self.r, 0)):
            try:
                zero_args = np.argwhere(np.isclose(self.r, 0))
                val =  self.alpha * np.sin(self.theta) * self.f() / self.r
                val[zero_args] = zero_val
            except TypeError:
                val = zero_val
        else:
            val = self.alpha * np.sin(self.theta) * self.f() / self.r
        return val

    # @staticmethod
    def f(self):
        return self.alpha * np.cos(self.alpha * self.r) - \
               np.sin(self.alpha * self.r) / self.r \
               - self.F_0 * self.r**2 / self.alpha**2


    # @staticmethod
    def f_prime(self):
        return ( - self.alpha**2 * np.sin(self.alpha * self.r) \
                    - self.alpha * np.cos(self.alpha * self.r) / self.r \
                    + np.sin(self.alpha * self.r) / self.r**2) \
                   - 2 * self.F_0 * self.r / self.alpha**2

b = structured_field(8.3, 1, 225)

