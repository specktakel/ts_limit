import numpy as np
np.seterr(divide='ignore', invalid='ignore')
alpha = 5.7634591968
c = 1
theta = np.radians(180+45)
F_0 = c * (alpha * np.cos(alpha) - np.sin(alpha)) * alpha**2


def f(r):
    return c * (alpha * np.cos(alpha * r) - np.sin(alpha * r) / r) \
           - F_0 * r**2 / alpha**2


def f_prime(r):
    return c * ( - alpha**2 * np.sin(alpha * r) \
                - alpha * np.cos(alpha * r) / r \
                + np.sin(alpha * r) / r**2) \
               - 2 * F_0 * r / alpha**2


def b_2(r):
    zero_val = np.sin(theta) * (6 * F_0 + 2 * alpha**5 * c) / (3 * alpha**2)
    val = - np.sin(theta) * f_prime(r) / r
    try:
        val[np.isclose(r, 0)] = zero_val
    except TypeError:
        val = zero_val
    return val

print(b_2(0))
