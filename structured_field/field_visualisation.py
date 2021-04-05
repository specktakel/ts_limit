import numpy as np
import matplotlib.pyplot as plt

# variable definitions

alpha = 5.7634591968
c = 1
theta = np.radians(180+45)


#function definitions

def b_r(r):
    return 2 * np.cos(theta) * f(r) / r**2
    

def b_theta(r):
    return - np.sin(theta) * f_prime(r) / r


def b_phi(r):
    return alpha * np.sin(theta) * f(r) / r


def f(r):
    return c * (alpha * np.cos(alpha * r) - np.sin(alpha * r) / r) \
           - F_0 * r**2 / alpha**2


def f_prime(r):
    return c * ( - alpha**2 * np.sin(alpha * r) \
                - alpha * np.cos(alpha * r) / r \
                + np.sin(alpha * r) / r**2) \
               - 2 * F_0 * r / alpha**2


F_0 = c * (alpha * np.cos(alpha) - np.sin(alpha)) * alpha**2

fig = plt.figure(1)
ax = fig.add_subplot(111)

r = np.linspace(0, 1, num=1000)

ax.plot(r, b_r(r), label='$B_r$')
ax.plot(r, b_phi(r), label='$B_\phi$')
ax.plot(r, b_theta(r), label=r'$B_\theta$')
ax.legend()
fig.savefig('structured_field.png', dpi=300)ich g
