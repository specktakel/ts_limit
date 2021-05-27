#imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.stats import norm
from matplotlib import cm
from math import floor, ceil
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from gammaALPs.nel.icm import NelICM
from scipy.spatial.transform import Rotation as R_
import sys
import argparse
from itertools import product


#set default values
R = 93
B0 = 8.3
theta = 225
phi = 0
num = 11
sphere = 'half'

#parse input arguments
parser = argparse.ArgumentParser(description='Handling data input and fig save options.')
parser.add_argument('--radius', '-r', type=float, help='Cavity radius in kpc. Default is 93kpc.')
parser.add_argument('--b_field', '-b', type=float, help='Central B field strenght in muG.')
parser.add_argument('--theta', '-the', type=float, help='Inclination of symmetry axis to LOS in degree.')
parser.add_argument('--phi', '-phi', type=float,
                    help='Position angle of symmetry axis w.r.t. galactic latitude unit vector in degree.')
parser.add_argument('--pixels', '-p', type=int, help='Number of pixels in both transversal directions.')
parser.add_argument('--sphere', '-sph', type=str, help='"half" or "full" sphere used for integration along LOS.')
parser.add_argument('--save', '-s', type=str, help='Save path for figure, if not given nothing is saved.')
args = parser.parse_args()

#update default values if otherwise specified
if args.radius: R = args.radius
if args.b_field: B0 = args.b_field
if args.theta: theta = args.theta
if args.phi: phi = args.phi
if args.pixels: num = args.pixels
if args.sphere: sphere = args.sphere
if args.save: savepath = args.save
#sys.exit()
#needed for Bfield definitions
alpha = 5.7634591968
F_0 = (alpha * np.cos(alpha) - np.sin(alpha)) * alpha**2
norm_ = np.sqrt((3 * F_0 + alpha**5)**2) * 2 / (3 * alpha**2)


def b_r(r, theta):
    zero_val = - np.cos(theta) * (6 * F_0 + 2 * alpha**5) / (3 * alpha**2)
    if np.isclose(r, 0):
        val = zero_val
    else:
        val = 2 * np.cos(theta) * f(r) / r**2
    if r > 1:
        val = 0
    return 1 / norm_ * val
    

def b_theta(r, theta):
    zero_val = np.sin(theta) * (6 * F_0 + 2 * alpha**5) / (3 * alpha**2)
    if np.isclose(r, 0):
        val = zero_val
    else:
        val = - np.sin(theta) * f_prime(r) / r
    if r > 1:
        val = 0
    return 1 / norm_ * val


def b_phi(r, theta):
    zero_val = 0
    if np.isclose(r, 0):
        val = zero_val
    else:
        val = alpha * np.sin(theta) * f(r) / r
    if r > 1:
        val = 0
    return 1 / norm_ * val


def f(r):
    return (alpha * np.cos(alpha * r) - np.sin(alpha * r) / r) \
           - F_0 * r**2 / alpha**2


def f_prime(r):
    return ( - alpha**2 * np.sin(alpha * r) \
                - alpha * np.cos(alpha * r) / r \
                + np.sin(alpha * r) / r**2) \
               - 2 * F_0 * r / alpha**2 


def B(B0, x1, x2, x3, in_sys, out_sys):
    #get input coordinates
    if in_sys == 'cart':
        r, theta, phi = cart_to_sphere(x1, x2, x3)
    elif in_sys=='sph':
        r, theta, phi = x1, x2, x3
    else:
        print('wrong coords')
    
    #calculate bfield
    B = np.array([b_r(r / R, theta), b_theta(r / R, theta), b_phi(r / R, theta)])
    #print(B.shape)
    #get output coordinates
    if out_sys == 'sph':
        return B * B0
    elif out_sys == 'cart':
        return B0 * np.matmul(trafo(r, theta, phi), B.transpose()).transpose()


def rotation_measure(z, b_par, nel):
    return 812 * simps(b_par * nel, z)


#coordinate stuff
#coordinate transformations
def cart_to_sphere(x, y, z):
    '''Uses theta=0 at north pole, theta=pi/2 at equator'''
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)     # for correct quadrant
    return r, theta, phi


def sphere_to_cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def trafo(r, theta, phi):
    '''Returns trafo used for vector fields, i.e. Field A cartesian = S * Field A spherical. S is returned'''
    mat = np.array([[np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), -np.sin(phi)],
                    [np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.cos(phi)],
                    [np.cos(theta), -np.sin(theta), 0]])
        #outdata[c, :, :] = mat[:, :]
    
    return mat


def axis_helper(x):
    '''Used for aligning pixels of colormap.'''
    l = x.shape[0]
    dx = x[1] - x[0]
    return np.linspace(x[0] - dx / 2, x[-1] + dx / 2, num=l + 1)


def B_rot(B, rotvec):
    '''Rotates vector B around rotvec by norm of rotvec.'''
    return np.matmul(rotvec.as_matrix(), B.transpose()).transpose()


def generate_B_struc(xx, yy, zz, B0, R, theta, phi):
    '''Generate structured field from input coordinate grid, central Bfield B0,
    radius R and orientation theta/phi (both in degrees!).
    phi=0 corresponds to jet axis aligned with unit vector of galactic latitude.
    See thesis for coordinate trafo between this system and gammaALPs.'''
    theta_ = np.radians(theta)
    phi_ = np.radians(phi)
    
    #get rotation
    #incl = R_.from_rotvec(theta_ * np.array([1, 0, 0]))    # tilts symmetry axis around x-axis
    pa = R_.from_rotvec(phi_ * np.array([0, 0, 1]))        # rotates around z axis, position angle phi
    
    incl = R_.from_rotvec(theta_ * np.array([1, 0, 0]))
    incl = R_.from_rotvec(B_rot(incl.as_rotvec(), pa))

    
    #get magnetic field at each point
    B_r = np.zeros((xx.flatten().shape[0], 3))
    c_list = list(zip(xx.flatten(), yy.flatten(), zz.flatten()))
    for c_, (x_, y_, z_) in enumerate(c_list):
        r_, theta_, phi_ = cart_to_sphere(x_, y_, z_)
        if np.isnan(theta_):
            #check for previous and next few entries for r=0 and take their theta
            #print(c_)
            _, theta_, _ = cart_to_sphere(*c_list[c_ + 1])
        b_temp = B_rot(B(B0, r_, theta_, phi_, 'sph', 'cart'), incl)
        B_r[c_] = b_temp #B_rot(b_temp, pa)
    B_r = B_r.reshape(*xx.shape, 3)
    #returns only B_z component of rotated field, as this is the one parallel to the LOS
    return B_r[:, :, :, 2]


def make_map(num, R, B0, theta, phi, sphere):
    '''Put everything together...'''
    int_steps = 100
    # init electron density model
    nel = NelICM(n0 = 39., n2 = 4.05, r_abell = 500., r_core = 80., r_core2 = 280., beta = 1.2, beta2= 0.58, eta = 0.5)
    
    #convert angles to radians
    theta_ = np.radians(theta)
    phi_ = np.radians(phi)
    
    #get rotation
    #incl = R_.from_rotvec(theta_ * np.array([1, 0, 0]))    # tilts symmetry axis around x-axis
    pa = R_.from_rotvec(phi_ * np.array([0, 0, 1]))        # rotates around z axis, position angle phi
    
    incl = R_.from_rotvec(theta_ * np.array([1, 0, 0]))
    incl = R_.from_rotvec(B_rot(incl.as_rotvec(), pa))
    
    #init coordinate space
    dx = 2 * R / num
    x = np.linspace(-R + dx / 2, R - dx / 2, num=num, endpoint=True)
    y = np.linspace(-R + dx / 2, R - dx / 2, num=num, endpoint=True)
    if sphere == 'half':
        z = np.linspace(0, R, num=int_steps, endpoint=True)   # (0, R) for front half sphere, (-R, R) for full sphere, (-R, 0) for back
    elif sphere == 'full':
        z = np.linspace(-R, R, num=2*int_steps, endpoint=True)
    #coordinate meshgrid
    xx, yy, zz = np.meshgrid(x, y, z)
    #print(xx)
    #get magnetic field at each point
    B_r = np.zeros((xx.flatten().shape[0], 3))
    c_list = list(zip(xx.flatten(), yy.flatten(), zz.flatten()))
    for c_, (x_, y_, z_) in enumerate(c_list):
        r_, theta_, phi_ = cart_to_sphere(x_, y_, z_)
        if np.isnan(theta_):
            #check for previous and next few entries for r=0 and take their theta
            #print(c_)
            _, theta_, _ = cart_to_sphere(*c_list[c_ + 1])
        #g = B_cart(r_, theta_, phi_, sys='sph')
        #B_C[c_] = g
        b_temp = B_rot(B(B0, r_, theta_, phi_, 'sph', 'cart'), incl)
        B_r[c_] = b_temp #B_rot(b_temp, pa)
    B_r = B_r.reshape(len(x), len(y), len(z), 3)
    #get rm at each projected point in x-y plane
    rm = np.zeros((x.shape[0],  y.shape[0]))
    rr, _, _ = cart_to_sphere(xx, yy, zz)
    NEL = nel(rr)*1e-3
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            rm[i, j] = rotation_measure(z, B_r[i, j, :, 2], NEL[i, j, :])

    #init figure
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect=1)
    tick_pos = (-100, -50, 0, 50, 100)
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    #for plotting s.t. pixels are properly aligned to the middle
    xh = axis_helper(x)
    yh = axis_helper(y)
    xxh, yyh = np.meshgrid(xh, yh)
    #print(xxh)
    #set up colormap
    cmap = plt.get_cmap('seismic')
    levels = MaxNLocator(nbins=cmap.N).tick_values(-1e4, 1e4)
    norm = BoundaryNorm(levels, cmap.N)
    
    #plot colormap
    cbticks = np.arange(-1e4, +1.2e4, step=2e3, dtype=int)
    pcol = ax.pcolor(xxh, yyh, rm, cmap=cmap, norm=norm, alpha=1)
    cb = fig.colorbar(pcol, ax=ax, extend='both', ticks=cbticks)
    ticklabels = cb.ax.get_yticklabels()
    cb.ax.set_yticklabels(ticklabels, ha='right')
    cb.ax.set_ylabel(r'RM [\SI{}{\radian\per\meter\squared}]', labelpad=10)
    cb.ax.yaxis.set_tick_params(pad=35)
    #cb.ax.yaxis.set_ticks(cbticks, cbticks)
    #text with central RM
    props = dict(boxstyle='round', facecolor='gray', alpha=0.2)
    # place a text box in upper left in axes coords
    # if num is odd, take middle pixel
    # if num is even, take mean value of the 4 pixel surrounding x, y=(0, 0)
    if num % 2 == 0:
        l_ind = int(num / 2 - 1)
        u_ind = int(num / 2)
        print(l_ind, u_ind)
        central_rm = 0
        for p in product((l_ind, u_ind), repeat=2):
            #did i really take 10 minutes to look this up, just to avoid typing 4 lines?
            #yes, i did.
            print(rm[p])
            central_rm += rm[p] / 4
    else:
        ind = floor(num / 2)
        central_rm = rm[ind, ind]
        print(ind)
    textstr = ' '.join((f'central RM: {central_rm:.0f}', r'\SI{}{\radian\per\meter\squared}'))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    if savepath:
        fig.savefig(savepath)
    plt.show()
    return rm


_ = make_map(num, R, B0, theta, phi, sphere)

