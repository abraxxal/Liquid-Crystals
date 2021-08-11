# TODO (main.py):
# - Compute error (stopping condition in iterative_solver) properly
# - Write frame data to a file instead of directly passing it to graphics
# - Potentially add a damping factor and electric field

# TODO (Graphics.py):
# - Make the user compute positions so that we can use ds
# - Do more matrix computations on the GPU (model can stay on CPU though)
# - Use green to indicate angular momentum magnitude, maybe make the blue colors more prevalent as well?
# - Add more advanced time control features, like slowdown, pause, play, as well as a time-controlled render loop
# - See if Python has any easy libraries for displaying text with OpenGL (for on-screen data)
# - Maybe add a light source for fun?

###############################
##        Parameters         ##
###############################

ds = 1/64; # Spatial step length
dt = 0.1 * ds; # Temporal step length
tolerance = 0.1 * ds**2 # Error tolerance for iterative solver

# Spatial boundary values
min_x, max_x = -0.5, 0.5
min_y, max_y = -0.5, 0.5
min_z, max_z = -0, 0

dim = 3 # Spatial dimension (should be either 2 or 3)

final_time = 1.0 # Simulation time

# Frank elastic constants
K1 = 0.5; # Splay
K2 = 1.0; # Twist
K3 = 0.5; # Bend


###############################
##          Imports          ##
###############################

import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import animation
import time
from Graphics import *

###############################
##     Indexing Helpers      ##
###############################

# Returns a multi-index in which all but one axis is free. For example, axial_index(3, 54) = (:,:,:,54).
def axial_index(axis, index):
  return (np.s_[:],) * axis + (index,)

# See axial_index.
def axial_array(array, axis, index):
  return array[axial_index(axis, index)]


###############################
##     OldNewPair Class      ##
###############################

# The OldNewPair class represents a pair of two NumPy arrays of the same shape, one representing an older value,
# and the other representing a newer value. 
class OldNewPair:
  # Initializes a pair to be all zeros, unless pair is not None, in which case the pair is initialized to the
  # given value.
  def __init__(self, shape, pair=None):
    if pair is None:
      self.pair = np.zeros(shape + (2,))
      self.__pair_axis = len(shape)
    else:
      self.pair = pair
      self.__pair_axis = len(np.shape(pair)) - 1

    self.__old_idx = axial_index(self.__pair_axis, 0)
    self.__new_idx = axial_index(self.__pair_axis, 1)

  # Returns a pair of the i'th component of old and new.
  def component(self, i):
    axis = self.__pair_axis - 1 # Dimension index should always come right before pairing index
    return axial_array(self.pair, axis, i)

  # Sets "old" to "new", and "new" to "newer".
  def update(self, newer):
    self.old = self.new
    self.new = newer

  # Returns the arithmetic mean of the old and new values.
  def mid(self):
    return (self.old + self.new) / 2

  @property
  def old(self):
    return self.__old

  @old.getter
  def old(self):
    return self.pair[self.__old_idx]

  @old.setter
  def old(self, old):
    self.pair[self.__old_idx] = old

  @property
  def new(self):
    return self.__new

  @new.getter
  def new(self):
    return self.pair[self.__new_idx]

  @new.setter
  def new(self, new):
    self.pair[self.__new_idx] = new

# Returns an OldNewPair for which the old and new values are the same.
def make_constant_pair(array):
  pair = OldNewPair(np.shape(array))
  pair.old = array.copy()
  pair.new = array.copy()
  return pair

# Returns the i'th component of a normal NumPy array (instead of an OldNewPair).
def component(array, i):
  axis = len(np.shape(array)) - 1 # Dimension index should be the last index
  return axial_array(array, axis, i)

# Returns the arithmetic mean of a pair of values which haven't been wrapped into an OldNewPair instance
def mid(pair):
  return OldNewPair(None, pair).mid()

###############################
##   Domain Initialization   ##
###############################

# Constants from the paper
C1 = K1 - K2
C2 = K2
C3 = K3 - K2

# Discrete time axis
t_axis = np.arange(0, final_time + dt, dt)
num_t, = np.shape(t_axis)
time_indices = np.arange(num_t)

# Discrete space axes
x_axis = np.arange(min_x, max_x + ds, ds)
y_axis = np.arange(min_y, max_y + ds, ds)
z_axis = np.arange(min_z, max_z + ds, ds)

# Axes, their associated indices, and the number of elements in each axis
space_axes = (x_axis, y_axis, z_axis)
space_indices = (np.arange(len(x_axis)), np.arange(len(y_axis)), np.arange(len(z_axis)))
space_sizes = (len(x_axis), len(y_axis), len(z_axis))


###############################
##    Discrete Operators     ##
###############################

# Treating 'array' as a function of its index domain, returns the partial derivative of 'array' along 'axis'.
# Possible values for 'kind' are 'C' (central derivative), '+' (forwards derivative), and '-' (backwards derivative).
def diff(axis, array, kind='C'):
  denom_mult = 2 if kind == 'C' else 1
  left = axial_array(array, axis, (space_indices[axis] + (0 if kind == '-' else 1)) % space_sizes[axis])
  right = axial_array(array, axis, (space_indices[axis] - (0 if kind == '+' else 1)) % space_sizes[axis])

  return (left - right) / (denom_mult * ds)

# Given an OldNewPair of vector fields, returns the value F^*(n) from the paper.
def f_star(nfield_pair):
  result = np.zeros(space_sizes + (dim,))
  
  for i in range(0, dim):
    sum = np.zeros(space_sizes)

    for j in range(0, dim):
      for k in range(0, dim):
        ni = nfield_pair.component(i)
        nj = nfield_pair.component(j)
        nk = nfield_pair.component(k)

        term_a = diff(i, diff(j, mid(nj)))
        term_b = diff(j, diff(j, mid(ni), '-'), '+')
        term_c = diff(j, mid(nj) * mid(nk * diff(k, ni)))
        term_d = mid(nj * diff(j, nk)) * diff(i, mid(nk))

        sum += C1 * term_a + C2 * term_b + C3 * (term_c - term_d)

    result[:,:,:,i] = sum
  
  return result

# Given a vectof field, returns the Frank-Oseen energy density field.
def energy(nfield):
  term_a, term_b, term_c = 0, 0, 0

  for i in range(0, dim):
    for j in range(0, dim):
      ni = component(nfield, i)
      nj = component(nfield, j)

      term_b += diff(j, ni, '-')**2
      term_c += nj * diff(j, ni)

    term_a += diff(i, ni)

  term_a = term_a**2
  term_c = term_c**2

  energy_field = 0.5 * (C1 * term_a + C2 * term_b + C3 * term_c)
  return np.sum(energy_field)

###############################
##          Solvers          ##
###############################

# wfield_pair is (w^m, w^{m,s})
# nfield_old is n^m
# returns n^{m,s+1}
def n_solver(nfield_old, wfield_pair):
  def cross_matrix(w):
    return np.array([[np.zeros(space_sizes), component(w, 2), -component(w, 1)], 
                     [-component(w, 2), np.zeros(space_sizes), component(w, 0)], 
                     [component(w, 1), -component(w, 0), np.zeros(space_sizes)]])

  def v_matrix(w):
    a = (dt / 2)**2 * np.einsum("xyzi,xyzi->xyz", w, w) # dt^2/4 * |w|^2

    term1 = np.einsum("xyz,ij->xyzij", 1 - a, np.eye(dim, dim)) # (1 - a) * I
    term2 = (dt**2 / 2) * np.einsum('xyzi,xyzj->xyzij', w, w) # (dt^2 / 2) * (w \otimes w)
    term3 = np.einsum(",ijxyz->xyzij", dt, cross_matrix(w)) # dt * Q(w)

    return np.einsum("xyz,xyzij->xyzij", 1/(1 + a), term1 + term2 + term3) # 1/(1 + a) * sum

  # n^{m,s+1} = V((w^{m,s} + w^m)/2) * n^m
  return np.einsum("xyzij,xyzj->xyzi", v_matrix(wfield_pair.mid()), nfield_old)

# wfield_old is w^m
# nfield_pair is (n^m, n^{m,s+1})
# returns w^{m,s+1}
def w_solver(wfield_old, nfield_pair):
  dw = np.cross(f_star(nfield_pair), nfield_pair.mid())
  # (w^{m,s+1} - w^m)/dt = F*(n^m, n^{m,s+1}) x (n^m + n^{m,s+1})/2
  return wfield_old + dt * dw

# nfield_pair is n^m
# wfield_pair is w^m
# return n^{m+1} and w^{m+1}
def iterative_solver(nfield_old, wfield_old):
  def error(nfield_pair):
    return energy(nfield_pair.new - nfield_pair.old)

  iterations = 0

  # Initial values (w^{m,0} = w^m and n^{m,0} = n^m)
  nfield_pair = make_constant_pair(nfield_old)
  wfield_pair = make_constant_pair(wfield_old)
  
  while True:
    nfield_pair.update(n_solver(nfield_pair.old, wfield_pair)) # Update (n^m, n^{m,s}) to (n^m, n^{m,s+1})
    wfield_pair.update(w_solver(wfield_pair.old, nfield_pair)) # Update (w^m, w^{m,s}) to (w^m, w^{m,s+1})

    err = error(nfield_pair)

    # If too many iterations have occurred or error is low enough, stop iterating and return.
    if iterations >= 400 or err <= 10 and (not iterations <= 1):
      break

    iterations += 1

  return nfield_pair.new, wfield_pair.new
  
###############################
##   Field Initialization    ##
###############################

# Initialize director and angular momentum fields
field_shape = space_sizes + (dim,) # Shape of all relevant vector fields
nfield_initial = np.zeros(field_shape)
wfield_initial = np.zeros(field_shape)

# Set initial values of director field
def af(r):
  return (1 - 2*r)**4

def rf(x,y):
  return np.sqrt(x**2 + y**2)

def denominatorf(x,y):
  return rf(x,y)**2 + af(rf(x,y))**2

x, y, z = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')

nfield_initial[:,:,:,0] = 2 * (rf(x,y) <= 0.5) * x * af(rf(x,y)) / denominatorf(x,y)
nfield_initial[:,:,:,1] = 2 * (rf(x,y) <= 0.5) * y * af(rf(x,y)) / denominatorf(x,y)
nfield_initial[:,:,:,2] = (rf(x,y) <= 0.5) * (af(rf(x,y))**2 - rf(x,y)**2) / denominatorf(x,y) - (rf(x,y) > 0.5)


###############################
##     Main Computation      ##
###############################

start_time = time.time()

frames = [(nfield_initial, wfield_initial)]
for i in range(0, num_t):
  print("Computing frame %i..." % i)
  frames.append(iterative_solver(frames[i][0], frames[i][1]))

print("Took %.2f seconds to compute %i frames. Launching graphics..." % (time.time() - start_time, num_t - 1))


###############################
##         Animation         ##
###############################

current_time = float(0)

graphics = Graphics(800, 800, "Time: %.2f seconds" % current_time)
graphics.start_rendering(frames[0])

i = 0
while graphics.window_is_open():
  i += 1
  current_time = (i % len(frames)) * dt
  graphics.set_window_title("Time: %.2f seconds" % current_time)
  
  graphics.set_render_data(frames[i % len(frames)])
  graphics.render()
  time.sleep(0.01)

graphics.stop_rendering()
