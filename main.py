# TODO:
# - Main program loop (repeatedly running iterative solver)
# - Set initial field values
# - Error calculation
# - Animation
# - Documentation

###############################
##        Parameters         ##
###############################

ds = 1/64; # Spatial step length
dt = 0.1 * ds; # Temporal step length
tolerance = 0.1 * ds**2 # Error tolerance for iterative solver

# Spatial boundary values
min_x, max_x = -0.5, 0.5
min_y, max_y = -0.5, 0.5
min_z, max_z = -ds, ds

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
import time


###############################
##     Indexing Helpers      ##
###############################

def axial_index(axis, index):
  return (np.s_[:],) * axis + (index,)

def axial_array(array, axis, index):
  return array[axial_index(axis, index)]


###############################
##      FieldPair Class      ##
###############################

class FieldPair:
  def __init__(self, shape, pair=None):
    if pair is None:
      self.pair = np.zeros(shape + (2,))
      self.__pair_axis = len(shape)
    else:
      self.pair = pair
      self.__pair_axis = len(np.shape(pair)) - 1

    self.__old_idx = axial_index(self.__pair_axis, 0)
    self.__new_idx = axial_index(self.__pair_axis, 1)

  def component(self, i):
    axis = self.__pair_axis - 1 # Dimension index should always come right before pairing index
    return axial_array(self.pair, axis, i)

  def update(self, new_new):
    self.old = self.new
    self.new = new_new

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

def make_constant_pair(array):
  pair = FieldPair(np.shape(array))
  pair.old = array.copy()
  pair.new = array.copy()
  return pair

# Unpaired component
def component(array, i):
  axis = len(np.shape(array)) - 1 # Dimension index should be the last index
  return axial_array(array, axis, i)

# Unpaired mid
def mid(pair):
  return FieldPair(None, pair).mid()

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

def diff(func, axis, kind='C'):
  denom_mult = 2 if kind == 'C' else 1
  left = axial_array(func, axis, (space_indices[axis] + (0 if kind == '-' else 1)) % space_sizes[axis])
  right = axial_array(func, axis, (space_indices[axis] - (0 if kind == '+' else 1)) % space_sizes[axis])

  return (left - right) / (denom_mult * ds)

def f_star(nfield):
  result = np.zeros(space_sizes + (dim,))
  
  for i in range(0, dim):
    sum = np.zeros(space_sizes)

    for j in range(0, dim):
      for k in range(0, dim):
        ni = nfield.component(i)
        nj = nfield.component(j)
        nk = nfield.component(k)

        term_a = diff(diff(mid(nj), j), i)
        term_b = diff(diff(mid(ni), j, '-'), j, '+')
        term_c = diff(mid(nj) * mid(nk * diff(ni, k)), j)
        term_d = mid(nj * diff(nk, j)) * diff(mid(nk), i)

        sum += C1 * term_a + C2 * term_b + C3 * (term_c - term_d)

    result[:,:,:,i] = sum
  
  return result

def energy(nfield):
  term_a, term_b, term_c = (0, 0, 0)

  for i in range(0, dim):
    for j in range(0, dim):
      ni = component(nfield, i)
      nj = component(nfield, j)

      term_b += np.sum(diff(ni, j, '-'))**2
      term_c += np.sum(nj * diff(ni, j))

    term_a += np.sum(diff(ni, i))

  term_a = term_a**2
  term_c = term_c**2

  return 0.5 * (C1 * term_a + C2 * term_b + C3 * term_c)


###############################
##          Solvers          ##
###############################

# wfield_pair is (w^m, w^{m,s})
# nfield_old is n^m
# returns n^{m,s+1}
def n_solver(nfield_old, wfield_pair):
  def q_matrix(w):
    return np.array([[np.zeros(space_sizes), component(w, 2), -component(w, 1)], 
                     [-component(w, 2), np.zeros(space_sizes), component(w, 0)], 
                     [component(w, 1), -component(w, 0), np.zeros(space_sizes)]])

  def v_matrix(w):
    a = (dt / 2)**2 * np.einsum("xyzi,xyzi->xyz", w, w) # dt^2/4 * |w|^2
    identity_mat = np.zeros(space_sizes + (dim, dim))
    identity_mat[:,:,:] = np.eye(dim, dim)

    term1 = np.einsum("xyz,xyzij->xyzij", 1 - a, identity_mat) # (1 - a) * I
    term2 = (dt**2 / 2) * np.einsum('xyzi,xyzj->xyzij', w, w) # (dt^2 / 2) * (w \otimes w)
    term3 = np.einsum(",ijxyz->xyzij", dt, q_matrix(w)) # dt * Q(w)

    return np.einsum("xyz,xyzij->xyzij", 1/(1 + a), term1 + term2 + term3) # 1/(1 + a) * sum

  # n^{m,s+1} = V(wfield_pair.mid()) * n^m
  return np.einsum("xyzij,xyzj->xyzi", v_matrix(wfield_pair.mid()), nfield_old)

# wfield_old is w^m
# nfield_pair is (n^m, n^{m,s+1})
# returns w^{m,s+1}
def w_solver(wfield_old, nfield_pair):
  dw = np.cross(f_star(nfield_pair), nfield_pair.mid())
  return wfield_old + dt * dw

# nfield_pair is n^m
# wfield_pair is w^m
# return n^{m+1} and w^{m+1}
def iterative_solver(nfield_old, wfield_old):
  def error(nfield_pair):
    return energy(nfield_pair.new) - energy(nfield_pair.old)

  iterations = 0

  nfield_pair = make_constant_pair(nfield_old)
  wfield_pair = make_constant_pair(wfield_old)
  
  while True:
    nfield_pair.update(n_solver(nfield_pair.old, wfield_pair)) # Update (n^m, n^{m,s}) to (n^m, n^{m,s+1})
    wfield_pair.update(w_solver(wfield_pair.old, nfield_pair)) # Update (w^m, w^{m,s}) to (w^m, w^{m,s+1})

    err = error(nfield_pair)

    import os
    os.system("clear")
    print("Iteration: " + str(iterations))
    print("Error: " + str(err) + "\n\n")

    iterations += 1

    if iterations >= 400 or abs(err) <= tolerance**2 and (not iterations == 1):
      print("Done! Error is " + str((abs(err) / (tolerance**2)) * 100) + "% of the tolerance, " + str(tolerance**2))
      break

  return nfield_pair.new, wfield_pair.new
  
###############################
##   Field Initialization    ##
###############################

# Initialize director and angular momentum fields
field_shape = space_sizes + (dim,) # Shape of all relevant vector fields
nfield_initial = np.zeros(field_shape)
wfield_initial = np.zeros(field_shape)

start_time = time.time()
iterative_solver(nfield_initial, wfield_initial)
print("--- %s seconds ---" % (time.time() - start_time))