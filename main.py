###############################
##        Parameters         ##
###############################

ds = 1/64; # Spatial step length
dt = 0.1 * ds; # Temporal step length

# Spatial boundary values
min_x, max_x = -0.5, 0.5
min_y, max_y = -0.5, 0.5
min_z, max_z = -0.5, 0.5

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
  pair.old = array
  pair.new = array
  return pair


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

def mid(pair):
  return FieldPair(None, pair).mid()

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
      ni = nfield.component(i)
      nj = nfield.component(j)

      term_b += np.sum(diff(ni, j, '-'))**2
      term_c += np.sum(nj * diff(ni, j))

    term_a += np.sum(diff(ni, i))

  term_a = term_a**2
  term_c = term_c**2

  return 0.5 * (C1 * term_a + C2 * term_b + C3 * term_c)


###############################
##          Solvers          ##
###############################

def w_solver(wfield_old, nfield_pair):
  dw = np.cross(f_star(nfield_pair), nfield_pair.mid())
  return wfield_old + dt * dw
  
def n_solver(nfield_old, wfield_old):
  return None

# D_t n^{m,v} = n^{m+1/2,v} x w^{m+1/2,v}

# Things for meeting:
# - Implement naive solver for n (needs the V matrix from section 4)
# - Implement iterative solver

###############################
##   Field Initialization    ##
###############################

# Initialize director and angular momentum fields
field_shape = space_sizes + (dim,) # Shape of all relevant vector fields
nfield = FieldPair(field_shape)
wfield = FieldPair(field_shape)

# Set initial values (TODO)
# nfield[:,:,:,0,1] = 
# nfield[:,:,:,1,1] = 
# nfield[:,:,:,2,1] = 
# wfield[:,:,:,0,1] = 
# wfield[:,:,:,1,1] = 
# wfield[:,:,:,2,1] = 

start_time = time.time()
wfield.update(w_solver(wfield.new, nfield))
print("--- %s seconds ---" % (time.time() - start_time))