#
# Authors: Jennifer Jin, Nathan Glover
#

###############################
##    Default Parameters     ##
###############################

# Discretization values
ds = 1/64; # Spatial step length
tolerance = 1E-20 * ds**3 # Error tolerance for iterative solver
alpha = 0.0 # Damping factor (coefficient of D_t n)

# Boundary conditions, either 'P' for "Periodic", 'N' for "Neumann", or 'D' for "Dirichlet"
# Note that Neumann only works when K1 = K2 due to some unimplemented boundary behavior (namely, we need div(n) = 0 on 
# the boundary for Neumann to properly work out)
boundary_behavior = 'P'

# Spatial boundary values
min_x, max_x = -0.5, 0.5
min_y, max_y = -0.5, 0.5
min_z, max_z = -0, 0

final_time = 1.0 # Simulation time

# Frank elastic constants
K1 = 0.5; # Splay
K2 = 1.0; # Twist
K3 = 0.5; # Bend

# Whether or not to allow central derivatives in certain expressions (terms in F* and energy)
a_term_central = True
c_term_central = True

###############################
##    Imports and Logging    ##
###############################

# Import NumPy for all batch computing operations
import numpy as np

# Import os.path for reading and writing pre-computed frame data
import os.path

# Import tqdm for nice command line progress bars, in this file for computing frame data
from tqdm import trange

# Import sys, types, and argparse for processing command line arguments
import sys
import types
import argparse

verbose = False # Verbose mode prints more computation data as the simulation is running.

###############################
##     Indexing Helpers      ##
###############################

# Returns a multi-index in which all but one axis is free. For example, axial_index(3, 54) = (:,:,:,54).
def axial_index(axis, index):
  return (np.s_[:],) * axis + (index,)

# See axial_index.
def axial_array(array, axis, index):
  return array[axial_index(axis, index)]

# Given a potentially out-of-bounds array of indices, returns a new in-bounds array of indices. The behavior of this
# function depends on the boundary conditions. Periodic conditions wrap out-of-bounds indices back around to smaller
# index values, as if indices were on a circle, whereas Neumann conditions clamp out-of-bounds indices to the bounds.
def boundary_handler(index, array_length):
  if boundary_behavior == 'P':
    # If periodic, wrap around
    return index % array_length
  else:
    # If Neumann or other, clamp to range
    def clamp(x):
      return min(max(x, 0), array_length - 1)

    return np.vectorize(clamp)(index)


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

dim = 3 # Simulation is always run in 3 dimensions, even if z-axis is a singleton

def init_computation_domain(input_dt = None):
  global dt, C1, C2, C3, t_axis, num_t, time_indices, x_axis, y_axis, z_axis, space_axes, space_indices, space_sizes

  if input_dt is None:
    dt = 0.1 * ds
  else:
    dt = input_dt

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
  left = axial_array(array, axis, boundary_handler(space_indices[axis] + (0 if kind == '-' else 1), space_sizes[axis]))
  right = axial_array(array, axis, boundary_handler(space_indices[axis] - (0 if kind == '+' else 1), space_sizes[axis]))

  return (left - right) / (denom_mult * ds)

# Returns the discrete Jacobian matrix (with specified direction; central, forwards, or backwards) of a vector field
def grad(field, kind='C'):
  result = np.zeros(space_sizes + (dim, dim))
  
  for i in range(dim):
    for j in range(dim):
      result[:,:,:,i,j] = diff(i, component(field, j), kind)

  return result

# Given an OldNewPair of vector fields, returns the value F^*(n) from the paper.
def f_star(nfield_pair):
  result = np.zeros(space_sizes + (dim,))
  
  for i in range(dim):
    sum = np.zeros(space_sizes)

    for j in range(dim):
      for k in range(dim):
        ni = nfield_pair.component(i)
        nj = nfield_pair.component(j)
        nk = nfield_pair.component(k)

        if a_term_central:
          term_a = diff(i, diff(j, mid(nj)))
        else:
          term_a = diff(i, diff(j, mid(nj), '-'), '+')

        term_b = diff(j, diff(j, mid(ni), '-'), '+')
        
        if c_term_central:
          term_c = diff(j, mid(nj) * mid(nk * diff(k, ni)))
          term_c -= mid(nj * diff(j, nk)) * diff(i, mid(nk))
        else:
          term_c = diff(j, mid(nj) * mid(nk * diff(k, ni, '-')), '+')
          term_c -= mid(nj * diff(j, nk, '-')) * diff(i, mid(nk), '-')

        sum += C1 * term_a + C2 * term_b + C3 * term_c

    result[:,:,:,i] = sum
  
  return result

# Given a vector field, returns the corresponding Frank-Oseen energy density field.
def energy(nfield, wfield):
  term_a, term_b, term_c = 0, 0, 0

  for i in range(dim):
    for j in range(dim):
      ni = component(nfield, i)
      nj = component(nfield, j)

      term_b += diff(j, ni, '-')**2

      if c_term_central:
        term_c += nj * diff(j, ni)
      else:
        term_c += nj * diff(j, ni, '-')

    if a_term_central:
      term_a += diff(i, ni)
    else:
      term_a += diff(i, ni, '-')

  term_a = term_a**2
  term_c = term_c**2

  energy_field = 0.5 * (C1 * term_a + C2 * term_b + C3 * term_c)
  n_energy = np.einsum("xyz->", energy_field)
  return n_energy + 0.5 * np.einsum("xyzi,xyzi->", wfield, wfield)

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
  c = 1/(1/dt + alpha/2)
  # w^{m,s+1} = w^m + c F*(n^m, n^{m,s+1}) x (n^m + n^{m,s+1})/2
  return wfield_old + c * np.cross(f_star(nfield_pair), nfield_pair.mid())

# nfield_pair is n^m
# wfield_pair is w^m
# return n^{m+1} and w^{m+1}
def iterative_solver(nfield_old, wfield_old):
  # Error from the paper
  def error(w_difference, n_difference):
    gradient = grad(n_difference)
    return np.einsum("xyzi,xyzi->", w_difference, w_difference) + np.einsum("xyzij,xyzij->", gradient, gradient)

  iterations = 0

  # Initial values (w^{m,0} = w^m and n^{m,0} = n^m)
  nfield_pair = make_constant_pair(nfield_old)
  wfield_pair = make_constant_pair(wfield_old)
  
  while True:
    nfield_s = nfield_pair.new.copy()
    wfield_s = wfield_pair.new.copy()

    nfield_pair.new = n_solver(nfield_pair.old, wfield_pair) # Update (n^m, n^{m,s}) to (n^m, n^{m,s+1})
    wfield_pair.new = w_solver(wfield_pair.old, nfield_pair) # Update (w^m, w^{m,s}) to (w^m, w^{m,s+1})

    err = error(wfield_pair.new - wfield_s, nfield_pair.new - nfield_s)

    # If too many iterations occur, there's a problem; notify the user that the simulation is failing
    if iterations >= 400:
      if verbose:
        print("Too many iterations; no convergence. Final error: " + str(err))
      break

    # If at leats 2 iterations have run and the error is low enough, return
    if err <= tolerance and (not iterations <= 2):
      break

    iterations += 1

  return iterations, nfield_pair.new, wfield_pair.new

###############################
##   Simulation and Output   ##
###############################

def compute_simulation_frames(output_vfd_filepath, initial_field=None):
  # Initialize director and angular momentum fields
  field_shape = space_sizes + (dim,) # Shape of all relevant vector fields
  nfield_initial = np.zeros(field_shape)
  wfield_initial = np.zeros(field_shape)

  x, y, z = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')

  # Set default initial values of director field
  if initial_field is None:
    def af(r):
      return (1 - 2*r)**4

    def rf(x,y):
      return np.sqrt(x**2 + y**2)

    def denominatorf(x,y):
      return rf(x,y)**2 + af(rf(x,y))**2

    def func(x, y, z):
      return (2 * (rf(x,y) <= 0.5) * x * af(rf(x,y)) / denominatorf(x,y),
              2 * (rf(x,y) <= 0.5) * y * af(rf(x,y)) / denominatorf(x,y),
              (rf(x,y) <= 0.5) * (af(rf(x,y))**2 - rf(x,y)**2) / denominatorf(x,y) - (rf(x,y) > 0.5))
    
    initial_field = func

  nfield_initial[:,:,:,0], nfield_initial[:,:,:,1], nfield_initial[:,:,:,2] = initial_field(x, y, z)

  # Make sure that the field is normalized everywhere
  inorm = 1 / np.sqrt(np.einsum("xyzi,xyzi->xyz", nfield_initial, nfield_initial))
  nfield_initial = np.einsum("xyz,xyzi->xyzi", inorm, nfield_initial)

  # Begin computing frames and writing them to output file, one-by-one to avoid excessive memory usage
  file = open(output_vfd_filepath, "w")

  # First write the positions, as per the file specification
  positions = []
  num_x, num_y, num_z = space_sizes
  for i in range(num_x):
    for j in range(num_y):
      for k in range(num_z):
        positions.extend([x_axis[i], y_axis[j], z_axis[k]])

  file.write(str(positions)[1:-1].replace(',', ''))

  # Next loop through all timesteps, computing and writing one frame per iteration
  nfield, wfield = nfield_initial, wfield_initial
  energy_initial = energy(nfield, wfield)

  if verbose:
    print("Writing to %s..." % output_vfd_filepath)

  total_abs_energy_diff = 0
  total_energy_diff = 0
  total_iterations = 0
  with trange(num_t + 1) as progress_bar:
    progress_bar.set_description("Computing frames")
    for f in progress_bar:
      # Write a new line to separate old frame from this one
      file.write('\n')
      frame_data = []

      # Add each point in space, and the field's direction/momentum at each point, to the frame data
      for i in range(num_x):
        for j in range(num_y):
          for k in range(num_z):
            n = nfield[i,j,k]
            w = wfield[i,j,k]
            frame_data.extend([n[0], n[1], n[2], w[0], w[1], w[2]])
      
      # Write the finished frame data to the file (remove commas and braces)
      file.write(str(frame_data)[1:-1].replace(',', ''))
      
      # If not on the last frame, compute another frame
      if f < num_t:
        energy_old = energy(nfield, wfield)

        # New frame
        iterations, nfield, wfield = iterative_solver(nfield, wfield)

        if boundary_behavior == 'D':
          nfield[0,:,:,0] = nfield_initial[0,:,:,0]
          nfield[num_x - 1,:,:,0] = nfield_initial[num_x - 1,:,:,0]
          
          nfield[:,0,:,1] = nfield_initial[:,0,:,1]
          nfield[:,num_y - 1,:,1] = nfield_initial[:,num_y - 1,:,1]
          
          nfield[:,:,0,2] = nfield_initial[:,:,0,2]
          nfield[:,:,num_z - 1,2] = nfield_initial[:,:,num_z - 1,2]

        # Compute some per-frame info, such as number of iterative solver iterations and energy difference between
        # this frame and the old one
        energy_new = energy(nfield, wfield)
        energy_difference = energy_new - energy_old
        total_energy_diff += energy_difference

        progress_bar.set_postfix(iters=iterations, ediff=energy_difference, total_ediff=total_energy_diff)

        # Keep track of total energy/iterations for post-compute averaging (see prints below)
        total_abs_energy_diff += abs(energy_difference)
        total_iterations += iterations

      
  if verbose:
    total_energy_diff = energy_new - energy_initial
    print("Done computing. Printing performance info...")
    print("Initial and final energy: %.2f --> %.2f" % (energy_initial, energy_new))
    print("Net energy change: %.2f" % total_energy_diff)
    print("Percent energy change: %.3f" % (total_energy_diff / energy_initial * 100))

    print("Absolute energy variation: %.2f" % total_abs_energy_diff)
    print("Percent absolute energy variation: %.3f" % (total_abs_energy_diff / energy_initial * 100))

    print("Average energy difference per frame: %.2f" % (total_energy_diff / num_t))
    print("Average absolute energy difference per frame: %.2f" % (total_abs_energy_diff / num_t))
    print("Average solver iterations per frame: %.2f" % (total_iterations / num_t))
  
  file.close()

###############################
##   Handle Terminal Inputs  ##
###############################

# Create a CLI parser object and return the user-specified command line arguments
def create_parser():
  parser = argparse.ArgumentParser(description="Simulate and render liquid crystal dynamics. Simulations frames are \
                                  computed one by one and stored in file. Likewise, the renderer reads frames from \
                                  these files.")

  parser.add_argument("--verbose", "-v", dest="verbose_mode", action="store_true", help="enable verbose mode, which \
                      prints more detailed information as frames are computed")
  parser.add_argument("--file-prefix", "-fp", dest="data_path_prefix", action="store", nargs=1, type=str, 
                      metavar="PREFIX", help="specifies a prefix string for automatically generated file names")
  parser.add_argument("--file", "-f", dest="data_path", action="store", nargs=1, type=str, metavar="PATH",
                      help="specifies the full file name (overwrites anything from --file-prefix and the file name \
                            generator)")
  parser.add_argument("--compute", "-c", dest="shouldCompute", action="store_true", help="run the simulator and output \
                      the computed frames to the given file")
  parser.add_argument("--display","-d", dest="shouldDisplay", action="store_true", help="display the interactive \
                      viewing window (after any computations if --compute was specified) from the given file input.")

  parser.add_argument("--time-step", "-dt", dest="time_step", action="store", nargs=1, type=float, metavar="DT",
                      help="specify the timestep between each frame in the simulation")
  parser.add_argument("--space-step", "-ds", dest="space_step", action="store", nargs=1, type=float, metavar="DS",
                      help="specify the distance between consecutive molecules in the simulation")
  parser.add_argument("--tolerance", "-tol", dest="tolerance", action="store", nargs=1, type=float, default=[tolerance],
                      metavar="TOL", help="specify the error tolerance to be used in the iterative solver")
  parser.add_argument("--damping-factor", "-dmp", dest="damping", action="store", nargs=1, type=float, metavar="ALPHA",
                      help="specify the damping factor; the coefficient of n_t in the PDE")

  parser.add_argument("--boundary-conditions", "-bc", dest="boundary_conditions", action="store", nargs=1, type=str,
                      choices=['P', 'N', 'D'], default=[boundary_behavior], help="specify the boundary conditions; \
                      P for periodic, N for Neumann, D for Dirichlet")
  parser.add_argument("--initial-conditions", "-ic", dest="initial_conditions", action="store", nargs=2, type=str,
                      metavar=("MODULE", "FUNCTION"), help="specify a Python module (i.e. a filename without the .py) \
                      and a function in that module to be used for the initial conditions. The function should take \
                      a 3-tuple (x,y,z) to an output 3-tuple. Use NumPy for all math functions (cos, exp, pow, etc.)")
  
  parser.add_argument("--x-bounds", dest="x_bounds", action="store", nargs=2, type=float, default=[min_x, max_x],
                      metavar=("XMIN", "XMAX"), help="specify the left and right extremes of the x-axis")
  parser.add_argument("--y-bounds", dest="y_bounds", action="store", nargs=2, type=float, default=[min_y, max_y],
                      metavar=("YMIN", "YMAX"), help="specify the left and right extremes of the y-axis")
  parser.add_argument("--z-bounds", dest="z_bounds", action="store", nargs=2, type=float, default=[min_z, max_z],
                      metavar=("ZMIN", "ZMAX"), help="specify the left and right extremes of the z-axis")
  parser.add_argument("--sim-time", "-time", dest="end_time", action="store", nargs=1, type=float, default=[final_time],
                      metavar="RUNTIME", help="specify the total amount of simulation time to compute (so the number \
                      of frames is RUNTIME / DT)")

  parser.add_argument("--elastic-constants", "-consts", dest="e_consts", action="store", nargs=3, type=float,
                      metavar=("K1", "K2", "K3"), help="specify the Frank elastic constants, which control the \
                      liquid crystal's elastic response to splay, twist, and bend respectively")
  parser.add_argument("--non-central-terms", "-ncterms", dest="ncterms", action="store", nargs=1, 
                      choices = ['A', 'AC', 'C'], help="specify which terms in F* and the energy (see paper) should \
                      use forward/backward derivatives instead of central derivatives")

  return parser

# Overwrite any default simulation parameters obtained from command line arguments (other than initial conditions)
def set_custom_parameters(args):
  global verbose, a_term_central, c_term_central
  global ds, dt, tolerance, alpha, boundary_behavior, min_x, max_x, min_y, max_y, min_z, max_z, final_time, K1, K2, K3

  verbose = args.verbose_mode

  if (r := args.boundary_conditions) is not None: 
    boundary_behavior = r[0]

  final_time = args.end_time[0]

  filename = ""
  # If there is one, begin the filename with a user-specified prefix
  if (r := args.data_path_prefix) is not None:
    filename += r[0] + "__"

  # Begin the main part of the filename with the boundary behavior and runtime (for example, N-1.0s is Neumann boundary
  # conditions running for 1 second)
  filename += boundary_behavior
  filename += "-%.1fs" % final_time

  # List to store any additional filename modifiers
  mods = []

  # If specified, overwrite the default mesh size
  if (r := args.space_step) is not None:
    ds = r[0]
    mods.append("ds=%.1e" % ds)

  # Set the error tolerance for the iterative solver
  tolerance = args.tolerance

  # Set the boundaries of the x, y, and z axes
  min_x = args.x_bounds[0]
  max_x = args.x_bounds[1]
  min_y = args.y_bounds[0]
  max_y = args.y_bounds[1]
  min_z = args.z_bounds[0]
  max_z = args.z_bounds[1]

  # If specified, overwrite the default Frank elastic constants
  if (r := args.e_consts) is not None: 
    K1 = r[0]
    K2 = r[1]
    K3 = r[2]
    mods.append("K=(%.1f,%.1f,%.1f)" % (K1, K2, K3))

  # Overwrite the value of dt if the user specified
  if (r := args.time_step) is not None:
    dt = r[0]
    mods.append("dt=%.1e" % dt)
  else:
    dt = None

  # Run this now to compute the default value of dt
  init_computation_domain(dt)

  # Overwrite the default derivative behavior if specified
  if (r := args.ncterms) is not None:
    r = r[0]
    if 'A' in r:
      a_term_central = False
    
    if 'C' in r:
      c_term_central = False

    mods.append("nc=%s" % r)

  # Attribs list for more fundamental simulation parameters (namely 3D and damping)
  attribs = ["3D"] if len(z_axis) > 1 else []

  # If damping is nonzero, set alpha and add a damping attribute to the filename
  if (r := args.damping) is not None:
    alpha = r[0]
    if alpha > 0.0001:
      attribs.append("dmp=%.0e" % alpha)

  # Add the modifiers to the filenames
  attribs += mods

  # Add the list of modifiers to the filename, removing unnecessary quotes and spaces
  if len(attribs) > 0: filename += str(attribs).replace('\'', '').replace(' ', '')

  # Finally, add the name of the initial condition function to the end of the filename
  if args.initial_conditions is not None:
    _, func_name = args.initial_conditions
    filename += "{%s}" % func_name

  # If a filename was not specified by the user, use the one that's been created above
  if args.data_path is not None:
    return args.data_path[0]
  else:
    return "outputs/" + filename + ".vfd"

# Overwrite default initial conditions if specified from command line arguments
def get_initial_conditions(args):
  if args.initial_conditions is None:
    return None
  else:
    module_name, func_name = args.initial_conditions

    exec("import %s" % module_name)
    code = "from %s import *\n" + "out = %s"
    load_func = types.FunctionType(compile(code % (module_name, func_name), "temp.py", "exec"), globals())
    load_func()
    return out #type: ignore (tell PyLance not to worry about 'out' being unbound)


if __name__ == "__main__":
  parser = create_parser()
  # If no command line arguments are given, print the command's usage and exit
  if len(sys.argv) == 1:
    parser.print_usage()
    parser.exit()

  args = parser.parse_args()
  simulation_filename = set_custom_parameters(args)
  initial_conditions = get_initial_conditions(args)
  
  # If the compute option is specified, compute with the given settings (if a file is found with these settings,
  # ask if the user wants to use that rather than overwrite it)
  if args.shouldCompute:
    if os.path.isfile(simulation_filename):
      print("File %s already exists, do you want to overwrite it? (yes - enter, no - n + enter)" % simulation_filename)
      ans = input()
      if ans != 'n':
        compute_simulation_frames(simulation_filename, initial_conditions)
    else:
      compute_simulation_frames(simulation_filename, initial_conditions)

  # If the graphics option is specified, display graphics (if a file is not found with the given settings, ask if
  # the user wants to compute a new data file with those settings)
  if args.shouldDisplay:
    if not os.path.isfile(simulation_filename):
      print("File %s not found, would you like to compute it? (yes - enter, no - n + enter)" % simulation_filename)
      ans = input()
      if ans == 'n':
        parser.exit()
      
      compute_simulation_frames(simulation_filename, initial_conditions)

    # Only import graphics if we need it, since it uses a bunch of uncommon libraries
    from Graphics import *
    graphics = Graphics(simulation_filename, 800, 800, "Time: 0.00 seconds", verbose)
    graphics.run(dt)
