import numpy as np

###############################
##        Parameters         ##
###############################

# Spatial boundary values
ds = 0.01; # Spatial step length
min_x, max_x = -0.5, 0.5
min_y, max_y = -0.5, 0.5
min_z, max_z = -0.5, 0.5

dim = 3 # Spatial dimension (should be either 2 or 3)

final_time = 10.0 # Simulation time
dt = 0.005; # Temporal step length (TODO: move to constants/setup, compute from ds)

# Frank elastic constants
K1 = 0.5; # Splay
K2 = 1.0; # Twist
K3 = 0.5; # Bend

###############################
##    Constants and Setup    ##
###############################

# Constants from the paper
C1 = K1 - K2
C2 = K2
C3 = K3 - K2

# Discrete spacetime, the domain of the director and angular momentum fields
t_axis = np.arange(0, final_time + dt, dt)
x_axis = np.arange(min_x, max_x + ds, ds)
y_axis = np.arange(min_y, max_y + ds, ds)
z_axis = np.arange(min_z, max_z + ds, ds)
domain_x, domain_y, domain_z = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')

# Discrete spacetime grid
num_t, = np.shape(t_axis)
num_x, num_y, num_z = np.shape(domain_x)
t_disc = np.arange(num_t)
x_disc = np.arange(num_x)
y_disc = np.arange(num_y)
z_disc = np.arange(num_z)
grid_x, grid_y, grid_z = np.meshgrid(x_disc, y_disc, z_disc, indexing='ij')

# Initialize director and angular momentum fields
nfield = np.zeros((num_x, num_y, num_z, 3))
wfield = np.zeros((num_x, num_y, num_z, 3))

# Set initial values (TODO)
# nfield[0,:,:,:,0] = 
# nfield[0,:,:,:,1] = 
# nfield[0,:,:,:,2] = 
# wfield[0,:,:,:,0] = 
# wfield[0,:,:,:,1] = 
# wfield[0,:,:,:,2] = 

###############################
##         Operators         ##
###############################

def diff_t(old, new):
  return (new - old) / dt

def mid_t(old, new):
  return (old + new) / 2

def diff_s(func, dir, kind='C'):
  denom_mult = 2 if kind == 'C' else 1
  left_shift = 0 if kind == '-' else 1
  right_shift = 0 if kind == '+' else 1

  if dir == 0:
    diff = func[(x_disc + left_shift) % num_x,:,:] - func[(x_disc - right_shift) % num_x,:,:]
  elif dir == 1:
    diff = func[:,(y_disc + left_shift) % num_y,:] - func[:,(y_disc - right_shift) % num_y,:]
  elif dir == 2:
    diff = func[:,:,(z_disc + left_shift) % num_z] - func[:,:,(z_disc - right_shift) % num_z]

  return diff / (denom_mult * ds)

def f_star(nfield_old, nfield_new):
  out = np.zeros((num_x, num_y, num_z, 3))
  
  for i in range(0, dim):
    sum = np.zeros((num_x, num_y, num_z))

    for j in range(0, dim):
      for k in range(0, dim):
        nmid = mid_t(nfield_old[:,:,:], nfield_new[:,:,:])

        ni = nfield_old[:,:,:,i]
        nj = nfield_old[:,:,:,j]
        nk = nfield_old[:,:,:,k]

        mid_ni = nmid[:,:,:,i]
        mid_nj = nmid[:,:,:,j]
        mid_nk = nmid[:,:,:,k]

        new_ni = nfield_new[:,:,:,i]
        new_nj = nfield_new[:,:,:,j]
        new_nk = nfield_new[:,:,:,k]

        term_a = diff_s(diff_s(mid_nj, j), i)
        term_b = diff_s(diff_s(mid_ni, j, '-'), j, '+')
        term_c = diff_s(mid_nj * nk * mid_t(diff_s(ni, k), diff_s(new_ni, k)), j)
        term_d = mid_t(nj * diff_s(nk, j), new_nj * diff_s(new_nk, j)) * diff_s(mid_nk, i)

        sum += C1 * term_a + C2 * term_b + C3 * (term_c - term_d)

    out[:,:,:,i] = sum
  
  return out

def energy(nfield):
  term_a, term_b, term_c = (0, 0, 0)

  for i in range(0, dim):
    for j in range(0, dim):
      ni = nfield[:,:,:,i]
      nj = nfield[:,:,:,i]

      term_b += np.sum(diff_s(ni, j, '-'))**2
      term_c += np.sum(nj * diff_s(ni, j))

    term_a += np.sum(diff_s(ni, i))

  term_a = term_a**2
  term_c = term_c**2

  return 0.5 * (C1 * term_a + C2 * term_b + C3 * term_c)


###############################
##          Solvers          ##
###############################

def w_solver(nfield_old, nfield_new):
  # D_t w^{m,v} = F*(n)^{m,v} x n^{m+1/2,v}
  dw = np.cross(f_star(nfield_old, nfield_new)[:,:,:], mid_t(nfield_old, nfield_new)[:,:,:])
  return wfield[:,:,:] + dt * dw
  

def n_solver(nfield_old, wfield_old):
  return 3

# D_t n^{m,v} = n^{m+1/2,v} x w^{m+1/2,v}

# Things for meeting:
# - Implement naive solver for n (needs the V matrix from section 4)
# - Implement iterative solver
