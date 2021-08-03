import numpy as np

###############################
##        Parameters         ##
###############################

# Spatial boundary values
ds = 0.25; # Spatial step length
min_x, max_x = -0.5, 0.5
min_y, max_y = -0.5, 0.5
min_z, max_z = -0.5, 0.5

dim = 3 # Spatial dimension (should be either 2 or 3)

final_time = 10.0 # Simulation time
dt = 0.25; # Temporal step length (TODO: move to constants/setup, compute from ds)

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
domain_t, domain_x, domain_y, domain_z = np.meshgrid(t_axis, x_axis, y_axis, z_axis, indexing='ij')

# Discrete spacetime grid
num_t, num_x, num_y, num_z = np.shape(domain_t)
t_disc = np.arange(num_t)
x_disc = np.arange(num_x)
y_disc = np.arange(num_y)
z_disc = np.arange(num_z)
grid_t, grid_x, grid_y, grid_z = np.meshgrid(t_disc, x_disc, y_disc, z_disc, indexing='ij')

# Initialize director and angular momentum fields
nfield = np.zeros((num_t, num_x, num_y, num_z, 3))
wfield = np.zeros((num_t, num_x - 2, num_y - 2, num_z - 2, 3))

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

def diff_t(func):
  times = t_disc[:-1]
  return (func[times + 1] - func[times]) / dt

def mid_t(func):
  times = t_disc[:-1]
  return (func[times + 1] + func[times]) / 2

def diff_s(func, dir, kind='C'):
  xs = x_disc[1:-1]
  ys = y_disc[1:-1]
  zs = z_disc[1:-1]

  denom_mult = 2 if kind == 'C' else 1
  left_shift = 0 if kind == '-' else 1
  right_shift = 0 if kind == '+' else 1

  if dir == 0:
    diff = func[:,xs + left_shift,:,:] - func[:,xs - right_shift,:,:]
  elif dir == 1:
    diff = func[:,:,ys + left_shift,:] - func[:,:,ys - right_shift,:]
  elif dir == 2:
    diff = func[:,:,:,zs + left_shift] - func[:,:,:,zs - right_shift]

  return diff / (denom_mult * ds)

def f_star(n):
  sum = np.zeros((num_t, num_x - 2, num_y - 2, num_z - 2, 3))
  
  for i in range(0, dim):
    for j in range(0, dim):
      for k in range(0, dim):
        ni = n[:,:,:,:,i]
        nj = n[:,:,:,:,j]
        nk = n[:,:,:,:,k]

        term_a = diff_s(diff_s(mid_t(nj, j), i))
        term_b = diff_s(diff_s(mid_t(ni), j, '-'), j, '+')
        term_c = diff_s(mid_t(nj) * nk * mid_t(diff_s(ni, k)))
        term_d = mid_t(nj * diff_s(nk, j)) * diff_s(mid_t(nk), i)

        sum += C1 * term_a + C2 * term_b + C3 * (term_c - term_d)

  return sum

def energy(n):
  term_a = np.zeros(num_t)
  term_b = np.zeros(num_t)
  term_c = np.zeros(num_t)

  for i in range(0, dim):
    for j in range(0, dim):
      print("Summing " + str(i) + ", " + str(j))
      ni = n[:,:,:,:,i]
      nj = n[:,:,:,:,i]

      term_b += np.sum(diff_s(ni, j, '-'))**2
      term_c += np.sum(nj * diff_s(ni, j))

    term_a += np.sum(diff_s(ni, i))

  term_a = term_a**2
  term_c = term_c**2

  return 0.5 * (C1 * term_a + C2 * term_b + C3 * term_c)

print(energy(nfield)) # Takes way too long

###############################
##          Solvers          ##
###############################

def w_solver(n, m):
  # D_t w^{m,v} = F*(n)^{m,v} x n^{m+1/2,v}
  dw = np.cross(f_star(n)[m,:,:,:], mid_t(n)[m,:,:,:])
  return wfield[m,:,:,:] + dt * dw
  

# D_t n^{m,v} = n^{m+1/2,v} x w^{m+1/2,v}
