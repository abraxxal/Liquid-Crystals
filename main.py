import numpy as np

# D_t w^{m,v} = F*(n)^{m,v} x n^{m+1/2,v}
# D_t n^{m,v} = n^{m+1/2,v} x w^{m+1/2,v}
# 
# F*(n) = C1 D_i(D_j n_j^{m+1/2}) + C2 D_j^+(D_j^- n_i^{m+1/2})
#       + C3 [D_j(n_j^{m+1/2}(n_k^m (D_k n_i^m)^{1/2})) - (n_j^m (D_j n_k^m))^{1/2} (D_i n_k^{m+1/2})]

###############################
##        Parameters         ##
###############################

# Spatial boundary values
ds = 0.01; # Spatial step length
min_x, max_x = -0.5, 0.5
min_y, max_y = -0.5, 0.5
min_z, max_z = -ds, ds

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
domain_t, domain_x, domain_y, domain_z = np.meshgrid(t_axis, x_axis, y_axis, z_axis, indexing='ij')

# Discrete spacetime grid
num_t, num_x, num_y, num_z = np.shape(domain_t)
grid = np.meshgrid(np.arange(num_t), np.arange(num_x), np.arange(num_y), np.arange(num_z), indexing='ij')
grid_t, grid_x, grid_y, grid_z = grid

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

def diff_t(func, t):
  return (func[t + 1] - func[t]) / dt

def mid_t(func, t):
  return (func[t + 1] + func[t]) / 2

def diff_s(func, dir, s, kind='C'):
  denom_mult = 2 if kind == 'C' else 1
  left_shift = 0 if kind == '-' else 1
  right_shift = 0 if kind == '+' else 1

  if dir == 0:
    left = func[:,s + left_shift,:,:]
    right = func[:,s - right_shift,:,:]
  elif dir == 1:
    left = func[:,:,s + left_shift,:]
    right = func[:,:,s - right_shift,:]
  elif dir == 2:
    left = func[:,:,:,s + left_shift]
    right = func[:,:,:,s - right_shift]

  return (left - right) / (denom_mult * ds)

# Implement F*(n) here (TODO)

###############################
##          Solvers          ##
###############################
