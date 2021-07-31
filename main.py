import numpy as np


# D_t w^m = F*(n)^m x n^{m+1/2}
# D_t n^m = n^{m+1/2} x w^{m+1/2}
# 
# F*(n) = C1 D_i(D_j n_j^{m+1/2}) + C2 D_j^+(D_j^- n_i^{m+1/2})
#       + C3 [D_j(n_j^{m+1/2}(n_k^m (D_k n_i^m)^{1/2})) - (n_j^m (D_j n_k^m))^{1/2} (D_i n_k^{m+1/2})]

###############################
##        Parameters         ##
###############################

ds = 0.01; # Spatial step length

# Frank elastic constants
K1 = 0.5; # Splay
K2 = 1.0; # Twist
K3 = 0.5; # Bend

###############################
##    Constants and Setup    ##
###############################

# Constants from paper
C1 = K1 - K2
C2 = K2
C3 = K3 - K2

dt = 0.05; # Temporal step length (TODO: Compute from ds)

###############################
##         Operators         ##
###############################