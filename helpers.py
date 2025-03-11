import numpy as np
def cart2sph(x, y, z):
   r = np.sqrt(x**2 + y**2 + z**2) # r = sqrt(x² + y² + z²)
   if r == 0:
      return 0,0
   # normalize
   x,y,z = x/r, y/r, z/r
   # compute
   xy = np.sqrt(x**2 + y**2) # sqrt(x² + y²)
   theta = np.arctan2(y, x) 
   phi = np.arctan2(xy, z) 

   return theta, phi
