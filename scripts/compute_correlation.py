import numpy as np
from Fisherinfor import correlation
from load_statistics import load_all_fiducial


# Specific path and parameters
root_path       = '/.../'
nsims_fiducial  = 15000

# Load statistics
statistic_fid =load_all_fiducial( root_path, nsims_fiducial, Imax=8, kmax=0.5 )

# Compute the correlation matrix
corr = correlation( statistic_fid,  statistic_fid.shape[1], nsims_fiducial )
# save 
np.save('/.../correlation.npy', corr)