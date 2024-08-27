import numpy as np
from load_statistics import *
from Fisherinfor import  signal2noise

# Specific path and parameters
root_path      = '/.../'
nsims_fiducial = 15000
scales         = np.linspace(0.1, 0.5, 8)
nscales        = len(scales)

# Initialize the signal-to-noise ratio
s2n = np.zeros( (5,nscales) )

# Load statistics
for i, kmax in enumerate(scales):
    fps_fid            = load_fps_fiducial( root_path, nsims_fiducial, kmax=kmax )
    scalevlydf_fid     = load_scale_vly_fiducial( root_path, nsims_fiducial, i+1 )
    scalepkhf_fid      = load_scale_pk_fiducial( root_path, nsims_fiducial, i+1 )
    scale_extrema_fid  = load_scale_extrema_fiducial( root_path, nsims_fiducial, i+1 )
    all_fid            = load_all_fiducial( root_path, nsims_fiducial, i+1, kmax=kmax  )

    statistics         = [fps_fid, scalevlydf_fid, scalepkhf_fid, scale_extrema_fid, all_fid]
    for s in range(len(statistics)):
        s2n[s,i] = signal2noise( statistics, statistics[s].shape[1], nsims_fiducial )

np.savez('/.../signal2noise.npz', 
         kmax = scales,
         s2n  = s2n
         )
