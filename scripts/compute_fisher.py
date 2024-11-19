import numpy as np
from load_statistics import *
from Fisherinfor import  fisher

# specific path and parameters
root_path  = '/Volumes/Seagate/Works/20240421_Work/CWTextrema/data/quijote/'
paras      = ['LC', 'EQ', 'OR_LSS','h','ns','Om','Ob2','s8']
delta     = {'LC': 200.,
             'EQ': 200.,
             'OR_LSS': 200.,
             'h': 0.04,
             'ns': 0.04,
             'Om': 0.02,
             'Ob2': 0.004,
             's8': 0.03,
             }
nsims_fiducial    = 15000
nsims_derivatives = 500
scales            = np.linspace(0.1, 0.5, 8)
nscales           = len(scales)

# Initialize the Fisher matrix for five statistics
fisher_info      = np.zeros( (5,len(paras),len(paras)) )

# Load statistics 
fps_fid            = load_fps_fiducial( root_path, nsims_fiducial, kmax=0.5 )
scalevlydf_fid     = load_scale_vly_fiducial( root_path, nsims_fiducial, Imax=8 )
scalepkhf_fid      = load_scale_pk_fiducial( root_path, nsims_fiducial, Imax=8 )
scale_extrema_fid  = load_scale_extrema_fiducial( root_path, nsims_fiducial, Imax=8 )
all_fid            = load_all_fiducial( root_path, nsims_fiducial, Imax=8, kmax=0.5  )

fps_deriv            = load_fps_derivatives(root_path, paras, nsims_derivatives, kmax=0.5  )
scalevlydf_deriv     = load_scale_vly_derivatives(root_path, paras, nsims_derivatives, Imax=8)
scalepkhf_deriv      = load_scale_pk_derivatives(root_path, paras, nsims_derivatives, Imax=8)
scale_extrema_deriv  = load_scale_extrema_derivatives(root_path, paras, nsims_derivatives, Imax=8)
all_deriv            = load_all_derivatives(root_path, paras, nsims_derivatives, Imax=8, kmax=0.5  )

statistics_fid   = [ fps_fid, scalevlydf_fid, scalepkhf_fid, scale_extrema_fid, all_fid ]
statistics_deriv = [ fps_deriv, scalevlydf_deriv, scalepkhf_deriv, scale_extrema_deriv, all_deriv ]

# Compute fisher matrix
for s in range(len(statistics_fid)):
    fisher_info[s,:,:] = fisher( statistics_fid[s],  statistics_deriv[s], statistics_fid[s].shape[1], nsims_fiducial, delta )

# Save 
np.save('/.../fisher_matrix.npy', fisher_info)

