import os
import numpy as np
from mpi4py import MPI
from CWTextrema import log_density, scaleXTREF

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
dataset       = 'fiducial/'      # EQ_m, EQ_p, LC_m, LC_p, ...
root_path     = '/.../'+dataset
nrealizs      = 15000            # 500 for derivative simulations 
nmesh         = 512
boxsize       = 1000.0           # Mpc/h
height_bins   = np.linspace(0.0, 4.5, 11, dtype=np.float32)
depth_bins    = np.linspace(-5.4,0.0,13,dtype=np.float32)
height_center = 0.5*(height_bins[1:] + height_bins[:-1])
depth_center  = 0.5*(depth_bins[1:] + depth_bins[:-1])

# Parallel
for i in range(rank, nrealizs, size):
    # Load the density fields, i.e. delta+1
    dens_field = np.load(root_path+'dens_fields/'+str(i)+'/z=0/dens_field.npy')
    # log-density
    log_dens   = log_density( dens_field )
    dens_field = None
    # Compute the scaleVLYDF and scalePKHF
    scales, result_vly, result_pk = scaleXTREF( log_dens, height_bins, depth_bins,
                                               boxsize, nmesh, 
                                               kmin=0.1, kmax=0.5, nscales=8, scale_interval='linear',
                                               wavelet='iso_gdw' )
    # Create the directory
    peaks_dir = root_path+'peaks/'+str(i)+'/z=0/'
    valleys_dir = root_path+'valleys/'+str(i)+'/z=0/'
    os.makedirs( peaks_dir )
    os.makedirs( valleys_dir )
    np.savez(valleys_dir+'valleys.npz',k_pseu=scales,depth=depth_center,scaleVLYDF=result_vly)
    np.savez(peaks_dir+'peaks.npz',k_pseu=scales,height=height_center,scalePKHF=result_pk)