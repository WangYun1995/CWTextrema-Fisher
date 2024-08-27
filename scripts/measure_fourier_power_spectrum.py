import os
import numpy as np
from mpi4py import MPI
from PowerSpectrum import fps

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
dataset       = 'fiducial/'      # EQ_m, EQ_p, LC_m, LC_p, ...
root_path     = '/.../'+dataset
nrealizs      = 15000            # 500 for derivative simulations 
nmesh         = 512
boxsize       = 1000.0           # Mpc/h

# Parallel
for i in range(rank, nrealizs, size):
    # Load the density fields, i.e. delta+1
    dens_field = np.load(root_path+'dens_fields/'+str(i)+'/z=0/dens_field.npy')
    # Compute the power spectrum
    wavenumber, power = fps( dens_field-1, boxsize, window='PCS' )
    # Create the directory
    grid_dir = root_path+'fps/'+str(i)+'/z=0/'
    os.makedirs( grid_dir )
    np.savez(grid_dir+'fps.npz',k=wavenumber, Pk=power)