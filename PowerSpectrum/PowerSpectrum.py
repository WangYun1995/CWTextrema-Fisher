import Pk_library as PKL


def fps( delta, boxsize, window ):
    '''
    Compute the Fourier-based power spectrum
    '''
    Pk = PKL.Pk(delta, boxsize, axis=0, MAS=window, threads=1, verbose=True)
    return Pk.k3D, Pk.Pk[:,0]