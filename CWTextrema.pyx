import numpy as np
import pyfftw
from scipy import stats
import multiprocessing
cimport numpy as cnp
cimport cython
cnp.import_array()

pyfftw.config.NUM_THREADS    = multiprocessing.cpu_count()
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

#-------------------------------------------------------
@cython.cdivision(True) 
def iso_gdw( float k_pseu, cnp.ndarray[cnp.float32_t, ndim=3] k ):

    '''
    The isotropic Gaussidan derived wavelet in the Fourier domain.
    '''

    cdef float cw, scale, cn 
    cdef cnp.ndarray k2  = np.empty_like(k, dtype=np.float32)
    cdef cnp.ndarray psi = np.empty_like(k, dtype=np.float32)

    cw    = 2/np.sqrt(7.0)
    scale = cw*k_pseu
    cn    = 16*np.sqrt( np.sqrt(2*np.pi**3)/15.0 )
    k2    = (k/scale)**2 
    psi   = cn*k2*np.exp(-k2)
    psi  /= np.sqrt( scale**3 )
    return psi

#-------------------------------------------------------
def periodic_index( int indx, int num ):

    '''
    This function is used to set periodic boundary condition.
    '''

    cdef int im1, ip1 

    if (indx==0):
        im1 = num-1;  ip1 = 1
    elif ( (indx>1)&(indx<num-1) ):
        im1 = indx-1; ip1 = indx+1
    else:
        im1 = num-2;  ip1 = 0
    return im1, ip1

#-------------------------------------------------------
def log_density( 
    cnp.ndarray[cnp.float32_t, ndim=3] density
    ):
    '''
    The logarithmic transform of the density field
    '''

    if (density.min()==0):
        density[density==0] = 0.5*( density[density>0].min() )
 
    return np.log(density)

#-------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def peaksfinder3D( 
    cnp.ndarray[cnp.float32_t, ndim=3] arr
    ):
    '''
    The peak finder for the 3D density field
    '''

    cdef int num0 = arr.shape[0]
    cdef int num1 = arr.shape[1]
    cdef int num2 = arr.shape[2]
    cdef int i, j, k 
    cdef int im1, jm1, km1
    cdef int ip1, jp1, kp1 

    # Intialize the empty list to store results
    heights      = []  # heights of the peaks
    coords       = []  # coordinates of the peaks
    
    # Find all peaks in arr
    for i in range(num0):
        im1, ip1 = periodic_index(i, num0)
        for j in range(num1):
            jm1, jp1 = periodic_index(j, num1)
            for k in range(num2):
                km1, kp1 = periodic_index(k, num2)
                # Delete the elements less than zero
                if( arr[i,j,k]<0 ):
                    continue
                # Check with top and bottom elements
                if ( (arr[i,j,k]<arr[im1,j,k])|(arr[i,j,k]<arr[ip1,j,k]) ):
                    continue
                # Check with front and back elements
                if ( (arr[i,j,k]<arr[i,jm1,k])|(arr[i,j,k]<arr[i,jp1,k]) ):
                    continue
                # Check with left and right elements
                if ( (arr[i,j,k]<arr[i,j,km1])|(arr[i,j,k]<arr[i,j,kp1]) ):
                    continue
                
                # results
                heights.append( arr[i,j,k] )
                coords.append( np.array([i,j,k]) )
    
    return np.asarray(heights,dtype=np.float32), np.asarray(coords,dtype=np.float32)

#-------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def valleysfinder3D( 
    cnp.ndarray[cnp.float32_t, ndim=3] arr
    ):
    
    '''
    The valley finder for the 3D density field.
    '''

    cdef int num0 = arr.shape[0]
    cdef int num1 = arr.shape[1]
    cdef int num2 = arr.shape[2]
    cdef int i, j, k 
    cdef int im1, jm1, km1
    cdef int ip1, jp1, kp1 

    # Intialize the empty list to store results
    depths      = []  # depths of the minima
    coords      = []  # coordinates of the minima
    
    # Find all peaks in arr
    for i in range(num0):
        im1, ip1 = periodic_index(i, num0)
        for j in range(num1):
            jm1, jp1 = periodic_index(j, num1)
            for k in range(num2):
                km1, kp1 = periodic_index(k, num2)
                # Delete the elements greater than zero
                if( arr[i,j,k]>0 ):
                    continue
                # Check with top and bottom elements
                if ( (arr[i,j,k]>arr[im1,j,k])|(arr[i,j,k]>arr[ip1,j,k]) ):
                    continue
                # Check with front and back elements
                if ( (arr[i,j,k]>arr[i,jm1,k])|(arr[i,j,k]>arr[i,jp1,k]) ):
                    continue
                # Check with left and right elements
                if ( (arr[i,j,k]>arr[i,j,km1])|(arr[i,j,k]>arr[i,j,kp1]) ):
                    continue
                
                # results
                depths.append( arr[i,j,k] )
                coords.append( np.array([i,j,k]) )
    
    return np.asarray(depths,dtype=np.float32), np.asarray(coords,dtype=np.float32)

#-------------------------------------------------------
@cython.cdivision(True)  
def cwt( 
    cnp.ndarray[cnp.complex64_t, ndim=3] complex_field, 
    int nmesh, 
    cnp.ndarray[cnp.float32_t, ndim=3] k, 
    float k_pseu, 
    str wavelet='iso_gdw'
    ):
    
    '''
    The continuous wavelet transform of the log-density at the scale of k_pseu.
    '''

    cdef cnp.ndarray    result = np.empty((nmesh,nmesh,nmesh), dtype=np.float32)

    if (wavelet=='iso_gdw'):
        complex_cwt = pyfftw.empty_aligned( (nmesh,nmesh,nmesh//2+1), dtype=np.complex64 )
        complex_cwt = complex_field*iso_gdw( k_pseu,k )
        ifft_object = pyfftw.builders.irfftn( complex_cwt, s=(nmesh,nmesh,nmesh), axes=(0,1,2) )
        result      = ifft_object()
        return result

#-------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)     
def scalePKHF( 
    cnp.ndarray[cnp.float32_t, ndim=3] log_dens, 
    cnp.ndarray[cnp.float32_t, ndim=1] height_bins, 
    float boxsize,
    int   nmesh,
    float kmin, 
    float kmax, 
    int nscales, 
    str wavelet='iso_gdw' 
    ):
    
    '''The scale-dependent peak height function
    '''

    cdef int i
    cdef cnp.ndarray            scales        = np.geomspace(kmin, kmax, nscales, dtype=np.float32) # set wavelet scales (pseu wave numbers)
    cdef cnp.ndarray            result        = np.empty( (nscales,len(height_bins)-1), dtype=np.float32)
    cdef cnp.ndarray            wavelet_trans = np.empty( (nmesh,nmesh,nmesh), dtype=np.float32) 
    cdef cnp.float32_t[:]       heights
    cdef cnp.ndarray            complex_field = np.empty( (nmesh,nmesh,nmesh//2+1), dtype=np.complex64)

    # wave vector
    k_ =  (2.0*np.pi/boxsize)*np.linspace(-nmesh/2., nmesh/2.-1.0, nmesh, endpoint=True, dtype=np.float32)
    kx = np.fft.ifftshift(k_)
    kkx, kky, kkz = np.meshgrid(kx,kx,kx[0:nmesh//2+1], indexing="ij")
    k  =  np.sqrt( kkx**2 + kky**2 + kkz**2 )

    # Fourier transform of the density field
    real_input    = pyfftw.empty_aligned( (nmesh,nmesh,nmesh), dtype=np.float32 )
    real_input    = log_dens 
    fft_object    = pyfftw.builders.rfftn( real_input, s=(nmesh,nmesh,nmesh), axes=(0,1,2) )
    complex_field = fft_object()

    # Loop
    for i in range(nscales):
        wavelet_trans      = cwt( complex_field, nmesh, k, scales[i], wavelet=wavelet )
        heights,_          = peaksfinder3D( wavelet_trans )
        heights           /= np.sqrt( np.mean(wavelet_trans**2) )
        result[i,:], _ , _ = stats.binned_statistic(heights, heights, 'count', bins=height_bins)

    return scales, result/boxsize**3

#-------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)     
def scaleVLYDF( 
    cnp.ndarray[cnp.float32_t, ndim=3] log_dens, 
    cnp.ndarray[cnp.float32_t, ndim=1] depth_bins, 
    float boxsize,
    int   nmesh,
    float kmin, 
    float kmax, 
    int nscales, 
    str wavelet='iso_gdw' 
    ):

    '''
    The scale-dependent valley depth function.
    '''
    
    cdef int i
    cdef cnp.ndarray            scales        = np.geomspace(kmin, kmax, nscales, dtype=np.float32) # set wavelet scales (pseu wave numbers)
    cdef cnp.ndarray            result        = np.empty( (nscales,len(depth_bins)-1), dtype=np.float32)
    cdef cnp.ndarray            wavelet_trans = np.empty( (nmesh,nmesh,nmesh), dtype=np.float32) 
    cdef cnp.float32_t[:]       depths
    cdef cnp.ndarray            complex_field = np.empty( (nmesh,nmesh,nmesh//2+1), dtype=np.complex64)

    # wave vector
    k_ =  (2.0*np.pi/boxsize)*np.linspace(-nmesh/2., nmesh/2.-1.0, nmesh, endpoint=True, dtype=np.float32)
    kx = np.fft.ifftshift(k_)
    kkx, kky, kkz = np.meshgrid(kx,kx,kx[0:nmesh//2+1], indexing="ij")
    k  =  np.sqrt( kkx**2 + kky**2 + kkz**2 )

    # Fourier transform of the density field
    real_input    = pyfftw.empty_aligned( (nmesh,nmesh,nmesh), dtype=np.float32 )
    real_input    = log_dens 
    fft_object    = pyfftw.builders.rfftn( real_input, s=(nmesh,nmesh,nmesh), axes=(0,1,2) )
    complex_field = fft_object()

    # Loop
    for i in range(nscales):
        wavelet_trans      = cwt( complex_field, nmesh, k, scales[i], wavelet=wavelet )
        depths,_           = valleysfinder3D( wavelet_trans )
        depths            /= np.sqrt( np.mean(wavelet_trans**2) )
        result[i,:], _ , _ = stats.binned_statistic(depths, depths, 'count', bins=depth_bins)

    return scales, result/boxsize**3

#-------------------------------------------------------
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)     
def scaleXTREF( 
    cnp.ndarray[cnp.float32_t, ndim=3] log_dens, 
    cnp.ndarray[cnp.float32_t, ndim=1] height_bins, 
    cnp.ndarray[cnp.float32_t, ndim=1] depth_bins, 
    float boxsize,
    int   nmesh,
    float kmin, 
    float kmax, 
    int nscales, 
    str scale_interval = 'linear',
    str wavelet='iso_gdw' 
    ):

    '''
    This function can yield the scale-PKHF and scale-VLYDF simultaneously.
    '''
    
    cdef int i
    cdef cnp.ndarray            scales_lin    = np.linspace(kmin, kmax, nscales, dtype=np.float32) # set wavelet scales (pseu wave numbers)
    cdef cnp.ndarray            scales_log    = np.geomspace(kmin, kmax, nscales, dtype=np.float32) # set wavelet scales (pseu wave numbers)
    cdef cnp.ndarray            scales        = np.empty( nscales, dtype=np.float32)
    cdef cnp.ndarray            result_vly    = np.empty( (nscales,len(depth_bins)-1), dtype=np.float32)
    cdef cnp.ndarray            result_pk     = np.empty( (nscales,len(height_bins)-1), dtype=np.float32)
    cdef cnp.ndarray            wavelet_trans = np.empty( (nmesh,nmesh,nmesh), dtype=np.float32) 
    cdef cnp.float32_t[:]       depths
    cdef cnp.float32_t[:]       heights 
    cdef cnp.ndarray            complex_field = np.empty( (nmesh,nmesh,nmesh//2+1), dtype=np.complex64)

    # wave vector
    k_ =  (2.0*np.pi/boxsize)*np.linspace(-nmesh/2., nmesh/2.-1.0, nmesh, endpoint=True, dtype=np.float32)
    kx = np.fft.ifftshift(k_)
    kkx, kky, kkz = np.meshgrid(kx,kx,kx[0:nmesh//2+1], indexing="ij")
    k  =  np.sqrt( kkx**2 + kky**2 + kkz**2 )

    # Fourier transform of the density field
    real_input    = pyfftw.empty_aligned( (nmesh,nmesh,nmesh), dtype=np.float32 )
    real_input    = log_dens 
    fft_object    = pyfftw.builders.rfftn( real_input, s=(nmesh,nmesh,nmesh), axes=(0,1,2) )
    complex_field = fft_object()

    # Loop
    if (scale_interval == 'linear'):
        scales = scales_lin 
    elif (scale_interval== 'log'):
        scales = scales_log
    for i in range(nscales):
        wavelet_trans      = cwt( complex_field, nmesh, k, scales[i], wavelet=wavelet )
        depths,_           = valleysfinder3D( wavelet_trans )
        heights,_          = peaksfinder3D( wavelet_trans )
        depths            /= np.sqrt( np.mean(wavelet_trans**2) )
        heights           /= np.sqrt( np.mean(wavelet_trans**2) )
        result_vly[i,:], _ , _ = stats.binned_statistic(depths, depths, 'count', bins=depth_bins)
        result_pk[i,:], _ , _  = stats.binned_statistic(heights, heights, 'count', bins=height_bins)

    return scales, result_vly/boxsize**3, result_pk/boxsize**3


