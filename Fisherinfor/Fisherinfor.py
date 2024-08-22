import numpy as np

#-------------------------------------------------------
def covariance( statistic_fiducial,  nbins, nsims ):

    '''
    The covariance matrix of the statistic vector.
    '''
    
    # The mean of the statistic over simulations, statistic.shape=(nsim,nbins)
    mean                = np.mean(statistic_fiducial, axis=0) 
    statistic_fiducial -= mean
    sum_stat            = None

    for i in range(nsims):
        stat = statistic_fiducial[i,:]
        if (sum_stat is None):
            sum_stat  = np.outer(stat,stat)
        else:
            sum_stat += np.outer(stat,stat)

    return sum_stat/(nsims-nbins-2)

#-------------------------------------------------------
def correlation( statistic_fiducial,  nbins, nsims ):

    '''
    The correlation matrix of the statistic vector.
    '''

    cov = covariance( statistic_fiducial,  nbins, nsims )
    cii = np.diagonal(cov)

    return cov/np.sqrt( np.outer(cii,cii) )

#-------------------------------------------------------
def derivative( statistics, delta ):

    '''
    The partial derivative of the statistic with respect to a parameter.
    '''

    result = []

    for para in delta.keys():
        dpara = delta[para]
        stat_p = np.mean( statistics[para][0], axis=0 )
        stat_m = np.mean( statistics[para][1], axis=0 )
        result.append( (stat_p - stat_m)/dpara )

    return np.asarray( result )

#-------------------------------------------------------       
def fisher( statistic_fiducial,  statistics, nbins, nsims_fiducial, delta ):

    '''
    The Fisher matrix.
    '''

    deriv   = derivative( statistics, delta )
    cov     = covariance( statistic_fiducial,  nbins, nsims_fiducial )
    cov_inv = np.linalg.inv(cov)

    return deriv @ cov_inv @ deriv.T 

#-------------------------------------------------------
def fisher_dia_cov( statistic_fiducial,  statistics, nbins, nsims_fiducial, delta ):

    '''
    The Fisher matrix using only the diagonal elements of covariance matrix. 
    '''

    deriv   = derivative( statistics, delta )
    cov     = covariance( statistic_fiducial,  nbins, nsims_fiducial )
    cov_dia = np.zeros_like(cov)
    np.fill_diagonal(cov_dia, np.diagonal(cov) )
    cov_inv = np.linalg.inv(cov_dia)

    return deriv @ cov_inv @ deriv.T 

#-------------------------------------------------------
def signal2noise( statistic_fiducial,  nbins, nsims_fiducial ):

    '''
    The signal-to-noise ratio of the statistics.
    '''

    mean    = np.mean(statistic_fiducial, axis=0)
    cov     = covariance( statistic_fiducial,  nbins, nsims_fiducial )
    cov_inv = np.linalg.inv(cov)

    return np.sqrt( mean @ cov_inv @ mean.T)

#-------------------------------------------------------
def signal2noise_dia_cov( statistic_fiducial,  nbins, nsims_fiducial ):

    '''
    The signal-to-noise ratio using only the diagonal elements of covariance matrix. 
    '''

    mean    = np.mean(statistic_fiducial, axis=0)
    cov     = covariance( statistic_fiducial,  nbins, nsims_fiducial )
    cov_dia = np.zeros_like(cov)
    np.fill_diagonal(cov_dia, np.diagonal(cov) )
    cov_inv = np.linalg.inv(cov_dia)

    return np.sqrt( mean @ cov_inv @ mean.T)
