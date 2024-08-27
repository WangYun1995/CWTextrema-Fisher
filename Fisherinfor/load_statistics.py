import numpy as np

#-------------------------------------------------------
def load_scale_pk_fiducial( root_path, nsims_fiducial, Imax ):
    # initialize
    statistic_fiducial = []

    # Load the scalePKHF from fiducial
    for i in range(nsims_fiducial): 
        peaks   = np.load(root_path+'fiducial/peaks/'+str(i)+'/z=0/peaks.npz')
        pkhf    = np.ravel( peaks['scalePKHF'][0:Imax,:] )
        statistic_fiducial.append( pkhf )
        
    return np.asarray(statistic_fiducial)

#-------------------------------------------------------
def load_scale_vly_fiducial( root_path, nsims_fiducial, Imax ):
    # initialize
    statistic_fiducial = []

    # Load the scalePKHF and scaleVLYDF from fiducial
    for i in range(nsims_fiducial): 
        valleys = np.load(root_path+'fiducial/valleys/'+str(i)+'/z=0/valleys.npz')
        vlydf   = np.ravel( valleys['scaleVLYDF'][0:Imax,:] )
        statistic_fiducial.append( vlydf )

    return np.asarray(statistic_fiducial)

#-------------------------------------------------------
def load_scale_extrema_fiducial( root_path, nsims_fiducial, Imax ):
    # initialize
    statistic_fiducial = []

    # Load the scalePKHF and scaleVLYDF from fiducial
    for i in range(nsims_fiducial): 
        valleys = np.load(root_path+'fiducial/valleys/'+str(i)+'/z=0/valleys.npz')
        peaks   = np.load(root_path+'fiducial/peaks/'+str(i)+'/z=0/peaks.npz')

        vlydf   = np.ravel( valleys['scaleVLYDF'][0:Imax,:] )
        pkhf    = np.ravel( peaks['scalePKHF'][0:Imax,:] )
        stat    = np.concatenate((vlydf, pkhf)) 
        statistic_fiducial.append( stat )
        

    return np.asarray(statistic_fiducial)

#-------------------------------------------------------
def load_fps_fiducial( root_path, nsims_fiducial, kmax ):
    # initialize
    statistic_fiducial = []

    # Load the scalePKHF and scaleVLYDF from fiducial
    for i in range(nsims_fiducial): 
        fps  = np.load(root_path+'fiducial/fps/'+str(i)+'/z=0/fps.npz')
        if ( i == 0 ):
            k = fps['k']
            kk = k[k<kmax]
        
        Pk = fps['Pk']  
        Pkk = Pk[0:len(kk)-1]
        statistic_fiducial.append( Pkk )
        
    return np.asarray(statistic_fiducial)

#-------------------------------------------------------
def load_all_fiducial( root_path, nsims_fiducial, Imax, kmax ):
    # initialize
    statistic_fiducial = []

    # Load the scalePKHF and scaleVLYDF from fiducial
    for i in range(nsims_fiducial): 
        valleys = np.load(root_path+'fiducial/valleys/'+str(i)+'/z=0/valleys.npz')
        peaks   = np.load(root_path+'fiducial/peaks/'+str(i)+'/z=0/peaks.npz')
        fps     = np.load(root_path+'fiducial/fps/'+str(i)+'/z=0/fps.npz')

        if ( i == 0 ):
            k = fps['k']
            kk = k[k<kmax]

        vlydf   = np.ravel( valleys['scaleVLYDF'][0:Imax,:] )
        pkhf    = np.ravel( peaks['scalePKHF'][0:Imax,:] )
        Pk      = fps['Pk'][0:len(kk)-1]  
        stat    = np.concatenate( (vlydf, pkhf, Pk) )
        statistic_fiducial.append( stat )
        
    return np.asarray(statistic_fiducial)

#-------------------------------------------------------
def load_scale_pk_derivatives(root_path, paras, nsims_derivatives, Imax):
    # initialize
    statistics = {}  # dictionary
    for para in paras:
        statistics[para] = None

    # Load the scaleVLYDF and scalePKHF from EQ, LS, OR_LSS, ...
    for para in paras:
        stat_p   = []
        stat_m   = []
        for i in range(nsims_derivatives):   
            peaks_p   = np.load(root_path+para+'_p/peaks/'+str(i)+'/z=0/peaks.npz')
            peaks_m   = np.load(root_path+para+'_m/peaks/'+str(i)+'/z=0/peaks.npz')
            pkhf_p    = np.ravel( peaks_p['scalePKHF'][0:Imax,:] )
            pkhf_m    = np.ravel( peaks_m['scalePKHF'][0:Imax,:] )

            stat_p.append( pkhf_p )
            stat_m.append( pkhf_m )

        statistics[para] = [stat_p, stat_m]

    return statistics

#-------------------------------------------------------
def load_scale_vly_derivatives(root_path, paras, nsims_derivatives, Imax):
    # initialize
    statistics = {}  # dictionary
    for para in paras:
        statistics[para] = None

    # Load the scaleVLYDF and scalePKHF from EQ, LS, OR_LSS, ...
    for para in paras:
        stat_p   = []
        stat_m   = []
        for i in range(nsims_derivatives):   
            valleys_p = np.load(root_path+para+'_p/valleys/'+str(i)+'/z=0/valleys.npz')
            valleys_m = np.load(root_path+para+'_m/valleys/'+str(i)+'/z=0/valleys.npz')
            vlydf_p   = np.ravel( valleys_p['scaleVLYDF'][0:Imax,:] )
            vlydf_m   = np.ravel( valleys_m['scaleVLYDF'][0:Imax,:] )


            stat_p.append( vlydf_p  )
            stat_m.append( vlydf_m  )

        statistics[para] = [stat_p, stat_m]

    return statistics

#-------------------------------------------------------
def load_scale_extrema_derivatives(root_path, paras, nsims_derivatives, Imax):
    # initialize
    statistics = {}  # dictionary
    for para in paras:
        statistics[para] = None

    # Load the scaleVLYDF and scalePKHF from EQ, LS, OR_LSS, ...
    for para in paras:
        stat_p   = []
        stat_m   = []
        for i in range(nsims_derivatives):   
            valleys_p = np.load(root_path+para+'_p/valleys/'+str(i)+'/z=0/valleys.npz')
            peaks_p   = np.load(root_path+para+'_p/peaks/'+str(i)+'/z=0/peaks.npz')
            valleys_m = np.load(root_path+para+'_m/valleys/'+str(i)+'/z=0/valleys.npz')
            peaks_m   = np.load(root_path+para+'_m/peaks/'+str(i)+'/z=0/peaks.npz')
            
            vlydf_p   = np.ravel( valleys_p['scaleVLYDF'][0:Imax,:] )
            pkhf_p    = np.ravel( peaks_p['scalePKHF'][0:Imax,:] )
            vlydf_m   = np.ravel( valleys_m['scaleVLYDF'][0:Imax,:] )
            pkhf_m    = np.ravel( peaks_m['scalePKHF'][0:Imax,:] )

            stat_p.append( np.concatenate((vlydf_p, pkhf_p))  )
            stat_m.append( np.concatenate((vlydf_m, pkhf_m))  )

        statistics[para] = [stat_p, stat_m]

    return statistics

#-------------------------------------------------------
def load_fps_derivatives(root_path, paras, nsims_derivatives, kmax):
    # initialize
    statistics = {}  # dictionary
    for para in paras:
        statistics[para] = None

    # Load the scaleVLYDF and scalePKHF from EQ, LS, OR_LSS, ...
    for para in paras:
        stat_p   = []
        stat_m   = []
        for i in range(nsims_derivatives): 
            fps_p = np.load(root_path+para+'_p/fps/'+str(i)+'/z=0/fps.npz')
            fps_m = np.load(root_path+para+'_m/fps/'+str(i)+'/z=0/fps.npz')  
            if ( i == 0 ):
                k  = fps_p['k']
                kk = k[k<kmax]
            Pk_p = fps_p['Pk'][0:len(kk)-1]
            Pk_m = fps_m['Pk'][0:len(kk)-1]


            stat_p.append( Pk_p  )
            stat_m.append( Pk_m  )

        statistics[para] = [stat_p, stat_m]

    return statistics

#-------------------------------------------------------
def load_all_derivatives(root_path, paras, nsims_derivatives, Imax, kmax):
    # initialize
    statistics = {}  # dictionary
    for para in paras:
        statistics[para] = None

    # Load the scaleVLYDF and scalePKHF from EQ, LS, OR_LSS, ...
    for para in paras:
        stat_p   = []
        stat_m   = []
        for i in range(nsims_derivatives):   
            valleys_p = np.load(root_path+para+'_p/valleys/'+str(i)+'/z=0/valleys.npz')
            peaks_p   = np.load(root_path+para+'_p/peaks/'+str(i)+'/z=0/peaks.npz')
            valleys_m = np.load(root_path+para+'_m/valleys/'+str(i)+'/z=0/valleys.npz')
            peaks_m   = np.load(root_path+para+'_m/peaks/'+str(i)+'/z=0/peaks.npz')
            fps_p     = np.load(root_path+para+'_p/fps/'+str(i)+'/z=0/fps.npz')
            fps_m     = np.load(root_path+para+'_m/fps/'+str(i)+'/z=0/fps.npz') 
            if ( i == 0 ):
                k  = fps_p['k']
                kk = k[k<kmax]
            
            vlydf_p   = np.ravel( valleys_p['scaleVLYDF'][0:Imax,:] )
            pkhf_p    = np.ravel( peaks_p['scalePKHF'][0:Imax,:] )
            vlydf_m   = np.ravel( valleys_m['scaleVLYDF'][0:Imax,:] )
            pkhf_m    = np.ravel( peaks_m['scalePKHF'][0:Imax,:] )
            Pk_p      = fps_p['Pk'][0:len(kk)-1]
            Pk_m      = fps_m['Pk'][0:len(kk)-1]

            stat_p.append( np.concatenate((vlydf_p, pkhf_p, Pk_p))  )
            stat_m.append( np.concatenate((vlydf_m, pkhf_m, Pk_m))  )

        statistics[para] = [stat_p, stat_m]

    return statistics