import numpy as np
from nitime import utils
from nitime import algorithms as alg
from scipy import interpolate
from scipy.linalg import toeplitz
from scipy.linalg import eigh,eig,svd

def standardize(x):
    if np.any(np.isnan(x)):
        x_ex = x[np.logical_not(np.isnan(x))]
        xm = np.mean(x_ex)
        xs = np.std(x_ex,ddof=1)
    else:
        xm = np.mean(x)
        xs = np.std(x,ddof=1)
    xstd = (x - xm)/xs
    return xstd

def mssa(data,M,MC=1000,f=0.3):
    '''Multi-channel SSA analysis
    (applicable for data including missing values)
    and test the significance by Monte-Carlo method

    Input:
    data: multiple time series (dimension: length of time series x total number of time series)
    M: window size
    MC: Number of iteration in the Monte-Carlo process
    nmode: The number of modes to extract
    f: fraction (0<f<=1) of good data points for identifying
    significant PCs [f = 0.3]

    Output:
    deval : eigenvalue spectrum
    q05: The 5% percentile of eigenvalues
    q95: The 95% percentile of eigenvalues
    PC      : matrix of principal components
    RC      : matrix of RCs (nrec,N,nrec*M) (only if K>0)
    '''

    #Xr = standardize(data)

    N = len(data[:,0])
    nrec = len(data[0,:])

    Y = np.zeros((N-M+1,nrec*M))
    for irec in np.arange(nrec):
        for m in np.arange(0,M):
            Y[:,m+irec*M]=data[m:N-M+1+m,irec]
    C = np.dot(np.nan_to_num(np.transpose(Y)),np.nan_to_num(Y))/(N-M+1)

    eig_val, eig_vec = eigh(C)

    sort_tmp = np.sort(eig_val)
    deval = sort_tmp[::-1]
    sortarg = np.argsort(-eig_val)

    eig_vec = eig_vec[:,sortarg]

    # test the signifiance using Monte-Carlo
    Ym = np.zeros((N-M+1,nrec*M))
    noise = np.zeros((nrec,N,MC))
    for irec in np.arange(nrec):
        noise[irec,0,:] = data[0,irec]
    Lamda_R = np.zeros((nrec*M,MC))
    # estimate coefficents of ar1 processes, and then generate ar1 time series (noise)
    for irec in np.arange(nrec):
        Xr = data[:,irec]
        coefs_est, var_est = alg.AR_est_YW(Xr[~np.isnan(Xr)], 1)
        sigma_est = np.sqrt(var_est)

        for jt in range(1,N):
            noise[irec,jt,:] = coefs_est*noise[irec,jt-1,:]+ sigma_est*np.random.randn(1,MC)

    for m in range(MC):
        for irec in np.arange(nrec):
            noise[irec,:,m] = (noise[irec,:,m] - np.mean(noise[irec,:,m]))/(np.std(noise[irec,:,m],ddof=1))
            for im in np.arange(0,M):
                Ym[:,im+irec*M]=noise[irec,im:N-M+1+im,m]
        Cn = np.dot(np.nan_to_num(np.transpose(Ym)),np.nan_to_num(Ym))/(N-M+1)
        #Lamda_R[:,m] = np.diag(np.dot(np.dot(eig_vec,Cn),np.transpose(eig_vec)))
        Lamda_R[:,m] = np.diag(np.dot(np.dot(np.transpose(eig_vec),Cn),eig_vec))

    q95 = np.percentile(Lamda_R,95,axis=1)
    q05 = np.percentile(Lamda_R,5,axis=1)

    #modes = np.arange(nmode)

    # determine principal component time series
    PC=np.zeros((N-M+1,nrec*M))
    PC[:,:]=np.nan
    for k in np.arange(nrec*M):
        for i in np.arange(0,N-M+1):
            #   modify for nan
            prod=Y[i,:]*eig_vec[:,k]
            ngood=sum(~np.isnan(prod))
            #   must have at least m*f good points
            if ngood>=M*f:
                PC[i,k]=sum(prod[~np.isnan(prod)]) # the columns of this matrix are Ak(t), k=1 to M (T-PCs)

    # compute reconstructed timeseries
    Np=N-M+1

    RC=np.zeros((nrec,N,nrec*M))

    for k in np.arange(nrec):
        for im in np.arange(M):
            x2 = np.dot(np.expand_dims(PC[:,im],axis=1),np.expand_dims(eig_vec[0+k*M:M+k*M,im],axis=0))
            x2 = np.flipud(x2)

            for n in np.arange(N):
                RC[k,n,im] = np.diagonal(x2,offset=-(Np-1-n)).mean()

    return deval,eig_vec,q95,q05,PC,RC

def ssa_all(data,M,MC=1000,f=0.3):
    '''SSA analysis for a time series
    (applicable for data including missing values)
    and test the significance by Monte-Carlo method

    Input:
    data: time series
    M: window size
    MC: Number of iteration in the Monte-Carlo process
    nmode: The number of modes to extract
    f: fraction (0<f<=1) of good data points for identifying
    significant PCs [f = 0.3]

    Output:
    deval : eigenvalue spectrum
    q05: The 5% percentile of eigenvalues
    q95: The 95% percentile of eigenvalues
    PC      : matrix of principal components
    RC      : matrix of RCs (N*M, nmode) (only if K>0)
    '''

    from nitime import utils
    from nitime import algorithms as alg

    Xr = standardize(data)
    N = len(data)

    c=np.zeros(M)

    for j in range(M):
        prod=Xr[0:N-j]*Xr[j:N]
        c[j]=sum(prod[~np.isnan(prod)])/(sum(~np.isnan(prod))-1)

    C=toeplitz(c[0:M])

    eig_val, eig_vec = eigh(C)

    sort_tmp = np.sort(eig_val)
    deval = sort_tmp[::-1]
    sortarg = np.argsort(-eig_val)

    eig_vec = eig_vec[:,sortarg]

    coefs_est, var_est = alg.AR_est_YW(Xr[~np.isnan(Xr)], 1)
    sigma_est = np.sqrt(var_est)

    noise = np.zeros((N,MC))
    noise[0,:] = Xr[0]
    Lamda_R = np.zeros((M,MC))

    for jt in range(1,N):
        noise[jt,:] = coefs_est*noise[jt-1,:]+ sigma_est*np.random.randn(1,MC)

    for m in range(MC):
        noise[:,m] = (noise[:,m] - np.mean(noise[:,m]))/(np.std(noise[:,m],ddof=1))
        Gn = np.correlate(noise[:,m],noise[:,m],"full")
        lgs = np.arange(-N+1,N)
        Gn = Gn/(N-abs(lgs))
        Cn=toeplitz(Gn[N-1:N-1+M])
        #Lamda_R[:,m] = np.diag(np.dot(np.dot(eig_vec,Cn),np.transpose(eig_vec)))
        Lamda_R[:,m] = np.diag(np.dot(np.dot(np.transpose(eig_vec),Cn),eig_vec))

    q95 = np.percentile(Lamda_R,95,axis=1)
    q05 = np.percentile(Lamda_R,5,axis=1)
    #modes = np.arange(nmode)

    # determine principal component time series
    PC=np.zeros((N-M+1,M))
    PC[:,:]=np.nan
    for k in np.arange(M):
        for i in np.arange(0,N-M+1):
            #   modify for nan
            prod=Xr[i:i+M]*eig_vec[:,k]
            ngood=sum(~np.isnan(prod))
            #   must have at least m*f good points
            if ngood>=M*f:
                PC[i,k]=sum(prod[~np.isnan(prod)])*M/ngood # the columns of this matrix are Ak(t), k=1 to M (T-PCs)

    # compute reconstructed timeseries
    Np=N-M+1

    RC=np.zeros((N,M))

    for im in np.arange(M):
        x2 = np.dot(np.expand_dims(PC[:,im],axis=1),np.expand_dims(eig_vec[0:M,im],axis=0))
        x2 = np.flipud(x2)

        for n in np.arange(N):
            RC[n,im] = np.diagonal(x2,offset=-(Np-1-n)).mean()

    return deval,eig_vec,q05,q95,PC,RC
