## DEPRECATED

import numpy as np
import wave

from ..signal import distribution
from ..signal import audio
from ..signal import cortical
import gmmdist

def compute_features(chunk,fs,features='cortical',method='moments'):
    if features=='mfcc':
        raise "NIY: requires newer version of scipy.fftpack"
        # f = audio.mfcc(chunk)
        # f[f[:,0] < -320] = -320
        # d = audio.deltas(f)
        # dd = audio.deltas(d)
        # feat = np.concatenate((f,d,dd),axis=1)
        # return(feat)
    elif features=='cortical':
        # convert to specgram
        (X,energy) = wav2aud(chunk,fs)
        # filter specgram to cortical
        cor = aud2cor(X)
        # summarize cortical features
        if method=='gmm':
            f = cor_gmm(cor)
        elif method=='moments':
            f = cor_moments(cor)
        #compute deltas
        f[np.isnan(f)] = 0
        d = audio.deltas(f)
        dd = audio.deltas(d)
        e = energy[:,None]
        #concatenate all features
        feat = np.concatenate((e,f,d,dd),axis=1)
        return(feat)



def wav2cor(wav,fs):
    return aud2cor(wav2aud(wav,fs)[0])

def wav2aud(wav, fs):
    (X, energy) = audio.db_fbank(wav,samplerate=fs,nfilt=128, nfft=1024,winstep=0.01,winlen=0.025)
    # offset to set 0 to mean nothing
    X = X + 150
    X[X<0] = 0
    # apply window
    fade=20
    win = np.ones((1,X.shape[1]))
    win[0,0:fade] = (1+np.arange(fade).astype(np.float))/fade
    win[0,-fade:] = (1+np.arange(fade).astype(np.float)[::-1])/fade
    X = X * win

    return(X,energy)

def aud2cor(aud, fl=10, bp=1):
    cor = cortical.filter(aud,BP=bp,fl=fl)
    #collapse +/- rates
    nr = cor.shape[1]
    cor = np.abs(cor[:,0:nr/2,:,:]) + np.abs(cor[:,nr/2:nr,:,:])
    return cor #np.abs(cor)

def cor_moments(cor):
    cor = np.abs(np.transpose(cor,[2,0,1,3]))
    (NT,NS,NR,NF) = cor.shape
    num_marginals = 4
    features = np.zeros((NT,num_marginals*3))
    for ii, dim in enumerate([1,2,3]):
        # take marginal in dim & time(0)
        marginal = distribution.marginal(cor,[0,dim])
        moments = distribution.moments(marginal, num=4, dim=1, keepdims=True)
        moments[np.isnan(moments)] = 0
        features[:,(0+ii*num_marginals):(num_marginals+ii*num_marginals)] = moments
    return features

def cor_gmm(cor):
    cor = np.abs(cor)
    (NS,NR,NT,NF) = cor.shape
    num_components = 2
    features = np.zeros((NT,num_components*(2*3 + 1)))
    g = gmmdist.GMM(num_components)

    for n in range(NT):
        slice = cor[:,:,n,:]
        g.fit(slice,n_iter = 20)
        features[n,:] = np.hstack((g.model['means'].flatten(),g.model['covars'].flatten(),g.model['weights']))
    return features

def cor_marginals(cor, sum_t = True):
    if cor.ndim==4:
        if sum_t:
            cor = cor.sum(axis=2)
            f_dim = 2
        else:
            f_dim = 3
    else:
        f_dim = 2

    s_dim = 0
    r_dim = 1

    sr = cor.sum(axis=f_dim)
    sf = cor.sum(axis=r_dim)
    rf = cor.sum(axis=s_dim)
    return(sr,sf,rf)
