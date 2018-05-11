# This file includes routines for basic audio signal processing including framing and computing power spectra.
# Contributors:
# - James Lyons 2012
# - Andrew Schwartz 2014

import numpy as np
import scipy as sp
import math
import scipy.signal.filter_design as fd

def envelope(x,fs,hp=100, axis=-1):
    Wp = 2*float(hp)/fs
    Ws = 3*float(hp)/fs
    Rp = 1
    As = 50
    b,a = fd.iirdesign(Wp, Ws, Rp, As, ftype='ellip')
    env = sp.signal.filtfilt(b,a,np.abs(x),axis=axis)
    return env

def moments(data, num, dim=0, normalized=False,keepdims=False):
    """Compute up to the num-th moment of the distribution data along specified axis (default 0)"""
    # to make things easier, swap dim into position 0 here, then swap it back if keepDims is True
    if dim != 0:
        data = data.swapaxes(0,dim)
    if not normalized:
        data = data / data.sum(axis=0)[None,...]
    # create support vector along (0) dim
    xshape = np.ones(data.ndim)
    n = data.shape[0]
    xshape[0] = n
    x = np.arange(n).reshape(xshape.astype(int))
    # output shape is same as input shape, but size along dim 0 gets replaced with num_moments
    mshape = np.array(data.shape)
    mshape[0] = num
    m = np.zeros(mshape)
    # first moments are means and var
    if num > 0:
        m[0] = (data * x).sum(axis=0)
    if num > 1:
        m[1] = (data * (x - m[0])**2).sum(axis=0)
    # beyond var, return normalized moments
    for ii in xrange(2,num):
        m[ii] = (data * (x - m[0])**(ii+1)).sum(axis=0) / m[1]**(float((ii+1))/2)
    if keepdims:
        m = m.swapaxes(0,dim)
    return m

def marginal(data, dim):
    """Return marginal of data, varying along dim (can be scalar or array of dims)"""
    dims = np.array([dim]).flatten()
    m = data
    for d in np.arange(data.ndim-1,0-1,-1):
        if d in dims:
            continue
        m = m.sum(axis=d)
    return m

def index_coordinate_matrix(shape):
    """Generate a nd matrix of coordinate vectors"""
    ind = [slice(0,n) for n in shape]
    return np.mgrid[ind].transpose(np.roll(np.arange(len(shape)+1),-1))

def framesig(sig,frame_len,frame_step,winfunc=lambda x:np.ones((1,x))):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(frame_len)
    frame_step = int(frame_step)
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))

    padlen = int((numframes-1)*frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig,zeros))

    indices = np.tile(np.arange(0,frame_len),(numframes,1)) + np.tile(np.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = np.array(indices,dtype=np.int32)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len),(numframes,1))
    return frames*win


def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:np.ones((1,x))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round(frame_len)
    frame_step = round(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = np.tile(np.arange(0,frame_len),(numframes,1)) + np.tile(np.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = np.array(indices,dtype=np.int32)
    padlen = (numframes-1)*frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = np.zeros((1,padlen))
    window_correction = np.zeros((1,padlen))
    win = winfunc(frame_len)

    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 #add a little bit so it is never zero
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]

    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]

def magspec(frames,NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = np.fft.rfft(frames,NFFT)
    return np.absolute(complex_spec)

def powspec(frames,NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0/NFFT * np.square(magspec(frames,NFFT))

def logpowspec(frames,NFFT,norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 1.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames,NFFT);
    ps[ps<=1e-30] = 1e-30
    lps = 10*np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps

def runcor(x, sig, mode='full', noise_floor=None):
    """Compute a running normalized correlation between a token x and a longer signal.
    Normalization is unique to each windowed comparison.

    """
    N = len(x)
    x = x - np.mean(x)
    ym = sp.signal.convolve(sig, np.ones(N), mode=mode)/N
    y2m = sp.signal.convolve(sig**2, np.ones(N), mode=mode)/N
    yy = (y2m - ym**2)
    yy[yy<0] = 0
    denom = np.sqrt(N * np.sum(x**2) * yy)
    if noise_floor != None: denom[denom<noise_floor] = noise_floor
    r = sp.signal.correlate(sig, x, mode=mode) / denom
    return r

def segcor(x, sig, winlen, step=None):
    """Segment x into windows and do a runcor on each window into sig"""
    if step==None: step = int(winlen/2)
    frames = framesig(x,winlen,step)
    N = sig.shape[-1]
    Nx = x.shape[-1]
    num_frames = int(math.ceil(float(Nx - winlen)/step))
    N_out = N+winlen-1+(num_frames-1)*step
    r = np.zeros((num_frames,N_out))
    offset = (num_frames-1)*step
    for ii in np.arange(num_frames):
        r[ii,offset:offset+N+winlen-1] = runcor(frames[ii],sig)
        offset = offset - step
    return r

def segcor_offsets(x,sig,winlen,step=None):
    if step==None: step = int(winlen/2)
    N = sig.shape[-1]
    Nx = x.shape[-1]
    num_frames = int(math.ceil(float(Nx - winlen)/step))
    m = np.arange(-(winlen-1)-(num_frames-1)*step, N)
    return m

def z_mean(x,axis=0,std_axis=None):
    """Take weighted mean of vectors, where weights are inverse stddev"""
    #TODO: really think about zero stddev cases
    if x.ndim==1: x = x[None,:]
    if std_axis == None:
        if axis == x.ndim-1 or axis == -1:
            std_axis=-2
        else:
            std_axis=-1
        #no keepdims in old numpy version
        #stds = np.std(x,axis=std_axis,keepdims=True)
        dimifier = [slice(None)] * x.ndim
        dimifier[std_axis] = None
        stds = np.std(x,axis=std_axis)[dimifier]
    if any(stds==0):
        min_std = min(stds);
        if min_std == 0: min_std = 1
        stds[stds==0] = min_std / 10

    return (x/stds).sum(axis=axis) / (1/stds).sum(axis=axis)
