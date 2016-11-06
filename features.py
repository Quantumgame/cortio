# -*- coding: utf-8 -*-
#
# 3Play Media: Cambridge, MA, USA
# support@3playmedia.com
#
# Copyright (c) 2012 3Play Media, Inc.  The following software is the sole and
# exclusive property of 3Play Media, Inc. and may not to be reproduced,
# modified, distributed or otherwise used, without the written approval
# of 3Play Media, Inc.
#
# This software is provided "as is" and any express or implied
# warranties, including but not limited to, an implied warranty of
# merchantability and fitness for a particular purpose are disclaimed.
#
# In no event shall 3Play Media, Inc. be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including but not limited to, procurement or substitute goods or
# services, loss of use, data or profits, or business interruption)
# however caused and on any theory of liability, whether in contract,
# strict liability, or tort (including negligence or otherwise) arising
# in any way out of the use of this software, even if advised of the
# possibility of such damage.

import numpy as np
import wave
import threepy.signal.sigproc as sigproc
import threepy.signal.audioproc as audioproc
import threepy.nonspeech.cortical as cortical
import threepy.nonspeech.gmmdist as gmmdist

def compute_features(chunk,fs,features='cortical',method='moments'):
	if features=='mfcc':
		raise "NIY: requires newer version of scipy.fftpack"
		# f = audioproc.mfcc(chunk)
		# f[f[:,0] < -320] = -320
		# d = audioproc.deltas(f)
		# dd = audioproc.deltas(d)
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
		d = audioproc.deltas(f)
		dd = audioproc.deltas(d)
		e = energy[:,None]
		#concatenate all features
		feat = np.concatenate((e,f,d,dd),axis=1)
		return(feat)



def wav2cor(wav,fs):
	return aud2cor(wav2aud(wav,fs)[0])

def wav2aud(wav, fs):
	(X, energy) = audioproc.db_fbank(wav,samplerate=fs,nfilt=128, nfft=1024,winstep=0.01,winlen=0.025)
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
		marginal = sigproc.marginal(cor,[0,dim])
		moments = sigproc.moments(marginal, num=4, dim=1, keepdims=True)
		moments[np.isnan(moments)] = 0
		features[:,(0+ii*num_marginals):(num_marginals+ii*num_marginals)] = moments
	return features

def cor_gmm(cor):
	cor = np.abs(cor)
	(NS,NR,NT,NF) = cor.shape
	num_components = 2
	features = np.zeros((NT,num_components*(2*3 + 1)))
	g = gmmdist.GMM(num_components)

	for n in xrange(NT):
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
