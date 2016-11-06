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
from threepy.signal import sigproc
import sys

class GMM:
	"""Gaussian Mixture Models for distribution data.

	Fit GMMs to distribution data (not discrete data points)"""

	def __init__(self, num_components=1, thresh=1e-8, n_iter=100):
		self.num_components = num_components
		self.model = {'weights': None, 'means': None, 'covars': None}
		self.fit_params = {'iterations': n_iter, 'threshold': thresh}
		self.model_score = -np.inf

	# generate initial means, coavrs, weights
	# primitive for now: even spread across feature space
	def init_EM(self, d):
		"""Generate initial model params (means, covars, weights)
		"""
		means = np.zeros((self.num_components,d.ndim))
		covars = np.ones((self.num_components,d.ndim))
		for ii in xrange(d.ndim):
			means[:,ii] = np.linspace(0,d.shape[ii],self.num_components)
			covars[:,ii] = np.ones(self.num_components).astype(np.float) * np.array(d.shape[ii]).astype(np.float)/4
		weights = 1./self.num_components * np.ones(self.num_components)
		self.model = {'weights': weights, 'means': means, 'covars': covars}
		return(self.model)

	def fit(self, d, verbose=False, thresh=None, n_iter=None):
		if n_iter == None:
			n_iter = self.fit_params['iterations']
		if thresh == None:
			thresh = self.fit_params['threshold']

		d = d/d.sum()
		d_axes = tuple(range(d.ndim))
		b = sigproc.index_coordinate_matrix(d.shape).astype(np.float)
		self.init_EM(d)

		for nn in xrange(n_iter):
			if verbose:
				sys.stdout.write('.')

			# E step
			w = _mv_gaussian_diag(b,means=self.model['means'],covars=self.model['covars']) * self.model['weights']
			denom = w.sum(axis=-1)
			k = denom>0
			for ii in xrange(self.num_components):
				w[...,ii][k] = w[...,ii][k]/denom[k]
				w[...,ii][~k] = 1./self.num_components

			# Check for convergence.
			score = self.score_model(d,p=denom/denom.sum(d_axes))
			if nn > 0 and score < 1e-8 and abs(self.model_score - score) < thresh:
				break
			self.model_score = score

			# M step
			self.model['weights'] = np.sum(w*d[...,None], axis=d_axes)
			for ii in xrange(d.ndim):
				# collapse data to this dimension to compute means and covars
				#TODO: test this; just fixed up the code to make it work with an earlier version of python
				# collapse_axes = tuple(set(d_axes) - {ii})
				# rwd = (w * d[...,None]).sum(axis=collapse_axes)
				rwd = sigproc.marginal(w * d[...,None], ii)

				rwd = rwd / rwd.sum(0)
				rb = np.arange(d.shape[ii])[:,None]
				self.model['means'][:,ii] = (rwd * rb).sum(axis=0)
				self.model['covars'][:,ii] = ((rb - self.model['means'][:,ii])**2 * rwd).sum(axis=0)

		else:
			# might raise an error here for not being converged?
			pass

		return score

	def score_model(self,d,p=None):
		"""Score the specified distribution
		This is just the average Euclidian distance between the distribution values
		Note that kl-divergence would be terrible, since the empirical distribution may 
		very well contain zeros
		TODO: weighted probability sums over the sampled space?
		"""
		if p==None:
			p = self.generate_distribution(d.shape)
		return (((p-d)**2).sum())**0.5 / p.size

	def generate_distribution(self, shape, means=None, covars=None, weights=None):
		if means == None:
			means = self.model['means']
		if covars == None:
			covars = self.model['covars']
		if weights == None:
			weights = self.model['weights']

		means = np.array(means).astype(np.float)
		covars = np.array(covars).astype(np.float)
		weights = np.array(weights).astype(np.float)

		b = sigproc.index_coordinate_matrix(shape).astype(np.float)
		return (_mv_gaussian_diag(b,means,covars) * weights).sum(-1)

#########################################################################
## some helper routines
#########################################################################

def _mv_gaussian_diag(X, means, covars):
	"""Compute Gaussian density at X for a diagonal model
	Assumes means and covars arguments are numpy arrays
	"""
	n_dim = X.shape[-1]
	X = X[...,None,:]
	pr = (2 * np.pi)**(-0.5*n_dim) * covars.prod(-1) ** (-0.5) * np.exp(-np.sum((X - means) * (X - means) / covars, axis=-1))
	return pr

def _log_mv_gaussian_diag(X,means,covars):
	"""Compute multivariate normal pdf for vectors in X
	X can be any number of dimensions. The first (n-1) dimensions
	are simply treated as multiple samples. The last dimension
	corresponds to the dimensionality of the space in which
	the normal density is evalutated
	In implementation, dim -2 is reserved to represent the different
	mixture components; other dimensions are shifted behind this.
	The output collapses dim -1, so dim -2 becomes the new dim -1
	"""
	n_dim = X.shape[-1]
	X = X[...,None,:]
	lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.log(covars).sum(-1) + np.sum((X - means) * (X - means) / covars, axis=-1))
	return lpr


