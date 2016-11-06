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

# This file includes routines to help visualize cortical features

import numpy as np
import math
import matplotlib.pyplot as plot
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as axes3d
import threepy.nonspeech.features as Features
import threepy.signal.audiostream as AudioStream
import time
import sys

def decompose(x,fs):
	(X,e) = Features.wav2aud(x,fs)
	cor = Features.aud2cor(X)
	return(X,cor)	

def analyze(x,fs):
	"""Plot cochleogram (t x f) and cortical features (s x r x f, summed over time)"""
	(X,cor) = decompose(x,fs)
	audfig = plot.figure()
	audfig.gca().imshow(X.transpose(),aspect='auto',origin='lower')
	corfig = plot_cortical(cor)
	return(audfig,corfig)

def cube_marginals(cube, normalize=False):
	"""Return 2D marginals for each of three dimensions of a cube of data"""
	c_fcn = np.mean if normalize else np.sum
	xy = c_fcn(cube, axis=2)
	xz = c_fcn(cube, axis=1)
	yz = c_fcn(cube, axis=0)
	return(xy,xz,yz)

def plot_cortical(cor,**kwargs):
	"""Plot cortical feature cube, summed over time"""
	if cor.ndim == 4: cor = np.mean(cor,axis=2)
	return plot_cube(cor/cor.max(),normalize=True,xlabel='scale',ylabel='rate',zlabel='freq',**kwargs)

def cortical_frame_generator(cor, fig = None, win=None, **kwargs):
	"""Plot one 3D frame of a cortical feature set at a time."""
	if cor.ndim < 4: raise "Must provide full 4-dimensions cortical feature tensor"
	(xy,xz,yz) = cube_marginals((cor/cor.max()).transpose((0,1,3,2)))
	n = cor.shape[2]
	if fig == None: fig = plot.figure()
	ax = fig.gca(projection='3d')
	for ii in np.arange(n):
		yield plot_cube((xy[:,:,ii],xz[:,:,ii],yz[:,:,ii]),normalize=True,xlabel='scale',ylabel='rate',zlabel='freq',ax=ax,**kwargs)

def cortical_t(cor, win=None, step=1, t=5e-3, output=None, **kwargs):
	"""Animate cortical cube over time"""
	frames = cortical_frame_generator(cor, win=win, **kwargs)
	n = cor.shape[2]
	for ii in np.arange(n):
		t0 = time.time()
		frames.next()
		t1 = time.time()
		if (t > t1-t0): plot.pause(t - (t1-t0))

def cortical_movie(file, cor, **kwargs):
	FFMpegWriter = animation.writers['ffmpeg']
	metadata = dict(title='Movie Test', artist='Matplotlib',
		comment='Movie support!')
	writer = FFMpegWriter(fps=15, metadata=metadata)

	fig = plot.figure()
	frames = cortical_frame_generator(cor, fig=fig, **kwargs)
#	l, = plot.plot([], [], 'k-o')

#	plot.xlim(-5, 5)
#	plot.ylim(-5, 5)

#	x0,y0 = 0, 0

	with writer.saving(fig, "writer_test.mp4", 100):
		for frame in frames:
			writer.grab_frame()

		# for i in range(100):
		# 	x0 += 0.1 * np.random.randn()
		# 	y0 += 0.1 * np.random.randn()
		# 	l.set_data(x0, y0)
		# 	writer.grab_frame()

	# fig = plot.figure()
	# movie = animation.FFMpegWriter()
	# with movie.saving(fig,file,30):
	# 	for frame in frames:
	# 		movie.grab_frame()

	# return movie

class CubePlot:
	"""Class for handling creation of cortical cube plots, animations, and movie files"""
	def __init__(self,x,fs=None):

		self.stream = None
		self.x = None
		self.aud = None
		self.cor = None
		self.marginals = None
		self.nX = 0
		self.nY = 0
		self.nZ = 0
		self.N = 0
		self.contours = {}
		self.figure = None
		self.axes = None
		self.fs = None

		if type(x) == str:
			self.stream = AudioStream.audiostream(x)
			self.fs = self.stream.fs
		else:
			if fs == None: raise Exception, "Must provide sample frequency for audio"
			self.stream = AudioStream.virtualstream(x,fs)
			self.fs = fs

		#hacky to get the correct shape
		self.process_audio(np.zeros(100))

	def frames(self, verbose=False):
		self.stream.rewind()
		while not self.stream.eof():
			if verbose: sys.stderr.write("Next audio chunk.\n")
			x = self.stream.next()
			if verbose: sys.stderr.write("Loaded %d samples.\n" % len(x))
			self.process_audio(x)
			if verbose: sys.stderr.write("Processed features\n")

			n = self.cor.shape[2]
			(xy,xz,yz) = self.marginals
			for ii in np.arange(n):
				yield (xy[:,:,ii],xz[:,:,ii],yz[:,:,ii])

	def process_audio(self,x):
		(aud,e) = Features.wav2aud(x,self.fs)
		cor = Features.aud2cor(aud)
		#reduce time resolution
		#TODO: do this better
		cor = cor[:,:,0::5,:]
		self.aud = aud
		self.cor = cor/(cor.max() or 1)
		self.marginals = cube_marginals(self.cor.transpose((0,1,3,2)),normalize=True)
		(self.nX, self.nY, self.N, self.nZ) = cor.shape


	def eof(self):
		return self.stream.eof()

	def setup_plot(self):
		if self.figure != None and plot.fignum_exists(self.figure.number): plot.close(figure)

		# draw wire cube to aid visualization
		self.figure = plot.figure()
		self.axes = self.figure.gca(projection='3d')

		self.axes.plot([0,self.nX-1,self.nX-1,0,0],[0,0,self.nY-1,self.nY-1,0],[0,0,0,0,0],'k-')
		self.axes.plot([0,self.nX-1,self.nX-1,0,0],[0,0,self.nY-1,self.nY-1,0],[self.nZ-1,self.nZ-1,self.nZ-1,self.nZ-1,self.nZ-1],'k-')
		self.axes.plot([0,0],[0,0],[0,self.nZ-1],'k-')
		self.axes.plot([self.nX-1,self.nX-1],[0,0],[0,self.nZ-1],'k-')
		self.axes.plot([self.nX-1,self.nX-1],[self.nY-1,self.nY-1],[0,self.nZ-1],'k-')
		self.axes.plot([0,0],[self.nY-1,self.nY-1],[0,self.nZ-1],'k-')

		self.axes.set_xlabel("scale")
		self.axes.set_ylabel("rate")
		self.axes.set_zlabel("freq")

		# this appears to be necessary because contourf has buggy behavior when the data range is not exactly (0,1)
		self.axes.set_xlim(0,self.nX-1)
		self.axes.set_ylim(0,self.nY-1)
		self.axes.set_zlim(0,self.nZ-1)

		return self

	def draw_contours(self, xy, xz, yz, plot_front=True):
		if self.figure == None or not plot.fignum_exists(self.figure.number): self.setup_plot()

		alpha = 0.95
		cmap=plot.cm.hot

		xy = xy/xy.max()
		xz = xz/xz.max()
		yz = yz/yz.max()

		x = np.arange(self.nX)
		y = np.arange(self.nY)
		z = np.arange(self.nZ)
		offsets = (self.nZ-1,0,self.nX-1) if plot_front else (0, self.nY-1, 0)
		self.contours["xy"] = self.axes.contourf(x[:,None].repeat(self.nY,axis=1), y[None,:].repeat(self.nX,axis=0), xy, zdir='z', offset=offsets[0], cmap=cmap, alpha=alpha)
		self.contours["xz"] = self.axes.contourf(x[:,None].repeat(self.nZ,axis=1), xz, z[None,:].repeat(self.nX,axis=0), zdir='y', offset=offsets[1], cmap=cmap, alpha=alpha)
		self.contours["yz"] = self.axes.contourf(yz, y[:,None].repeat(self.nZ,axis=1), z[None,:].repeat(self.nY,axis=0), zdir='x', offset=offsets[2], cmap=cmap, alpha=alpha)


	def clear_contours(self):
		# # ref: https://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg17614.html
		if self.figure == None or not plot.fignum_exists(self.figure.number): return None
		for plane in self.contours:
			for col in self.axes.collections: #self.contours[plane].collections:
				self.axes.collections.remove(col)

	def plot(self,**kwargs):
		block = kwargs.pop('block') if kwargs.has_key('block') else False
		(xy,xz,yz) = self.marginals
		self.draw_contours(xy.mean(axis=2),xz.mean(axis=2),yz.mean(axis=2))
		plot.show(block=block)

	# def animate(self, **kwargs):
	# 	t = 50e-3
	# 	plot.show(block=False)
	# 	for frame in self.frames():
	# 		t0 = time.time()
	# 		self.clear_contours()
	# 		self.draw_contours(*frame)
	# 		plot.draw()
	# 		t1 = time.time()
	# 		if (t > t1-t0):
	# 			plot.pause(t - (t1-t0))
	# 		else:
	# 			sys.stderr.write("Frame took too long to render in animation!\n")

	def write_movie(self, filename = "cortical.mp4", **kwargs):
		if self.figure == None or not plot.fignum_exists(self.figure.number): self.setup_plot()

		FFMpegWriter = animation.writers['ffmpeg']
		metadata = dict(title='Cortical Features', artist='Matplotlib',
			comment='Cortical features')
		writer = FFMpegWriter(fps=20, metadata=metadata)

		with writer.saving(self.figure, filename, 100):
			for frame in self.frames(verbose=True):
				self.clear_contours()
				self.draw_contours(*frame)
				writer.grab_frame()

class SquarePlot:
	"""Class for handling generation of cortical-marginal plane plots, animations, and movie files"""
	def __init__(self):
		pass

def plot_cube(cube,x=None,y=None,z=None,normalize=False,plot_front=True, alpha=0.9,cmap=plot.cm.hot,xlabel='x',ylabel='y',zlabel='z',ax=None,block=False):
	"""Use contourf to plot cube marginals. Optionally, cube can be passed in as pre-computed marginals"""
	if type(cube) == list or type(cube) == tuple:
		(xy,xz,yz) = cube
		X = xy.shape[0]
		Y = xy.shape[1]
		Z = xz.shape[1]
		if xz.shape[0] != X: raise Exception("Bad shape for marginals")
		if yz.shape[0] != Y: raise Exception("Bad shape for marginals")
		if yz.shape[1] != Z: raise Exception("Bad shape for marginals")
	else:
		(xy,xz,yz) = cube_marginals(cube,normalize=normalize)
		(X,Y,Z) = cube.shape
	if x == None: x = np.arange(X)
	if y == None: y = np.arange(Y)
	if z == None: z = np.arange(Z)
	if ax == None:
		fig = plot.figure()
		ax = fig.gca(projection='3d')
	else:
		fig = ax.figure
		ax.cla()

	# draw edge marginal surfaces
	offsets = (Z-1,0,X-1) if plot_front else (0, Y-1, 0)
	cset_xy = ax.contourf(x[:,None].repeat(Y,axis=1), y[None,:].repeat(X,axis=0), xy, zdir='z', offset=offsets[0], cmap=cmap, alpha=alpha)
	cset_xz = ax.contourf(x[:,None].repeat(Z,axis=1), xz, z[None,:].repeat(X,axis=0), zdir='y', offset=offsets[1], cmap=cmap, alpha=alpha)
	cset_yz = ax.contourf(yz, y[:,None].repeat(Z,axis=1), z[None,:].repeat(Y,axis=0), zdir='x', offset=offsets[2], cmap=cmap, alpha=alpha)

	# draw wire cube to aid visualization
	ax.plot([0,X-1,X-1,0,0],[0,0,Y-1,Y-1,0],[0,0,0,0,0],'k-')
	ax.plot([0,X-1,X-1,0,0],[0,0,Y-1,Y-1,0],[Z-1,Z-1,Z-1,Z-1,Z-1],'k-')
	ax.plot([0,0],[0,0],[0,Z-1],'k-')
	ax.plot([X-1,X-1],[0,0],[0,Z-1],'k-')
	ax.plot([X-1,X-1],[Y-1,Y-1],[0,Z-1],'k-')
	ax.plot([0,0],[Y-1,Y-1],[0,Z-1],'k-')

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)

	# this appears to be necessary because contourf has buggy behavior when the data range is not exactly (0,1)
	ax.set_xlim(0,X-1)
	ax.set_ylim(0,Y-1)
	ax.set_zlim(0,Z-1)

#	plot.draw()

	return (fig, cset_xy, cset_xz, cset_yz)
