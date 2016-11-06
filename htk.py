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

import struct
import numpy as np

BYTES_IN_HEADER = 12
HEADER_FORMAT = '>iihh'
BYTES_PER_FEATURE = 4

class htkfile:
	"""Interface to an htk file.
	h = Htk.htkfile(filename)

	h.write_features(f) write each row of f to the file
	h.update_header(numSamples, samplePeriod, numFeatureRows) can be used at any time,
	which is useful if you don't know ahead of time some of this header info
	Don't forget to h.close() !
	see write_htk.py for a full example
	"""

	def __init__(self, filename, mode='rb', nSamples=0, samplePeriod=0, nFeatures=0):
		if not 'b' in mode:
			raise "HTK file must be opened with a binary mode (e.g. wb or rb+)"
		self.file = open(filename, mode)
		self.mode = mode

		if 'r' in mode:
			self.read_header()
			self.rewind()
		else:
			self.nSamples = nSamples
			self.samplePeriod = samplePeriod
			self.nFeatures = nFeatures
			self.write_header(nSamples, samplePeriod, nFeatures)

	def write_features(self,features):
		s = ''
		for f in features:
			s += struct.pack('>f', f)
		if 'r' in self.mode:
			self.file.seek(self.file.tell())
		self.file.write(s)
		return len(s)

	def update_header(self, *args, **kwargs):
		pos = self.file.tell()
		self.write_header(*args,**kwargs)
		self.file.seek(pos)

	def write_header(self, nSamples, samplePeriod, nFeatures):
		self.file.seek(0)
		h = struct.pack(
          HEADER_FORMAT,
          nSamples,
          samplePeriod,
          BYTES_PER_FEATURE*nFeatures,
          9) # user features
		assert len(h) == BYTES_IN_HEADER
		self.file.write(h)

	def read_header(self):
		"""Read header data, update class attributes, don't change file pointer position
		"""
		pos = self.file.tell()
		self.file.seek(0)
		bytes = self.file.read(BYTES_IN_HEADER)
		(nSamples, samplePeriod, nBytes, type) = struct.unpack(HEADER_FORMAT,bytes)
		self.nSamples = nSamples
		self.samplePeriod = samplePeriod
		self.nFeatures = nFeatures = nBytes/BYTES_PER_FEATURE
		self.file.seek(pos)
		return (nSamples, samplePeriod, nFeatures)

	def rewind(self):
		self.file.seek(BYTES_IN_HEADER)

	def tell(self):
		return self.file.tell()

	def read_frame(self):
		return (self.read_frames(n=1))[0]

	def read_frames(self, n=1):
		frames = np.empty((n,self.nFeatures))
		for ii in np.arange(n):
			bytes = self.file.read(self.nFeatures*BYTES_PER_FEATURE)
			frames[ii] = struct.unpack('>'+'f'*self.nFeatures,bytes)
		return frames

	def close(self):
		self.file.close()

	def flush(self):
		self.file.flush()
