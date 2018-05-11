import wave
import numpy as np

class audiostream:
    """Stream audio from a file in chunks"""
    def __init__(self, file, chunk_size=10.0):
        self.audiofile = wave.open(file)
        self.info = { 'channels': self.audiofile.getnchannels(),
        'samplewidth': self.audiofile.getsampwidth(),
        'framerate': self.audiofile.getframerate(),
        'numframes': self.audiofile.getnframes(),
        'compression_type': self.audiofile.getcomptype()}
        self.fs = self.info['framerate']
        self.chunk_size = chunk_size
        self.chunk_frames = int(float(chunk_size) * self.fs)
        self.streamer = self.stream()

        if self.chunk_frames <= 0: raise Exception, "Invalid chunk size"


    def __iter__(self):
        return(self)

    def rewind(self):
        self.audiofile.rewind()
        self.streamer = self.stream()

    def eof(self):
        return (self.audiofile.tell() >= self.info['numframes'])

    def next(self):
        return(self.streamer.next())

    def stream(self):
        while self.audiofile.tell() < self.info['numframes']:
            bytes = self.audiofile.readframes(self.chunk_frames)
            yield np.frombuffer(bytes,dtype='<i2').astype(np.float)/(2**15-1)

    def close(self):
        self.audiofile.close()

class virtualstream:
    """Virtual AudioStream class for audio data in memory"""
    def __init__(self,x,fs,chunk_size=10.0):
        self.x = x
        self.fs = fs
        self.chunk_size = chunk_size
        self.n = 0
        self.info = { 'channels': 1,
        'samplewidth': None,
        'framerate': fs,
        'numframes': len(x),
        'compression_type': None}
        self.chunk_frames = int(float(self.chunk_size) * self.fs)
        self.streamer = self.stream()

        if self.chunk_frames <= 0: raise Exception, "Invalid chunk size"

    def rewind(self):
        self.n = 0
        self.streamer = self.stream()

    def eof(self):
        return self.n >= self.info['numframes']

    def next(self):
        return self.streamer.next()

    def stream(self):
        while not self.eof():
            n0 = self.n
            self.n = self.n + self.chunk_frames
            yield self.x[n0:n0+self.chunk_frames]
