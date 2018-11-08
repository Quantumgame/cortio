import numpy as np
from io.audiostream import AudioStream
from io.audiostream import VirtualStream
from model.cortex import Cortex
from model.filter_settings import FilterSettings

class Cortio:
    """Transform audio into cortical feature representation.

    The cortical representation of audio is modeled after A1 auditory cortex
    and is a 4-D vector: scale x rate x time x freq.
    scale: periodicity in the freq domain of the STFT (harmonicity).
    rate: periodicity in the time domain of the STFT (rhythm).
    time: can't stop it.
    freq: preiodicity in the time-domain of the waveform.

    A Cortio instance can be generated from an audio vector or file.
    Use the various static methods to do this (e.g. Cortio.stream_file).
    Creating a new Cortio() using the class constructor requires passing in
    an AudioStream or VirtualStream object.

    Use the `stream` method to obtain a generator for frames of cortical
    features, or `gulp` to generate the entire (remaining) 4-D vector.
    """

    @staticmethod
    def transform_audio(audio, fs, settings = FilterSettings()):
        """Transform full audio waveform into cortical representation"""
        stream = VirtualStream(audio, fs, chunk_size=settings.chunk_size)
        cortio = Cortio(stream, settings)
        return cortio.gulp()

    @staticmethod
    def transform_file(filepath, settings = FilterSettings()):
        """Transform full audio file into cortical representation"""
        stream = AudioStream(filepath, chunk_size=settings.chunk_size)
        cortio = Cortio(stream, settings)
        return cortio.gulp()

    @staticmethod
    def stream_audio(audio, fs, settings = FilterSettings()):
        """Generate a Cortio instance from an audio vector"""
        stream = VirtualStream(audio, fs, chunk_size=settings.chunk_size)
        return Cortio(stream, settings)

    @staticmethod
    def stream_file(filepath, settings = FilterSettings()):
        """Generate a Cortio instance from an audio file"""
        stream = AudioStream(filepath, chunk_size=settings.chunk_size)
        return Cortio(stream, settings)

    def __init__(self, audio_streamer, settings = FilterSettings()):
        self.audio_streamer = audio_streamer
        self.cortex = Cortex(settings)
        self.settings = settings
        self.fs = self.audio_streamer.fs

    def shape(self):
        """Returns shape of cortical slice based on filter settings"""
        # This is a tacky-hacky but gets the job done for now
        # TODO: compute shape directly from filter settings
        seed = np.zeros(100)
        return self.cortex.wav2cor(seed, self.fs).shape

    def stream(self):
        while not self.audio_streamer.eof():
            yield self.cortex.wav2cor(
                    self.audio_streamer.next(),
                    self.fs)

    def gulp(self):
        # TODO: precomupute deimensions
        for (ii, cor_chunk) in enumerate(self.stream()):
            if (ii == 0):
                cor = cor_chunk
            else:
                cor = np.append(cor, cor_chunk, 3)
        return cor

    def rewind(self):
        self.audio_streamer.rewind()

# TODO script execution:
#      take input file, write output in some format to file
# if __name__ == "__main__":
#     c = Cortio("data/music1.wav")
#     f = c.transform()
#     print f
