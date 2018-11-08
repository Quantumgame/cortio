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

    Use factory methods Cortio.from_file and Cortio.from_vector to create
    a new instance of a Cortio object. Using the class constructor requires
    passing in an AudioStream or VirtualStream object.

    Use `Cortio#stream` to obtain a generator for frames of cortical
    features, or `gulp` to generate the entire (remaining) 4-D vector.

    Use Cortio.transform_file or Cortio.transform_audio to directly output
    the transformation without handling a Cortio instance.
    """

    @staticmethod
    def from_vector(audio, fs, settings = FilterSettings()):
        """Create Cortio instance from audio vector"""
        stream = VirtualStream(audio, fs, chunk_size=settings.chunk_size)
        return Cortio(stream, settings)

    @staticmethod
    def from_file(filepath, settings = FilterSettings()):
        """Create Cortio instance from audio file"""
        stream = AudioStream(filepath, chunk_size=settings.chunk_size)
        return Cortio(stream, settings)

    @staticmethod
    def transform_vector(audio, fs, settings = FilterSettings()):
        """Transform audio vector into cortical representation"""
        cortio = Cortio.from_vector(audio, fs, settings)
        return cortio.gulp()

    @staticmethod
    def transform_file(input_file, output_file, settings = FilterSettings(), format = 'npy'):
        """Transform audio file into cortical representation, write to output file"""
        cortio = Cortio.from_file(input_file, settings)
        cor = cortio.gulp()
        np.save(output_file, cor)

    def __init__(self, audio_streamer, settings = FilterSettings()):
        self.audio_streamer = audio_streamer
        self.cortex = Cortex(settings)
        self.settings = settings
        self.fs = self.audio_streamer.fs
        self._validate()

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

    def _validate(self):
        if self.audio_streamer.info['channels'] != 1: raise Exception, "{} channel audio not supported".format(self.audio_streamer.info['channels'])

# TODO script execution:
#      take input file, write output in some format to file
# if __name__ == "__main__":
#     c = Cortio("data/music1.wav")
#     f = c.transform()
#     print f
