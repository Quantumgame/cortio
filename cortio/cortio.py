import numpy as np
from io.audiostream import AudioStream
from io.audiostream import VirtualStream
import signal.cortical as cortical
from instance.filter_settings import FilterSettings
# import wave
# import sigproc as sigproc
# import audioproc as audioproc
# import cortical as cortical

class Cortio:
    """Transform audio into cortical feature representation"""

    @staticmethod
    def transform_audio(audio, fs, settings = FilterSettings()):
        stream = VirtualStream(audio, fs)
        cortio = Cortio(stream, settings)
        return cortio.gulp()

    @staticmethod
    def transform_file(filepath, settings = FilterSettings()):
        stream = AudioStream(filepath)
        cortio = Cortio(stream, settings)
        return cortio.gulp()

    @staticmethod
    def stream_audio(audio, fs, settings = FilterSettings()):
        stream = VirtualStream(audio, fs)
        return Cortio(stream, settings)

    @staticmethod
    def stream_file(filepath, settings = FilterSettings()):
        stream = AudioStream(filepath)
        return Cortio(stream, settings)

    def __init__(self, audio_streamer, settings = FilterSettings()):
        self.audio_streamer = audio_streamer
        self.settings = settings
        self.fs = self.audio_streamer.fs

    def shape(self):
        """Returns shape of cortical slice based on filter settings"""
        # This is a tacky-hacky but gets the job done for now
        # TODO: compute shape directly from filter settings
        seed = np.zeros(100)
        return cortical.wav2cor(seed, self.fs).shape

    def stream(self):
        while not self.audio_streamer.eof():
            yield cortical.wav2cor(
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
