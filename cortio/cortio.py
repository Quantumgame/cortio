import numpy as np
from io.audiostream import AudioStream
from io.audiostream import VirtualStream
import signal.cortical as cortical
# import wave
# import sigproc as sigproc
# import audioproc as audioproc
# import cortical as cortical

class Cortio:
    """Transform audio into cortical feature representation"""
    def __init__(self, file):
        self.file = file

    def transform(self):
        stream = AudioStream(self.file, 30.0)
        fs = stream.fs
        # TODO: precomupute deimensions
        for (ii, chunk) in enumerate(stream):
            cor_chunk = cortical.wav2cor(chunk,fs)
            if (ii == 0):
                cor = cor_chunk
            else:
                cor = np.append(cor, cor_chunk, 3)
        return cor

if __name__ == "__main__":
    c = Cortio("data/music1.wav")
    f = c.transform()
    print f
