class FilterSettings:
    """Settings for cortical filter
    chunk_size : size of audio chunks to analyze
    nfft     : num points for DFT
    nfilt    : number of frequency filters
    winlen   : length of analysis window
    winstep  : time between analysis window starts
    highfreq : highest frequency to include (None for Nyquist)
    preemph  : preemphasis constant
    logfloor : floor for log operations (avoid/bound -Inf)
    fl       : frame length in ms
    full_T   : fullnss of temporal margin, between [0,1]
    full_X   : fullness of spectral margin, between [0,1]
    bandpass : pure bandpass indicator
    rates    : rate vector in Hz, e.g., 2.^(1:.5:5).
    scales   : scale vector in cyc/oct, e.g., 2.^(-2:.5:3)
    """

    # TODO rename and document these params
    def __init__(self,
            chunk_size = 30.0,
            winlen=0.025,
            winstep=0.01,
            nfilt=128,
            nfft=1024,
            lowfreq=0,
            highfreq=None,
            preemph=0.97,
            logfloor = 1e-16,
            rates=[1, 2, 4, 8, 16, 32],
            scales=[0.5, 1, 2, 4, 8],
            fl=10,
            full_T=0,
            full_X=0,
            bandpass=1
            ):
        self.chunk_size = chunk_size
        self.winlen = winlen
        self.winstep = winstep
        self.nfilt = nfilt
        self.nfft = nfft
        self.lowfreq = lowfreq
        self.highfreq = highfreq
        self.preemph = preemph
        self.logfloor = logfloor
        self.rates = rates
        self.scales = scales
        self.fl = fl
        self.full_T = full_T
        self.full_X = full_X
        self.bandpass = bandpass
