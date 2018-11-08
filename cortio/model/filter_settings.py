class FilterSettings:
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
            tc=8,
            fac=-2,
            shift=0,
            full_T=0,
            full_X=0,
            BP=1
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
        self.tc = tc
        self.fac = fac
        self.shift = shift
        self.full_T = full_T
        self.full_X = full_X
        self.BP = BP
