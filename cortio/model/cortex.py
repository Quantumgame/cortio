import numpy as np
from ..signal import audio

class Cortex:
    def __init__(self, filter_settings):
        self.filter_settings = filter_settings

    def wav2cor(self, wav,fs):
        return self.aud2cor(self.wav2aud(wav,fs)[0])

    def wav2aud(self, wav, fs):
        (X, energy) = audio.db_fbank(
                wav,
                samplerate=fs,
                nfilt=self.filter_settings.nfilt,
                nfft=self.filter_settings.nfft,
                winstep=self.filter_settings.winstep,
                winlen=self.filter_settings.winlen)
        # offset to set 0 to mean nothing
        X = X + 150
        X[X<0] = 0
        # apply window
        fade=20
        win = np.ones((1,X.shape[1]))
        win[0,0:fade] = (1+np.arange(fade).astype(np.float))/fade
        win[0,-fade:] = (1+np.arange(fade).astype(np.float)[::-1])/fade
        X = X * win

        return(X,energy)

    def aud2cor(self, aud):
        cor = self.filter(aud)
        #collapse +/- rates
        nr = cor.shape[1]
        cor = np.abs(cor[:,0:nr/2,:,:]) + np.abs(cor[:,nr/2:nr,:,:])
        return cor #np.abs(cor)


    def filter(self, specgram):
        """
        AUD2COR (forward) cortical rate-scale representation

        AUD2COR implements the 2-D wavelet transform
        possibly executed by the A1 cortex. The auditory
        spectrogram (Y) is the output generated by the
        cochlear model (WAV2AUD) according to the parameter
        set PARA1. RV (SV) is the characteristic frequencies
        (ripples) of the temporal (spatial) filters. This
        function will store the output in a file with a
        conventional extension .COR. Roughly, one-second
        signal needs about 22 MB if 8-ms-frame is adopted.
        Choosing truncated fashion (FULL = 0) will reduce
        the size to 1/4, which will also reduce runing time
        to half.
        See also: WAV2AUD, COR_INFO, CORHEADR, COR_RST

        Output dims: [scale, rate, time, freq]
        """

        K1 = len(self.filter_settings.rates)
        K2 = len(self.filter_settings.scales)
        (N, M) = specgram.shape

        N1 = int(2**np.ceil(np.log2(N)))
        N2 = 2*N1
        M1 = int(2**np.ceil(np.log2(M)))
        M2 = 2*M1

        # 2D FT of specgram to perform rate/scale filter in secgram-freq domain
        Y = np.fft.rfft2(specgram,s=(N2,M2))[:,0:M1]

        STF = 1000.0 / self.filter_settings.fl    # frame per second
        SRF = 24        # channel per octave (fixed)

        # freq. index
        dM   = int(float(M)/2*self.filter_settings.full_X)
        mdx1 = np.hstack((np.arange(dM)+M2-dM, np.arange(M)+dM))

        # temp. index
        dN   = int(float(N)/2*self.filter_settings.full_T)
        ndx  = np.arange(N)+2*dN
        ndx1 = ndx

        z  = np.zeros((N+2*dN, M+2*dM), dtype='complex128')
        cr = np.zeros((K2, K1*2, N+2*dN, M+2*dM), dtype='complex128')

        for rdx in range(K1):
            # rate filtering
            fc_rt = self.filter_settings.rates[rdx]
            HR = self.temporal_filter(fc_rt, N1, STF, [1+rdx+self.filter_settings.bandpass, K1+self.filter_settings.bandpass*2])

            for sgn in (1, -1):
                # rate filtering modification
                if sgn > 0:
                    HR = np.hstack((HR, np.zeros(N1)))
                else:
                    HR = np.hstack( (HR[0], np.conj(HR[N2:0:-1])) )
                    if N2 > 2:
                        HR[N1] = np.abs(HR[N1+1])

                # first inverse fft (w.r.t. time axis)
                z1 = HR[:,None] * Y
                z1 = np.fft.ifft(z1,axis=0)
                z1 = z1[ndx1,:]

                for sdx in range(K2):
                    # scale filtering
                    fc_sc = self.filter_settings.scales[sdx]
                    HS = self.frequency_filter(fc_sc, M1, SRF, [1+sdx+self.filter_settings.bandpass, K2+self.filter_settings.bandpass*2])

                    # second inverse fft (w.r.t frequency axis)
                    z[ndx,:] = np.fft.ifft(z1*HS,axis=1,n=M2)[ndx[:,None],mdx1]
                    cr[sdx, rdx+(sgn==1)*K1, :, :] = z

        return cr


    def temporal_filter(self, fc, L, srt, PASS = [2,3]):
        """Generate (bandpass) cortical filter transfer function
        fc: characteristic frequency
        L: filter length (use power of 2)
        srt: sample rate
        PASS: (vector) [idx K]
        idx = 1, lowpass; 1<idx<K, bandpass; idx = K, highpass.

        GEN_CORT generate (bandpass) cortical temporal filter for various
        length and sampling rate. The primary purpose is to generate 2, 4,
        8, 16, 32 Hz bandpass filter at sample rate ranges from, roughly
        speaking, 65 -- 1000 Hz. Note the filter is complex and non-causal.
        see also: AUD2COR, COR2AUD, MINPHASE
        """

        t = np.arange(L).astype(np.float)/srt
        k = t*fc
        h = np.sin(2*np.pi*k) * k**2 * np.exp(-3.5*k) * fc

        h = h-np.mean(h)
        H0 = np.fft.fft(h, n=2*L)
        A = np.angle(H0[0:L])
        H = np.abs(H0[0:L])
        maxi = np.argmax(H)
        H = H / (H[maxi] or 1)

        # passband
        if PASS[0] == 1:
            #low pass
            H[0:maxi] = 1
        elif PASS[0] == PASS[1]:
            #high pass
            H[maxi+1:L] = 1

        H = H * np.exp(1j*A)
        return H

    def frequency_filter(self, fc, L, srf, KIND=2):
        """
        GEN_CORF generate (bandpass) cortical filter transfer function
        h = gen_corf(fc, L, srf);
        h = gen_corf(fc, L, srf, KIND);
        fc: characteristic frequency
        L: length of the filter, power of 2 is preferable.
        srf: sample rate.
        KIND: (scalar)
              1 = Gabor function; (optional)
              2 = Gaussian Function (Negative Second Derivative) (defualt)
              (vector) [idx K]
              idx = 1, lowpass; 1<idx<K, bandpass; idx = K, highpass.

        GEN_CORF generate (bandpass) cortical filter for various length and
        sampling rate. The primary purpose is to generate 2, 4, 8, 16, 32 Hz
        bandpass filter at sample rate 1000, 500, 250, 125 Hz. This can also
        be used to generate bandpass spatial filter .25, .5, 1, 2, 4 cyc/oct
        at sample ripple 20 or 24 ch/oct. Note the filter is complex and
        non-causal.
        see also: AUD2COR, COR2AUD
        """

        if hasattr(KIND, "__len__"):
            PASS = KIND
            KIND = 2
        else:
            PASS = [2,3]
            KIND = [KIND]

        # fourier transform of lateral inhibitory function

        # tonotopic axis
        R1    = np.arange(L).astype(np.float)/L*srf/2/np.abs(fc)

        if KIND == 1:
            # Gabor function
            C1      = 1./2/0.3/0.3
            H       = np.exp(-C1*(R1-1)**2) + np.exp(-C1*(R1+1)**2)
        else:
            # Gaussian Function
            R1    = R1 ** 2
            H    = R1 * np.exp(1-R1)

        # passband
        if PASS[0] == 1:
            #lowpass
            maxi = np.argmax(H)
            sumH = H.sum()
            H[0:maxi] = 1
            H = H / (H.sum()  or 1) * sumH
        elif PASS[0] == PASS[1]:
            # highpass
            maxi = np.argmax(H)
            sumH = H.sum()
            H[maxi+1:L] = 1
            H = H / (H.sum() or 1) * sumH

        return H
