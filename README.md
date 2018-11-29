## Disclaimer (2018-11-29)

Cortio is in active development, being reworked from its initial state of translated MATLAB code with no concept of good OOP design into a much more developer-friendly library! As such, please do not consider any part of the current API as stable.

## Introduction

Cortio ("Cortical Audio") is an audio processing package motivated by work in auditory neuroscience. Like a short-time Fourier transform on steroids, the primary audio processing algorithm decomposes audio signals into scale, rate, frequency, and time dimensions.

Here, "scale" refers to the structure in the frequency transform (that is, a Fourier decomposition of the frequency transform of an audio frame), and "rate" refers to the structure in time of the short-time Fourier transform)

NOTE: Cortio currently ONLY supports 1-channel, 16kHz audio in PCM format. I'm working on it. Patience.

To see a demo of the features produced, use the `cortical_movie` script to generate a video visualization of the computed features:

```bash
cortical_movie $input_file $output_file
```

where `input_file` must be a 16kHz wav file, and `$output_file` is the desired output video file. Cortio uses ffmpeg for video transcoding, so any format extension supported by ffmpeg will be accepted. When in doubt, use mp4.

## Using Cortio: Creating a Cortio instance

To use the algorithm in Python create a `Cortio` instance using one of the factory methods:

```python
x, sample_rate = load_audio_data(...)
cortio = Cortio.from_vector(x, sample_rate)

file = 'data/speech1.wav'
cortio = Cortio.from_file(file)
```

From here, you can transform the entire audio in one go (this may be memory-expensive depending on the size of your audio -- keep it to a few minutes max!)

```python
y = cortio.gulp()
```

But for most application, you will want to stream the data in chunks:

```python
for cor in cortio.stream():
  # cor is a matrix of cortical audio features
  do_something_with(cor)
```

You can modify settings such as the chunk_size by passing a `cortio.FilterSettings` object to the `Cortio` factory methods.

```python
file = 'data/speech1.wav'
settings = cortio.FilterSettings(chunk_size=15.0, winstep=0.2, winlen=0.4)
cortio = Cortio.from_file(file, settings)
```

Documentation of these filter settings is coming soon(ish). Patience.

## Using Cortio: Static methods

You can also use Cortio a little more opaquely by simply using the following static methods without dealing with a Cortio instance:

```python
x, sample_rate = load_audio_data(...)
y = Cortio.transform_vector(x, sample_rate)

file = 'data/speech1.wav'
y = Cortio.transform_file(file)
```

Both of these methods also support passing in a `cortio.FilterSettings` instance. Both of these methods are as memory-expensive as you audio is large. Tread carefully. Coming sometime: streaming implementation of `transform_file` that uses the file system instead of memory for large transformation. Patience. Or pull requests!
