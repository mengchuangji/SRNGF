# import the numpy and pyplot modules

import numpy as np

import matplotlib.pyplot as plot

# Get time values of the signal

time = np.arange(0, 65, .25);

# Get sample points for the discrete signal(which represents a continous signal)

signalAmplitude = np.sin(time)

# plot the signal in time domain

plot.subplot(211)

plot.plot(time, signalAmplitude, 'bs')

plot.xlabel('time')

plot.ylabel('amplitude')

# plot the signal in frequency domain

plot.subplot(212)

# sampling frequency = 4 - get a magnitude spectrum

plot.magnitude_spectrum(signalAmplitude, Fs=4)

# display the plots

plot.show()