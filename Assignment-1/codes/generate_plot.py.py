import soundfile as sf

import matplotlib.pyplot as plt

from scipy import signal, fftpack
import numpy as np

#  magnitude of filter upto order n



def plotting_filter(n, Wn):
    c_map = plt.cm.get_cmap('hsv', n)
    for i in range(2,n):
        b, a = signal.butter(i, Wn, 'low', analog=True)
        w, h = signal.freqs(b, a)
        plt.semilogx(w, 20 * np.log10(abs(h)), c = c_map(i-2))
    plt.title('Butterworth filter amplitude ')
    plt.xlabel('freq ')
    plt.ylabel('Amplitude (in dB)')
    plt.xlim([0.01, 1])
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.legend(range(2,n))
    plt.axvline(100, color='blue') # cutoff frequency
    plt.savefig('../figs/es18btech11024_1.eps', format='eps')
    plt.savefig('../figs/es18btech11024_1.png', format='png')
    plt.show()
    return

Wn = 4000/44100.0
plotting_filter(10, Wn)


# Printing the order of values of coefficients
b,a = signal.butter(4, Wn, 'low')
print("\nFor order 4, b values are :{}\n".format(b))

b,a = signal.butter(10, Wn, 'low')
print("For order 10, b values are :{}\n".format(b))
