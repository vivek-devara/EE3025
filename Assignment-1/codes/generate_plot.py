import soundfile as sf
from scipy import signal, fftpack
import numpy as np
import matplotlib.pyplot as plt

# Plots the magnitude of filter upto order n
def plot_filter(n, Wn):
    cmap = plt.cm.get_cmap('hsv', n)
    for i in range(2,n):
        b, a = signal.butter(i, Wn, 'low', analog=True)
        w, h = signal.freqs(b, a)
        plt.semilogx(w, 20 * np.log10(abs(h)), c = cmap(i-2))
    plt.title('Butterworth filter magnitude')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude (in dB)')
    plt.xlim([0.01, 1])
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.legend(range(2,n))
    plt.axvline(100, color='green') # cutoff frequency
    plt.savefig('../figs/ee18btech11051_1.eps', format='eps')
    plt.savefig('../figs/ee18btech11051_1.png', format='png')
    plt.show()
    return

Wn = 4000/44100.0
plot_filter(10, Wn)


# Printing the order of values of coefficients
b,a = signal.butter(4, Wn, 'low')
print("\nFor order 4, the values of b are:{}\n".format(b))

b,a = signal.butter(10, Wn, 'low')
print("For order 10, the values of b are:{}\n".format(b))
