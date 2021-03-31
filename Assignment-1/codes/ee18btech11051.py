import soundfile as sf
from scipy import signal, fftpack
import numpy as np
import matplotlib.pyplot as plt


input_signal, fs = sf.read('../soundfiles/Sound_Noise.wav')
sampl_freq = fs 
print(fs)
# Time Period
Ts = 1.0/fs
t = np.arange(0, len(input_signal)*Ts, Ts)

# Order of the filter 
order = 4 


def get_fft(input_signal, Ts):
    N = input_signal.shape[0]
    FFT = abs(np.fft.fft(input_signal))
    FFT_side = FFT[range(N//2)] # one side FFT range
    freqs = fftpack.fftfreq(input_signal.size, Ts)
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N//2)] # one side frequency range
    fft_freqs_side = np.array(freqs_side)
    return fft_freqs_side, FFT_side

# Filter cutoff frequency
#cutoff_freq =4000.0

# Find the last peak frequency above specified threshold
def find_cutoff(input_signal, Ts, height_cutoff):
    freqs, original_fft = get_fft(input_signal, Ts)
    peaks = signal.find_peaks(original_fft, height=height_cutoff)
    return freqs[peaks[0]][-1]

cutoff_freq = find_cutoff(input_signal, Ts, 500)
print("New Cutoff frequency:{}".format(cutoff_freq))

Wn = 2*1.2*cutoff_freq/sampl_freq



def apply_filtfilt(input_signal, order, Wn):
    b,a = signal.butter(order, Wn, 'low')
    #print(b,a)
    output_signal = signal.filtfilt(b,a,input_signal)
    return output_signal


# Plotting the frequency response
plt.subplot(2,1,1)
freqs, original_fft = get_fft(input_signal, Ts)
plt.plot(freqs, original_fft, 'r')
plt.xlim([0,6000])
plt.title("Original")
plt.axvline(cutoff_freq, color='black', linestyle='dotted')
plt.text(cutoff_freq+30,1000,'Cutoff',rotation=90)

plt.subplot(2,1,2)
filtered = apply_filtfilt(input_signal, order, Wn)

# Applying the filter multiple times for better results
for i in range(10):
    filtered = apply_filtfilt(filtered, order, Wn)
freqs, filtered_fft = get_fft(filtered, Ts)
plt.plot(freqs, filtered_fft, 'b')
plt.xlim([0,6000])
plt.title("Filtered")
plt.axvline(cutoff_freq, color='black', linestyle='dotted')
plt.text(cutoff_freq+30,1000,'Cutoff',rotation=90)

plt.savefig('../figs/ee18btech11051_freq_result.eps')
plt.savefig('../figs/ee18btech11051_freq_result.png')
plt.show()


# Plotting the time response
T = 800
plt.plot(np.arange(T)*Ts, input_signal[:T], 'r', label='Original')
plt.plot(np.arange(T)*Ts, filtered[:T], 'k', label = 'Filtered')
plt.legend()
plt.axhline(0,color='black')
plt.axvline(0,color='black')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.savefig('../figs/ee18btech11051_time_result.eps')
plt.savefig('../figs/ee18btech11051_time_result.png')
plt.show()

sf.write('../soundfiles/Sound_optimized.wav', filtered, fs)

# Printing the results

original_beforecutoff = 0
original_aftercutoff = 0
optimized_beforecutoff = 0
optimized_aftercutoff = 0

for i in range(len(freqs)):
    if freqs[i]<cutoff_freq:
        original_beforecutoff += original_fft[i]
        optimized_beforecutoff += filtered_fft[i]
    else:
        original_aftercutoff += original_fft[i]
        optimized_aftercutoff += filtered_fft[i]

print("Integral from 0 to cutoff of original signal:  {}".format(round(original_beforecutoff,2)))
print("Integral from 0 to cutoff of filtered signal:  {}".format(round(optimized_beforecutoff,2)))
print("Integral after cutoff of original signal:  {}".format(round(original_aftercutoff,2)))
print("Integral after cutoff of filtered signal:  {}".format(round(optimized_aftercutoff,2)))
print("Ratio of components after cutoff to before the cutoff of original signal:  {}".format(round(original_aftercutoff/original_beforecutoff,2)))
print("Ratio of components after cutoff to before the cutoff of filtered signal:  {}".format(round(optimized_aftercutoff/optimized_beforecutoff,2)))
