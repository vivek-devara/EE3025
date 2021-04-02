import soundfile as sf

import matplotlib.pyplot as plt

from scipy import signal, fftpack
import numpy as np


input, F_s = sf.read('../soundfiles/Sound_Noise.wav')
sampleFreq = F_s 
print(F_s)

T_S = 1.0/F_s   # T_S = time Period
t = np.arange(0, len(input)*T_S, T_S)

#Filter Order 
order = 4 


def func_fft(input, T_S):
    N = input.shape[0]
    var_FFT = abs(np.fft.fft(input))
    FFT = var_FFT[range(N//2)]
    
     # one side var_FFT range
    dFreq = fftpack.fftfreq(input.size, T_S)
    fft_freqs = np.array(dFreq)
    frequencies_side = dFreq[range(N//2)] 
    
    # one side frequency range
    fft_freqs_side = np.array(frequencies_side)
    return fft_freqs_side, FFT

# cut_frequency =4000.0

#  last peak frequency above the threshold



def find_cutoff(input, T_S, height_cutoff):
    dFreq, original_fft = func_fft(input, T_S)
    peaks = signal.find_peaks(original_fft, height=height_cutoff)
    return dFreq[peaks[0]][-1]

cut_frequency = find_cutoff(input, T_S, 500)
print("New Cutoff freq:{}".format(cut_frequency))

Wn = 2*1.2*cut_frequency/sampleFreq



def apply_filtfilt(input, order, Wn):
    b,a = signal.butter(order, Wn, 'low')
    #print(b,a)
    output_signal = signal.filtfilt(b,a,input)
    return output_signal


# Plot frequency response


plt.subplot(2,1,1)
dFreq, original_fft = func_fft(input, T_S)
plt.plot(dFreq, original_fft, 'r')
plt.xlim([0,6000])

plt.title("INITIAL ")
plt.axvline(cut_frequency, color='black', linestyle='dotted')
plt.text(cut_frequency+30,1000,'Cutoff',rotation=90)

plt.subplot(2,1,2)
filtered = apply_filtfilt(input, order, Wn)

# Applying the filter multiple time .



for i in range(10):
    filtered = apply_filtfilt(filtered, order, Wn)
dFreq, filtered_fft = func_fft(filtered, T_S)
plt.plot(dFreq, filtered_fft, 'b')
plt.xlim([0,6000])
plt.title("Filtered Output ")
plt.axvline(cut_frequency, color='blue', linestyle='dotted')
plt.text(cut_frequency+30,1000,'Cutoff',rotation=90)

plt.savefig('../figs/es18btech11024_freq_result.eps')
plt.savefig('../figs/es18btech11024_freq_result.png')
plt.show()


# Plotting the time response
T = 800
plt.plot(np.arange(T)*T_S, input[:T], 'r', label='Initial Output ')
plt.plot(np.arange(T)*T_S, filtered[:T], 'k', label = 'Filtered Output ')
plt.legend()
plt.axhline(0,color='blue')
plt.axvline(0,color='blue')
plt.xlabel("time(t)")
plt.ylabel("Amplitude A ")
plt.grid()
plt.savefig('../figs/es18btech11024_time_result.eps')
plt.savefig('../figs/es18btech11024_time_result.png')
plt.show()

sf.write('Sound_optimized.wav', filtered, F_s)

# Printing the results

original_beforecutoff = 0
original_aftercutoff = 0
optimized_beforecutoff = 0
optimized_aftercutoff = 0

for i in range(len(dFreq)):
    if dFreq[i]<cut_frequency:
        original_beforecutoff += original_fft[i]
        optimized_beforecutoff += filtered_fft[i]
    else:
        original_aftercutoff += original_fft[i]
        optimized_aftercutoff += filtered_fft[i]

print("Integral from 0 to cutoff frequency  of original signal:  {}".format(round(original_beforecutoff,2)))
print("Integral from 0 to cutoff frequency of filtered signal:  {}".format(round(optimized_beforecutoff,2)))
print("Integral after cutoff frequency of original signal:  {}".format(round(original_aftercutoff,2)))
print("Integral after cutoff frequency  of filtered signal:  {}".format(round(optimized_aftercutoff,2)))
print("Ratio of components after cutoff to before the cutoff of original signal:  {}".format(round(original_aftercutoff/original_beforecutoff,2)))
print("Ratio of components after cutoff to before the cutoff of filtered signal:  {}".format(round(optimized_aftercutoff/optimized_beforecutoff,2)))
