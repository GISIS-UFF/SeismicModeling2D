import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

audio = 'C:/Users/juanp/Downloads/Áudio.wav'
sinal,samplerate = sf.read(audio)
tempo = np.arange(0,len(sinal) * 1/samplerate, 1/samplerate)

plt.plot(tempo,sinal)
plt.title('Sinal de Áudio no Domínio do Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

fft_result = np.fft.rfft(sinal)

amplitudes = np.abs(fft_result) / len(sinal)
frequencias = np.fft.rfftfreq(len(sinal), 1 / samplerate)

plt.figure(figsize=(14, 5))
plt.stem(frequencias, amplitudes)
plt.title('Espectro de Amplitude')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

#cutoff_freq = 1000
#filtro = np.abs(frequencias) < cutoff_freq
#filtro_indices = np.where(np.abs(frequencias) > cutoff_freq)
#fft_result[filtro_indices] = 0

f1 = 400
f2 = 1000
f3 = 200
f4 = 1200
filtro_indices1 = np.where((np.abs(frequencias) < f3) & (np.abs(frequencias) > f4))
filtro_indices2 = np.where((np.abs(frequencias) >= f3) & (np.abs(frequencias) < f1))
filtro_indices3 = np.where((np.abs(frequencias) >= f1) & (np.abs(frequencias) <= f2))
filtro_indices4 = np.where((np.abs(frequencias) > f2) & (np.abs(frequencias) <= f4))

fft_result[filtro_indices1] = 0
fft_result[filtro_indices2] = np.linspace(0,1,np.sum((frequencias >= f3) & (frequencias < f1)))
fft_result[filtro_indices3] = 1
fft_result[filtro_indices4] = np.linspace(1,0,np.sum((frequencias > f2) & (frequencias <= f4)))

sinal_reconstruido = np.fft.irfft(fft_result)

plt.figure(figsize=(14, 5))
plt.plot(frequencias, np.abs(fft_result))
plt.title('Espectro de Amplitude Filtrado')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(tempo, sinal_reconstruido)
plt.title('Sinal de Áudio Filtrado no Domínio do Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

