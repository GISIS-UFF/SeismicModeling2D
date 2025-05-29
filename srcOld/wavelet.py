import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
t_min = -0.05
t_max = 0.05
num = int(t_max/dt)
t = np.linspace(t_min,t_max,num)
f = 30
n = 2
sigma = n / (2 * np.pi * f)
ricker = (1 -(t/sigma)**2)*np.exp(-(t)**2/(2*sigma**2))

#morlet = np.pi**(-1/4) * np.exp(-(t ** 2) /2 ) * (np.exp(1j*sigma*t) - np.exp(-(sigma ** 2) /2)) #constante de normalização igual a 1

def dft(função):
    N = len(função)
    dft_result = []
    for k in range(N):
        real = sum(função[n] * np.cos(2 * np.pi * k * n / N) for n in range(N))
        imag = sum(-função[n] * np.sin(2 * np.pi * k * n / N) for n in range(N))
        amplitude = np.sqrt(real ** 2 + imag ** 2) / N
        dft_result.append(amplitude)
    return dft_result

def dft_real_imag(função):
    N = len(função)
    dft_real = []
    dft_imag = []
    for k in range(N):
        real = sum(função[n] * np.cos(2 * np.pi * k * n / N) for n in range(N))
        imag = sum(-função[n] * np.sin(2 * np.pi * k * n / N) for n in range(N))
        dft_real.append(real / N)
        dft_imag.append(imag / N)
    return dft_real, dft_imag

dft_real, dft_imag = dft_real_imag(ricker)
dft_result = dft(ricker)

frequencias = []
for i in range(len(t)):
    frequencias.append(i/(t[-1] - t[0]))

frequencias_positivas = []
for i in range (len(frequencias)//2):
    frequencias_positivas.append(frequencias[i])

dft_real_positivas = []
for i in range (len(dft_real)//2):
    dft_real_positivas.append(dft_real[i])

dft_imag_positivas = []
for i in range(len(dft_imag) // 2):
    dft_imag_positivas.append(dft_imag[i])

fig, ax = plt.subplots(3, 1, figsize=(12, 12))

ax[0].plot(t, ricker)
ax[0].set_xlabel('Tempo')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)

ax[1].stem(frequencias_positivas, dft_real_positivas)
ax[1].set_title('Parte Real do Espectro')
ax[1].set_xlabel('Frequência')
ax[1].set_ylabel('Amplitude')
ax[1].grid(True)

ax[2].stem(frequencias_positivas, dft_imag_positivas)
ax[2].set_title('Parte Imaginária do Espectro')
ax[2].set_xlabel('Frequência')
ax[2].set_ylabel('Amplitude')
ax[2].grid(True)

plt.tight_layout()
plt.show()

#frequencias = []
#for i in range (len(t)):
    #frequencias.append(i/(t[-1] - t[0]))

#frequencias_positivas = []
#for i in range (len(frequencias)//2):
    #frequencias_positivas.append(frequencias[i])

#amplitudes_positivas = []
#for i in range (len(dft_result)//2):
    #amplitudes_positivas.append(dft_result[i])


#fig, ax = plt.subplots(2, 1, figsize=(12, 8))

#ax[0].plot(t,ricker)
#ax[0].set_xlabel('Tempo')
#ax[0].set_ylabel('Amplitude')
#ax[0].grid(True)

#ax[1].stem(frequencias_positivas,amplitudes_positivas)
#ax[1].set_title('Espectro de Amplitude')
#ax[1].set_xlabel('Frequência')
#ax[1].set_ylabel('Amplitude')
#ax[1].grid(True)

#plt.tight_layout()
#plt.show()


