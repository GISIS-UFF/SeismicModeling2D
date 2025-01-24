import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker

Nx = 501
dx = 10
Nt = 1001
dt = 0.001

x = np.linspace(0,dx*(Nx-1),Nx)
t = np.linspace(0,dt*(Nt-1),Nt) # O tempo representará a profundidade nesse caso

t1,t2          = 0.3, 0.6 # tempos (profundidade) das interfaces
rho1,rho2,rho3 = 2200,2400,2700 # densidade de cada camada
vp1,vp2,vp3    = 2500,3000,3500 # velocidade de cada camada
z1,z2,z3       = rho1*vp1,rho2*vp2,rho3*vp3

# criando os perfis
rho = np.zeros(Nt)
vp  = np.zeros(Nt)

rho[:int(t1/dt)]           = rho1
rho[int(t1/dt):int(t2/dt)] = rho2
rho[int(t2/dt):]           = rho3

vp[:int(t1/dt)]           = vp1
vp[int(t1/dt):int(t2/dt)] = vp2
vp[int(t2/dt):]           = vp3

Z = vp * rho

# Criando uma secao 2D
z2D = np.zeros([Nt,Nx])
for ix in range(Nx):
    z2D[:,ix] = Z

plt.figure()
plt.imshow(z2D,cmap="jet",aspect="auto",extent=[x[0],x[-1],t[-1],t[0]])
plt.xlabel("distancia lateral (m)")
plt.ylabel("tempo (s)")
plt.title("Impedancia")
plt.colorbar()
plt.tight_layout()


r12 = (z2 - z1) / (z2 + z1)
r23 = (z3 - z2) / (z3 + z2)

R = np.zeros_like(t)
R[int(t1/dt)] = r12
R[int(t2/dt)] = r23

wavelet = ricker(Nt, 4)
sinal = np.convolve(R,wavelet,mode='same')


R_fft = np.fft.fft(R)
wavelet_fft = np.fft.fft(wavelet)
sinal_fft = wavelet_fft * R_fft
sinal_tempo = np.fft.ifft(sinal_fft)

plt.subplot(1,6,1)
plt.plot(rho,t)
plt.xlabel('Densidade')
plt.gca().invert_yaxis()

plt.subplot(1,6,2)
plt.plot(vp,t)
plt.xlabel('Velocidade Compressional')
plt.gca().invert_yaxis()

plt.subplot(1,6,3)
plt.plot(Z,t)
plt.xlabel('Impedância acústica')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.subplot(1,6,4)
plt.plot(R,t)
plt.xlabel('Coeficiente de Reflexão')
plt.ylabel('Tempo(s)')
plt.gca().invert_yaxis()

plt.subplot(1,6,5)
plt.plot(sinal_tempo,t)
plt.ylabel('Sinal Sísmico fft')
plt.xlabel('Tempo')
plt.gca().invert_yaxis()

plt.subplot(1,6,6)
plt.plot(sinal,t)
plt.ylabel('Sinal Sísmico convolve')
plt.xlabel('Tempo')
plt.gca().invert_yaxis()

plt.show()

R_sec = np.zeros((Nt, Nx))
R_sec[int(t1/dt), :] = r12
R_sec[int(t2/dt), :] = r23

sec = np.zeros_like(R_sec)
for x in range(Nx):
    sec[:, x] = np.convolve(R_sec[:, x], wavelet, mode='same')


k = np.max(np.abs(sec))

plt.figure(figsize=(10, 6))
plt.imshow(sec, cmap="seismic", aspect="auto", extent=[0, dx*(Nx-1), t[-1], t[0]], vmin=-k, vmax=k)
plt.colorbar(label='Amplitude')
plt.xlabel("Distância lateral (m)")
plt.ylabel("Tempo (s)")
plt.title("Seção Sísmica")
plt.tight_layout()
plt.show()








