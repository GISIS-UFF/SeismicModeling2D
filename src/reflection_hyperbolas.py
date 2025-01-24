import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker

h = 1000
v1 = 1500
v2 = 4000
v_gr= 400

Nx = 501
dx = 10
Nt = 6001
dt = 0.001

wavelet = ricker(Nt, 4)
x = np.linspace(0,dx*(Nx-1),Nx)
t = np.linspace(0,dt*(Nt-1),Nt)

t_ref = np.sqrt((2*h/v1)**2 + (x/v1)**2)
t_direct = x/v1
t_hw = x/v2 + (2*h*np.sqrt(v2**2-v1**2))/(v1*v2)
t_gr= x/v_gr

plt.plot(x, t_direct, label = "direct wave")
plt.plot(x,t_ref, label = "reflection wave")
plt.plot(x,t_hw, label = "head wave")
plt.plot(x, t_gr, label = "ground role")
plt.ylim(t[-1],0)
plt.xlabel('Distância horizontal (m)')
plt.ylabel('Tempo (s)')
plt.grid(True)
plt.legend()
plt.show()

sism = np.zeros((Nt,Nx))

for ix in range(Nx):
    k = int(t_direct[ix]/dt)
    y = int(t_ref[ix]/dt)
    z = int(t_hw[ix]/dt)
    u = int(t_gr[ix]/dt)

    sism[k,ix] = 1
    sism[y,ix] = 1
    sism[z,ix] = 1
    if u < Nt:
        sism[u,ix] = 1

for x in range(Nx):
    sism[:, x] = np.convolve(sism[:, x], wavelet, mode='same')

k = np.max(np.abs(sism))

plt.figure(figsize=(10, 6))
plt.imshow(sism, cmap="seismic", aspect="auto", extent=[0, dx*(Nx-1), t[-1], t[0]], vmin=-k, vmax=k)
plt.colorbar(label='Amplitude')
plt.title("Sismograma")
plt.tight_layout()
plt.show()

