import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ricker(f0, td):
    pi = np.pi
    # td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td))
    return source

def half_derivative(sinal):
    fwavelet = np.fft.fft(sinal)
    omega = 2*np.pi*np.fft.fftfreq(len(sinal))
    fwavelet_half = np.sqrt(1j*omega)*fwavelet
    wavelet_half = np.real(np.fft.ifft(fwavelet_half))
    return wavelet_half
    

receiverTable = pd.read_csv('D:/GitHub/ModelagemSismica/inputs/receivers.csv')
sourceTable = pd.read_csv('D:/GitHub/ModelagemSismica/inputs/sources.csv')

rec_x = receiverTable['coordx'].to_numpy()
rec_z = receiverTable['coordz'].to_numpy()
shot_x = sourceTable['coordx'].to_numpy()
shot_z = sourceTable['coordz'].to_numpy()

L = 5000        
H = 5000          
T = 2            
dt = 0.0005        
dx = dz = 10      
f0 = 60
           
nx = int(L/dx) + 1
nz = int(H/dz) + 1
nt = int(T/dt) + 1

x = np.linspace(0, L, nx, endpoint=False)
z = np.linspace(0, H, nz, endpoint=False)
t = np.linspace(-T/2, T/2, nt, endpoint=False) 


v1=3000
v2=4000

R1 = (v2 - v1)/ (v2 + v1)

td = 0 #2 * np.sqrt(np.pi) / f0
wavelet = ricker(f0, t-td)
plt.figure()
plt.plot(t, wavelet) 
plt.show()
wavelet_half = half_derivative(wavelet)
plt.figure()    
plt.plot(t, wavelet_half)
plt.show()

h1 = H/2 - shot_z[0]

t_direct = np.zeros((len(shot_x), len(rec_x)))
t_ref1 = np.zeros((len(shot_x), len(rec_x)))

sism_shot = []
t_lag = 2 * np.sqrt(np.pi) / f0
for i in range(len(shot_x)):
    sism = np.zeros((nt, len(rec_x)))
    for j in range(len(rec_x)):
        dist = np.sqrt((shot_x[i] - rec_x[j]) ** 2 + (shot_z[i] - rec_z[j]) ** 2)
        t_direct[i, j] = dist / v1 + t_lag
        t_ref1[i, j] = np.sqrt(dist**2 + (2*h1)**2) / v1 + t_lag

        if (t_direct[i, j] < T):
            sism[int(t_direct[i, j]/dt), j] = 1
        if (t_ref1[i, j] < T):
            sism[int(t_ref1[i, j]/dt), j] = R1

        sism[:,j] = np.convolve(sism[:,j], wavelet, mode='same')
    sism_shot.append(sism.copy())   

for i in range(len(sism_shot)):
    perc = np.percentile(sism_shot[i], 99)
    plt.imshow(sism_shot[i], aspect='auto', cmap='gray', vmin=-perc, vmax=perc)
    plt.colorbar(label='Amplitude')
    plt.title(" shot %s"%i)
    plt.show()
for i, shot in enumerate(sism_shot):
    shot.tofile(f'../ModelagemSismica/outputs/seismograms/sismograma_analitico_shot_{i}_{shot.shape[0]}x{shot.shape[1]}.bin')