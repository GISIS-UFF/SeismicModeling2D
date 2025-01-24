import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ricker(f0, t):
    pi = np.pi
    td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td))
    return source

receiverTable = pd.read_csv('D:/GitHub/Acousticwave/inputs/receivers.csv')
sourceTable = pd.read_csv('D:/GitHub/Acousticwave/inputs/sources.csv')

rec_x = receiverTable['coordx'].to_numpy()
rec_z = receiverTable['coordz'].to_numpy()
shot_x = sourceTable['coordx'].to_numpy()
shot_z = sourceTable['coordz'].to_numpy()

L = 10000         
H = 3000          
T = 2            
dt = 0.001        
dx = dz = 5      
f0 = 60           

x = np.arange(0, L + dx, dx) 
z = np.arange(0, H + dz, dz) 
t = np.arange(0, T + dt, dt)  

nx = len(x) 
nz = len(z)  
nt = len(t)  

v1 = 1500  
v2 = 2000 
v3 = 3000  
v_gr = 780

vp = np.zeros((nz, nx))
vp[0:int(nz / 4), :] = v1
vp[int(nz / 4):int(nz / 2), :] = v2
vp[int(nz / 2):, :] = v3


wavelet = ricker(f0, t)
plt.plot(t, wavelet)
plt.show()

t_direct_list = []
t_ref_list = []
t_hw_list = []
t_gr_list = []

for s, shot_x_val in enumerate(shot_x):
    t_direct_wave = []
    t_ref_wave = []
    t_hw_wave = []
    t_gr_wave = []
    
    for r, rec_x_val in enumerate(rec_x):
        dx = np.abs(rec_x_val - shot_x_val)
        dz = np.abs(rec_z[r] - shot_z[s])
        dist = np.sqrt(dx**2 + dz**2)
        
        t_direct = dist / v1
        t_ref = np.sqrt((2 * H / v1) ** 2 + (dist / v1) ** 2)
        t_hw = dist / v2 + (2 * H * np.sqrt(v2 ** 2 - v1 ** 2)) / (v1 * v2)
        t_gr = dist / v_gr

        t_direct_wave.append(t_direct)
        t_ref_wave.append(t_ref)
        t_hw_wave.append(t_hw)
        t_gr_wave.append(t_gr)
     
    t_direct_list.append(t_direct_wave)
    t_ref_list.append(t_ref_wave)
    t_hw_list.append(t_hw_wave)
    t_gr_list.append(t_gr_wave)

#plot de todos os shots 
plt.figure()
for t_direct in t_direct_list:
    plt.plot(rec_x, t_direct)
# for t_ref in t_ref_list:
#     plt.plot(rec_x, t_ref)
# for t_hw in t_hw_list:
#     plt.plot(rec_x, t_hw)
# for t_gr in t_gr_list:
#     plt.plot(rec_x, t_gr)
plt.gca().invert_yaxis() 
plt.show()

#plot de cada shot
for i in range(len(shot_x)):
    plt.figure()
    plt.title(" shot %s"%i)
    plt.plot(rec_x, t_direct_list[i], label="Direct wave")
    # plt.plot(rec_x, t_ref_list[i], label="Reflection wave")
    # plt.plot(rec_x, t_hw_list[i], label="Head wave")
    # plt.plot(rec_x, t_gr_list[i], label="Ground roll")
    plt.gca().invert_yaxis()
    plt.show()

sism = np.zeros((nt, len(rec_x) ,len(shot_x)))

for s, shot_x_val in enumerate(shot_x):
    for r, rec_x_val in enumerate(rec_x):
        dx = np.abs(rec_x_val - shot_x_val)
        dz = np.abs(rec_z[r] - shot_z[s])
        dist = np.sqrt(dx**2 + dz**2)

        k = int((dist / v1) / dt)  
        y = int((np.sqrt((2 * H / v1) ** 2 + (dist / v1) ** 2)) / dt)  
        z = int((dist / v2 + (2 * H * np.sqrt(v2 ** 2 - v1 ** 2)) / (v1 * v2)) / dt)  
        u = int((dist / v_gr) / dt)  

        if k < nt:
            sism[k, r, s] = 1
        if y < nt:
            sism[y, r, s] = 1
        if z < nt:
            sism[z, r, s] = 1
        if u < nt:
            sism[u, r, s] = 1

    for r in range(nx):
        sism[:, r, s] = np.convolve(sism[:, r, s], wavelet, mode='same')

def plot_sismograma(sism):
    plt.figure()
    for i in range(sism.shape[2]):
        plt.imshow(sism[:, :, i], aspect='auto', cmap='gray', extent=[0, len(rec_x), T, 0])
        plt.title(f'Sismograma - Shot {i}')
    plt.colorbar()
    plt.show()

plot_sismograma(sism)