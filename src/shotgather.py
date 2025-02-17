import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ricker(f0, td):
    pi = np.pi
    # td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td))
    return source

receiverTable = pd.read_csv('D:/GitHub/ModelagemSismica/inputs/receivers.csv')
sourceTable = pd.read_csv('D:/GitHub/ModelagemSismica/inputs/sources.csv')

rec_x = receiverTable['coordx'].to_numpy()
rec_z = receiverTable['coordz'].to_numpy()
shot_x = sourceTable['coordx'].to_numpy()
shot_z = sourceTable['coordz'].to_numpy()

L = 5000        
H = 5000          
T = 2            
dt = 0.001        
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

h1 = H/2 - shot_z[0]


t_direct = np.zeros((len(shot_x), len(rec_x)))
t_ref1 = np.zeros((len(shot_x), len(rec_x)))
t_ref2 = np.zeros((len(shot_x), len(rec_x)))
t_hw1 = np.zeros((len(shot_x), len(rec_x)))
t_hw2 = np.zeros((len(shot_x), len(rec_x)))
t_gr = np.zeros((len(shot_x), len(rec_x)))


sism_shot = []
t_lag = 2 * np.sqrt(np.pi) / f0
for i in range(len(shot_x)):
    sism = np.zeros((nt, len(rec_x)))
    for j in range(len(rec_x)):
        dist = np.sqrt((shot_x[i] - rec_x[j]) ** 2 + (shot_z[i] - rec_z[j]) ** 2)
        t_direct[i, j] = dist / v1 + t_lag
        t_ref1[i, j] = np.sqrt((2 * h1 / v1) ** 2 + (dist / v1) ** 2) + t_lag
        # t_ref2[i, j] = np.sqrt((2 * h2 / v2) ** 2 + (dist / v2) ** 2) + (2 * h1 / v1)
        # t_hw1[i,j] = dist/v2 + (2*h1) * np.sqrt(v2**2 - v1**2) / (v1 * v2)
        # t_hw2[i,j] = dist/v3 + (2*h1)*(np.sqrt(v3**2 - v1**2))/v3*v1 + (2*h2)*(np.sqrt(v3**2 - v2**2))/v3*v2 
        # t_gr[i, j] = dist / v_gr

        if (t_direct[i, j] < T):
            sism[int(t_direct[i, j]/dt), j] = 1
        if (t_ref1[i, j] < T):
            sism[int(t_ref1[i, j]/dt), j] = R1
        # if (t_ref2[i, j] < T):
        #     sism[int(t_ref2[i, j]/dt), j] = 1
        # if (t_hw1[i, j] < T):
        #     sism[int(t_hw1[i, j]/dt), j] = 1
        # if (t_hw2[i, j] < T):
        #     sism[int(t_hw2[i, j]/dt), j] = 1
        # # if (t_gr[i, j] < T):
        # #     sism[int(t_gr[i, j]/dt), j] = 1

        sism[:,j] = np.convolve(sism[:,j], wavelet, mode='same')
    sism_shot.append(sism.copy())   

for i in range(len(sism_shot)):
    perc = np.percentile(sism_shot[i], 99)
    plt.imshow(sism_shot[i], aspect='auto', cmap='gray', vmin=-perc, vmax=perc)
    plt.colorbar(label='Amplitude')
    plt.title(" shot %s"%i)
    plt.show()
for i, shot in enumerate(sism_shot):
    shot.tofile(f'D:/GitHub/ModelagemSismica/outputs/seismograms/sismograma_analitico_shot_{i}_{shot.shape[0]}x{shot.shape[1]}.bin')



# plt.figure()
# for i in range(len(shot_x)):
#     plt.plot(rec_x, t_direct[i,:],label='Onda Direta')
#     plt.plot(rec_x, t_ref1[i,:],label='Reflexão interface 1')
#     plt.plot(rec_x, t_ref2[i,:],label='Reflexão interface 2')
#     plt.plot(rec_x, t_hw1[i,:], label='Head Wave interface 1')
#     plt.plot(rec_x, t_hw2[i,:], label='Head Wave interface 2')
#     # plt.plot(rec_x, t_gr[i,:], label='Ground Roll')
#     plt.gca().invert_yaxis() 
#     plt.legend()
#     plt.show()


