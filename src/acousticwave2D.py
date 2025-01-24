import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit
import pandas as pd
    
def v(nx, nz, v1=1500, v2=2000, v3=3000):
    vp = np.zeros([nz,nx])
    vp[0:int(nz/4),:]= v1
    vp[int(nz/4):int(nz/2),:] = v2
    vp[int(nz/2):nz,:] = v3
    return vp

def ler_modelo(caminho_arquivo, shape):
    vp = np.fromfile(caminho_arquivo, dtype=np.float32)
    vp = vp.reshape(shape).T
    print(f"Modelo de velocidade carregado de: {caminho_arquivo}")
    return vp

def expand_vp(v,nx_abc,nz_abc, N):
    v_expand = np.zeros((nz_abc, nx_abc))
    v_expand[N:nz_abc-N, N:nx_abc - N] = v
    v_expand[0:N,N: nx_abc - N]= v[0, :]
    v_expand[nz_abc - N : nz_abc, N: nx_abc - N]= v[-1, :]
    v_expand[N:nz_abc-N, 0:N] = v[:, 0][:, np.newaxis]
    v_expand[N:nz_abc-N, nx_abc-N:nx_abc] = v[:, -1][:, np.newaxis]
    v_expand[0:N, 0:N] = v[0, 0]  
    v_expand[0:N, nx_abc-N:nx_abc] = v[0, -1] 
    v_expand[nz_abc-N:nz_abc, 0:N] = v[-1, 0]  
    v_expand[nz_abc-N:nz_abc, nx_abc-N:nx_abc] = v[-1, -1]
    return v_expand


def ricker(f0, t):
    pi = np.pi
    td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td))
    return source

def ondas(nx,nz):
    u_anterior = np.zeros((nz,nx))
    u = np.zeros((nz,nx))
    u_posterior = np.zeros((nz,nx))
    return u_anterior, u, u_posterior

def borda (nx,nz,fator, N):
    A = np.ones((nz, nx))  
    for i in range(nx):
        for j in range(nz):
            if i < N: 
                A[j, i] *= np.exp(-((fator * (N - i)) ** 2))
            elif i >= nx - N:  
                A[j, i] *= np.exp(-((fator * (i - (nx - N))) ** 2))

            if j < N:  
                A[j, i] *= np.exp(-((fator * (N - j)) ** 2))
            elif j >= nz - N:  
                A[j, i] *= np.exp(-((fator * (j - (nz - N))) ** 2))

    return A


@numba.jit(parallel=True, nopython=True)
def marcha_no_espaço(u_anterior, u, u_posterior, nx, nz, c, dt, dx, dz):
    c0 = -205 / 72
    c1 = 8 / 5
    c2 = -1 / 5
    c3 = 8 / 315
    c4 = -1 / 560
    for i in numba.prange(4, nx - 4):  
        for j in numba.prange(4, nz - 4):  
            pxx = (c0 * u[j, i] + c1 * (u[j, i+1] + u[j, i-1]) + c2 * (u[j, i+2] + u[j, i-2]) +c3 * (u[j, i+3] + u[j, i-3]) +c4 * (u[j, i+4] + u[j, i-4])) / (dx * dx)
            pzz = (c0 * u[j, i] + c1 * (u[j+1, i] + u[j-1, i]) + c2 * (u[j+2, i] + u[j-2, i]) + c3 * (u[j+3, i] + u[j-3, i]) + c4 * (u[j+4, i] + u[j-4, i])) / (dz * dz)
            u_posterior[j, i] = (c[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * u[j, i] - u_anterior[j, i]
    return u_posterior

def marcha_no_tempo(u_anterior, u, u_posterior, source, nt, nx, nz, c, recx, recz, dt, A, shot_x, shot_z, dx, dz):
    isx = np.round(shot_x / dx).astype(int)
    isz = np.round(shot_z / dz).astype(int)
    sism = np.zeros((nt, nx))
    sism_shot = []
    u_snapshot = np.zeros((len(shot_x),nt,nz, nx))  
    for i_shot, (sx, sz) in enumerate(zip(isx, isz)):
        u_anterior.fill(0)  
        u.fill(0)
        u_posterior.fill(0)
        sism_atual = np.zeros((nt, nx))

        for k in range(nt):
            u[sz,sx]= u[sz,sx] + source[k]*(dt*c[sz, sx])**2
            u_posterior = marcha_no_espaço(u_anterior, u, u_posterior, nx, nz, c, dt, dx, dz) 
            u_posterior *= A
            u_anterior = np.copy(u) 
            u_anterior *= A
            u = np.copy(u_posterior)

            sism_atual[k, recx] = u[recz, recx]
            sism[k, recx] += u[recz, recx]
            u_snapshot[i_shot, k] = u.copy()

        sism_shot.append(sism_atual)
    return sism, sism_shot, u_snapshot
                         
def snapshot(u_snapshot, shot, nt, nz, nx):
    fig, ax = plt.subplots(figsize=(10, 10))
    for k in range(nt):
        if (k%100 == 0):
            ax.cla()
            ax.imshow(u_snapshot[shot, k])
            plt.pause(0.1)   
    u_snapshot.astype(np.float32).tofile(f'D:/GitHub/Acousticwave/outputs/snapshots/snapshot_{u_snapshot.shape[0]}x{nt}x{nz}x{nx}.bin')
    # print(u_snapshot.shape)
                    
def plot_shot(sism_shot):
    for i in range(len(sism_shot)):
        perc = np.percentile(sism_shot[i], 99)
        plt.imshow(sism_shot[i], aspect='auto', cmap='gray', vmin=-perc, vmax=perc)
        plt.colorbar(label='Amplitude')
        plt.title(" shot %s"%i)
        plt.show()

def salvar_sismograma(sism_shot):
    for i, shot in enumerate(sism_shot):
        shot.tofile(f'D:/GitHub/Acousticwave/outputs/seismograms/sismograma_shot_{i}_{shot.shape[0]}x{shot.shape[1]}.bin')
       
receiverTable = pd.read_csv("D:/GitHub/Acousticwave/inputs/receivers.csv")
sourceTable = pd.read_csv("D:/GitHub/Acousticwave/inputs/sources.csv")
rec_x = receiverTable['coordx'].to_numpy()
rec_z = receiverTable['coordz'].to_numpy()
shot_x = sourceTable['coordx'].to_numpy()
shot_z = sourceTable['coordz'].to_numpy()

T = 2 
dt = 0.001

L  = 9550
H = 3500
dx = 25
dz = 25
N = 50

nx = int(L/dx) + 1
nz = int(H/dz) + 1
nt = int(T/dt) + 1

nx_abc = nx + 2*N
nz_abc = nz + 2*N


shot_x = np.clip(shot_x, (N+1)* dx, (nx + N - 2) * dx)
shot_z = np.clip(shot_z, (N+1) * dz, (nz + N - 2) * dz)
rec_x= np.clip(rec_x, (N+1)* dx, (nx + N - 2) * dx)
rec_z= np.clip(rec_z, (N+1)* dz, (nz + N - 2) * dz)
f0 = 30

x = np.arange(0,L+dx,dx)
z = np.arange(0,H+dz,dz)
t = np.arange(0,T+dt,dt)
source = ricker(f0, t)

#c = v(nx,nz)
c = ler_modelo('D:/GitHub/Acousticwave/inputs/marmousi_vp_383x141.bin', (nx, nz))
c_expand = expand_vp(c,nx_abc,nz_abc, N)

plt.figure()
plt.imshow(c_expand,aspect='equal')
plt.show()

#critérios de dispersão e estabilidade
vp_min= np.min(c_expand)
vp_max = np.max(c_expand)
lambda_min = vp_min / f0
dx_lim = lambda_min / 5
dt_lim = dx_lim / (4 * vp_max)
if (dt>=dt_lim and dx>=dx_lim):
    print("Condições de estabilidade e dispersão satisfeitas")
else:
    print("Condições de estabilidade e dispersão não satisfeitas")
    print("dt_critical = %f dt = %f" %(dt_lim,dt))
    print("dx_critical = %f dx = %f" %(dx_lim,dx))
    print("fcut = %f " %(f0))

u_anterior, u, u_posterior = ondas(nx,nz)
recx= range(nx)
recz = N + 10
A = borda(nx, nz, fator=0.015, N = 50)
sism, sism_shot, u_snapshot = marcha_no_tempo(u_anterior, u, u_posterior, source, nt, nx, nz, c, recx, recz,dt, A, shot_x, shot_z, dx, dz)
sism_shot = sism_shot[::-1]
u_snapshot = u_snapshot[::-1]
plot_shot(sism_shot)
salvar_sismograma(sism_shot)
snapshot(u_snapshot, shot = 0, nt=nt, nz=nz, nx=nx)

