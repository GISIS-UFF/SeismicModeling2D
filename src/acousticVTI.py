import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit
import pandas as pd


def ricker(f0, t):
    pi = np.pi
    td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td))
    return source

def ondas(nx,nz):
    p_anterior = np.zeros((nz,nx))
    p = np.zeros((nz,nx))
    p_posterior = np.zeros((nz,nx))
    q_anterior = np.zeros((nz,nx))
    q = np.zeros((nz,nx))
    q_posterior = np.zeros((nz,nx))
    return p_anterior, p, p_posterior, q_anterior, q, q_posterior

def borda (nx,nz,fator, N):
    A = np.ones((nz, nx))
    sb = 3*N 
    for i in range(nx):
        for j in range(nz):
            if i < N:  
                fb = (N - i) / (np.sqrt(2) * sb)
                A[j, i] *= np.exp(-fb * fb)
            elif i >= nx - N: 
                fb = (i - (nx - N)) / (np.sqrt(2) * sb)
                A[j, i] *= np.exp(-fb * fb)
            if j < N:  
                fb = (N - j) / (np.sqrt(2) * sb)
                A[j, i] *= np.exp(-fb * fb)
            elif j >= nz - N:  
                fb = (j - (nz - N)) / (np.sqrt(2) * sb)
                A[j, i] *= np.exp(-fb * fb)
    return A


@numba.jit(parallel=True, nopython=True)
def marcha_no_espaço(p_anterior, p, p_posterior, q, q_anterior, q_posterior, nx, nz, dt, dx, dz, vpz, epsilon, delta):
    cx = vpz**2 * (1 + 2 * epsilon)
    bx = vpz**2 * (1 + 2 * delta)
    cz = bz = vpz**2    
    c0 = -205 / 72
    c1 = 8 / 5
    c2 = -1 / 5
    c3 = 8 / 315
    c4 = -1 / 560
    for i in numba.prange(4, nx - 4):  
        for j in numba.prange(4, nz - 4):  
            pxx = (c0 * p[j, i] + c1 * (p[j, i+1] + p[j, i-1]) + c2 * (p[j, i+2] + p[j, i-2]) + 
                   c3 * (p[j, i+3] + p[j, i-3]) + c4 * (p[j, i+4] + p[j, i-4])) / (dx * dx)
            qzz = (c0 * q[j, i] + c1 * (q[j+1, i] + q[j-1, i]) + c2 * (q[j+2, i] + q[j-2, i]) + 
                   c3 * (q[j+3, i] + q[j-3, i]) + c4 * (q[j+4, i] + q[j-4, i])) / (dz * dz)
            p_posterior[j, i] = 2 * p[j, i] - p_anterior[j, i] + (dt**2) * (cx * pxx  + cz * qzz)
            q_posterior[j, i] = 2 * q[j, i] - q_anterior[j, i] + (dt**2) * (bx * pxx  + bz * qzz)

    return p_posterior, q_posterior


def marcha_no_tempo(p_anterior, p, p_posterior, q, q_anterior, q_posterior, source, nt, nx, nz, recx, recz, dt, A, shot_x, shot_z, dx, dz, frame, vpz, epsilon, delta):
    sism_shot = []
    p_snapshot = []
    for i_shot, (sx, sz) in enumerate(zip(shot_x, shot_z)):
        p_anterior.fill(0)  
        p.fill(0)
        p_posterior.fill(0)
        q_anterior.fill(0)  
        q.fill(0)
        q_posterior.fill(0)
        sism = np.zeros((nt, len(recx)))
        for k in range(nt):
            p[sz,sx]= p[sz,sx] + source[k]
            q[sz,sx]= q[sz,sx] + source[k]
            p_posterior, q_posterior = marcha_no_espaço(p_anterior, p, p_posterior, q, q_anterior, q_posterior, nx, nz, dt, dx, dz, vpz, epsilon, delta)
            p_posterior *= A
            q_posterior *= A
            p_anterior = np.copy(p) 
            q_anterior = np.copy(q)
            p_anterior *= A
            q_anterior *= A
            p = np.copy(p_posterior)
            q = np.copy(q_posterior)

            sism[k, :] = p[recz, recx]
            if k == frame:
                p_snapshot.append(p.copy())

        sism_shot.append(sism)
    return sism_shot,p_snapshot


def snapshot(p_snapshot, shot, frame):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(p_snapshot[shot], cmap='gray')
    plt.title(f"Snapshot no frame {frame} para o shot {shot}")
    plt.show()
    # filename = f'../outputs/snapshots/snapshot_frame_{frame}_shot{shot}.bin'
    # p_snapshot[shot].astype(np.float32).tofile(filename)
    # print(f"Snapshot do frame {frame} salvo em: {filename}")

                    
def plot_shot(sism_shot):
    for i in range(len(sism_shot)):
        perc = np.percentile(sism_shot[i], 99)
        plt.imshow(sism_shot[i], aspect='auto', cmap='gray', vmin=-perc, vmax=perc)
        plt.colorbar(label='Amplitude')
        plt.title(" shot %s"%i)
        plt.show()
    # for i, shot in enumerate(sism_shot):
    #     filename = f'../outputs/seismograms/sismograma_shot_{i}_{shot.shape[0]}x{shot.shape[1]}.bin'
    #     print(filename)
    #     shot.tofile(filename)


T = 2 
dt = 0.0005

L = 5000
H = 5000
dx = 10
dz = 10
N = 100

nx = int(L/dx) + 1
nz = int(H/dz) + 1
nt = int(T/dt) + 1

nx_abc = nx + 2*N
nz_abc = nz + 2*N

rec_x = np.arange(N, nx_abc-N, 1).astype(int)
rec_z = 100*np.ones(len(rec_x)).astype(int)
shot_x = [nx_abc // 2]
shot_z = [nx_abc // 2] 

plt.figure()
plt.plot(shot_x,shot_z,"r*", markersize=5)
plt.plot(rec_x,rec_z,'bv',markersize = 2)
plt.ylim(rec_x[-1],0)
plt.show()

vpz = 3000     
epsilon = 0.24 
delta = 0.1

t = np.linspace(0, T, nt, endpoint=False)

f0 = 60
source = ricker(f0, t)

#critérios de dispersão e estabilidade
vp_min = np.min(vpz)
vpx = vpz*np.sqrt(1+2*epsilon)
vpx_max = np.max(vpx)
lambda_min = vp_min / f0
dx_lim = lambda_min / 5
dt_lim = dx_lim / (4 * vpx_max)
if (dt<=dt_lim and dx<=dx_lim):
    print("Condições de estabilidade e dispersão satisfeitas")
else:
    print("Condições de estabilidade e dispersão não satisfeitas")
    print("dt_critical = %f dt = %f" %(dt_lim,dt))
    print("dx_critical = %f dx = %f" %(dx_lim,dx))
    print("fcut = %f " %(f0))

p_anterior, p, p_posterior, q_anterior, q, q_posterior= ondas(nx_abc,nz_abc)
A = borda(nx_abc, nz_abc, 0.015, N)
frame = 1000
sism_shot, p_snapshot = marcha_no_tempo(p_anterior, p, p_posterior, q, q_anterior, q_posterior, source, nt, nx_abc, nz_abc, rec_x, rec_z, dt, A, shot_x, shot_z, dx, dz, frame, vpz, epsilon, delta)
sism_shot = sism_shot[::-1]
p_snapshot = p_snapshot[::-1]
plot_shot(sism_shot)
snapshot(p_snapshot, 0, frame)

