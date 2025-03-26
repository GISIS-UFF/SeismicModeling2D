import numpy as np
import matplotlib.pyplot as plt
import numba
import pandas as pd

class Wavefield:
    def __init__(self, nx, nz, nt, dx, dz, T, dt, f0, N, fator_atenuacao, v1, v2, nx_abc, nz_abc, rec_x, rec_z, shot_x, shot_z):
        self.nx = nx
        self.nz = nz
        self.nt = nt
        self.dx = dx
        self.dz = dz
        self.T = T
        self.dt = dt
        self.f0 = f0
        self.N = N
        self.fator_atenuacao = fator_atenuacao
        self.v1 = v1
        self.v2 = v2
        self.nx_abc = nx_abc
        self.nz_abc = nz_abc
        self.recx = rec_x    
        self.recz = rec_z    
        self.shot_x = shot_x
        self.shot_z = shot_z

        self.vp = self.criar_modelo()
        self.c_expand = self.expandir()
        self.source = self.ricker()
        self.A = self.criar_borda()
        
        self.u_anterior = np.zeros((self.nz_abc, self.nx_abc))
        self.u = np.zeros((self.nz_abc, self.nx_abc))
        self.u_posterior = np.zeros((self.nz_abc, self.nx_abc))
    
    def criar_modelo(self):
        vp = np.zeros((self.nz, self.nx))
        vp[0:int(self.nz / 2), :] = self.v1
        vp[int(self.nz / 2):self.nz, :] = self.v2
        return vp
    
    def expandir(self):
        v_expand = np.zeros((self.nz_abc, self.nx_abc))
        v_expand[self.N:self.nz_abc-self.N, self.N:self.nx_abc-self.N] = self.vp
        v_expand[0:self.N, self.N:self.nx_abc-self.N] = self.vp[0, :]
        v_expand[self.nz_abc-self.N:self.nz_abc, self.N:self.nx_abc-self.N] = self.vp[-1, :]
        v_expand[self.N:self.nz_abc-self.N, 0:self.N] = self.vp[:, 0:1]
        v_expand[self.N:self.nz_abc-self.N, self.nx_abc-self.N:self.nx_abc] = self.vp[:, -1:]
        v_expand[0:self.N, 0:self.N] = self.vp[0, 0]
        v_expand[0:self.N, self.nx_abc-self.N:self.nx_abc] = self.vp[0, -1]
        v_expand[self.nz_abc-self.N:self.nz_abc, 0:self.N] = self.vp[-1, 0]
        v_expand[self.nz_abc-self.N:self.nz_abc, self.nx_abc-self.N:self.nx_abc] = self.vp[-1, -1]
        return v_expand
    
    def check_dispersionstability(self):
        vp_min= np.min(self.c_expand)
        vp_max = np.max(self.c_expand)
        lambda_min = vp_min / self.f0
        dx_lim = lambda_min / 5
        dt_lim = dx_lim / (4 * vp_max)
        if (self.dt>=dt_lim and self.dx>=dx_lim):
            print("Condições de estabilidade e dispersão satisfeitas")
        else:
            print("Condições de estabilidade e dispersão não satisfeitas")
            print("dt_critical = %f dt = %f" %(dt_lim,self.dt))
            print("dx_critical = %f dx = %f" %(dx_lim,self.dx))
            print("fcut = %f " %(self.f0))
    
    def ricker(self):
        t = np.linspace(0, self.T, self.nt, endpoint=False)
        pi = np.pi
        td = t - 2 * np.sqrt(pi) / self.f0
        fcd = self.f0 / (np.sqrt(pi) * 3)
        return (1 - 2 * (pi * fcd * td) ** 2) * np.exp(-(pi * fcd * td) ** 2)
    
    def criar_borda(self):
        A = np.ones((self.nz_abc, self.nx_abc))
        for i in range(self.nx_abc):
            for j in range(self.nz_abc):
                if i < self.N:
                    A[j, i] *= np.exp(-((self.fator_atenuacao * (self.N - i)) ** 2))
                if i >= self.nx_abc - self.N:
                    A[j, i] *= np.exp(-((self.fator_atenuacao * (i - (self.nx_abc - self.N))) ** 2))
                if j < self.N:
                    A[j, i] *= np.exp(-((self.fator_atenuacao * (self.N - j)) ** 2))
                if j >= self.nz_abc - self.N:
                    A[j, i] *= np.exp(-((self.fator_atenuacao * (j - (self.nz_abc - self.N))) ** 2))
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
                pxx = (c0 * u[j, i] + c1 * (u[j, i+1] + u[j, i-1]) + 
                       c2 * (u[j, i+2] + u[j, i-2]) + 
                       c3 * (u[j, i+3] + u[j, i-3]) + 
                       c4 * (u[j, i+4] + u[j, i-4])) / (dx * dx)
                pzz = (c0 * u[j, i] + c1 * (u[j+1, i] + u[j-1, i]) + 
                       c2 * (u[j+2, i] + u[j-2, i]) + 
                       c3 * (u[j+3, i] + u[j-3, i]) + 
                       c4 * (u[j+4, i] + u[j-4, i])) / (dz * dz)
                u_posterior[j, i] = (c[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * u[j, i] - u_anterior[j, i]
        return u_posterior
    
    def marcha_no_tempo(self):
        self.sism = np.zeros((self.nt, len(self.recx)))
        self.sism_shot = []
        self.u_snapshot = np.zeros((len(self.shot_x), self.nt, self.nz_abc, self.nx_abc), dtype=np.float32)  
        for i_shot, (sx, sz) in enumerate(zip(self.shot_x, self.shot_z)):
            self.u_anterior.fill(0)  
            self.u.fill(0)
            self.u_posterior.fill(0)
            self.sism_atual = np.zeros((self.nt, len(self.recx)))
            for k in range(self.nt):
                self.u[sz, sx] = self.u[sz, sx] + self.source[k] * (self.dt * self.c_expand[sz, sx]) ** 2
                self.u_posterior = Wavefield.marcha_no_espaço(self.u_anterior, self.u, self.u_posterior, self.nx, self.nz, self.c_expand, self.dt, self.dx, self.dz) 
                self.u_posterior *= self.A
                self.u_anterior = np.copy(self.u) 
                self.u_anterior *= self.A
                self.u = np.copy(self.u_posterior)

                self.sism_atual[k, :] = self.u[self.recz, self.recx]
                self.sism[k, :] += self.u[self.recz, self.recx]
                self.u_snapshot[i_shot, k] = self.u.copy()

            self.sism_shot.append(self.sism_atual)
        
    
    def snapshot(self, shot):
        fig, ax = plt.subplots(figsize=(10, 10))
        for k in range(self.nt):
            if (k%100 == 0):
                ax.cla()
                ax.imshow(self.u_snapshot[shot, k], cmap='gray')
                plt.pause(0.1)   
        # u_snapshot.astype(np.float32).tofile(f'D:/GitHub/ModelagemSismica/outputs/snapshots/snapshot_{u_snapshot.shape[0]}x{nt}x{nz}x{nx}.bin')
        # print(u_snapshot.shape)  
    
    def plot_shot(self):
        for i in range(len(self.sism_shot)):
            perc = np.percentile(self.sism_shot[i], 99)
            plt.imshow(self.sism_shot[i], aspect='auto', cmap='gray', vmin=-perc, vmax=perc)
            plt.colorbar(label='Amplitude')
            plt.title(" shot %s"%i)
            plt.show()
        for i, shot in enumerate(self.sism_shot):
            filename = f'D:/GitHub/ModelagemSismica/outputs/seismograms/sismograma_shot_{i}_{shot.shape[0]}x{shot.shape[1]}.bin'
            print(filename)
            shot.tofile(filename)

    def plot_modelo(self):
        plt.figure()
        plt.plot(self.shot_x,self.shot_z,"r*", markersize=5)
        plt.plot(self.recx,self.recz,'bv',markersize = 2)
        plt.imshow(self.c_expand,aspect='equal')
        plt.show()

receiverTable = pd.read_csv("D:/GitHub/ModelagemSismica/inputs/receivers.csv")
sourceTable = pd.read_csv("D:/GitHub/ModelagemSismica/inputs/sources.csv")
rec_x = receiverTable['coordx'].to_numpy()
rec_z = receiverTable['coordz'].to_numpy()
shot_x = sourceTable['coordx'].to_numpy()
shot_z = sourceTable['coordz'].to_numpy()

T = 2 
dt = 0.001

L = 5000
H = 5000
dx = 10
dz = 10
N = 50
fator_atenuacao = 0.015

nx = int(L/dx) + 1
nz = int(H/dz) + 1
nt = int(T/dt) + 1
nx_abc = nx + 2*N
nz_abc = nz + 2*N

rec_x = np.round(rec_x/dx).astype(int) + N 
rec_z = np.round(rec_z/dz).astype(int) + N
shot_x = np.round(shot_x/dx).astype(int) + N
shot_z = np.round(shot_z/dz).astype(int) + N

f0 = 60
v1, v2 = 3000, 4000

wavefield = Wavefield(nx, nz, nt, dx, dz, T, dt, f0, N, fator_atenuacao, v1, v2, nx_abc, nz_abc, rec_x, rec_z, shot_x, shot_z)
wavefield.plot_modelo()
wavefield.check_dispersionstability()
wavefield.marcha_no_tempo()
wavefield.plot_shot()
wavefield.snapshot(0)