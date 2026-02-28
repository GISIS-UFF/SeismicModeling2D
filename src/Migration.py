import numpy as np
import time
import random

from utils import updateWaveEquation
from utils import updateWaveEquationCPML
from utils import updateWaveEquationVTI
from utils import updateWaveEquationVTICPML
from utils import updateWaveEquationTTI
from utils import AbsorbingBoundary
from utils import updatePsi
from utils import updateZeta
from utils import Mute

class migration: 

    def __init__(self,wavefield,parameters):
        self.pmt = parameters
        self.wf = wavefield
    
    def loadSeismogram(self, shot):
        seismogramFile = f"{self.pmt.seismogramFolder}seismogram_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        seismogram = np.fromfile(seismogramFile, dtype=np.float32).reshape(self.pmt.nt,self.pmt.Nrec) 
        return seismogram
        
    def laplacian_filter(self, f):
        dim1,dim2 = np.shape(f)
        g = np.zeros([dim1,dim2])
        lap_z = 0
        lap_x = 0
        for ix in range(1, dim2 - 1):
            for iz in range(1, dim1 - 1):
                lap_z = f[iz+1, ix] + f[iz-1, ix] - 2 * f[iz, ix]
                lap_x = f[iz, ix+1] + f[iz, ix-1] - 2 * f[iz, ix]
                g[iz, ix] = lap_z/(self.pmt.dz*self.pmt.dz) + lap_x/(self.pmt.dx*self.pmt.dx)

        for ix in range(dim2):
            g[0, ix] = g[1, ix]
            g[-1, ix] = g[-2, ix]
        for iz in range(dim1):
            g[iz, 0] = g[iz, 1]
            g[iz, -1] = g[iz, -2]

        return(-g)
    
    def gaussian_kernel(self, x, z, sigma):
        fator = 1. / (2.*np.pi*sigma*sigma)
        expoente = -(x * x + z * z)/(2.*sigma*sigma)
        return fator * np.exp(expoente)

    def gaussian_filter2D(self,sigma):
        kernel_size = np.ceil(2 * sigma + 1).astype(int)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel2d = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        total = 0.0

        for lin in range(kernel_size):
            for col in range(kernel_size):
                x = lin - kernel_size // 2
                y = col - kernel_size // 2
                val = self.gaussian_kernel(x, y, sigma)
                kernel2d[lin, col] = val
                total += val

        kernel2d /= total

        return kernel2d

    def smooth_model(self,f,sigma):
        s = 1.0 / f
        kernel = self.gaussian_filter2D(sigma)
        ksize = kernel.shape[0]
        half = ksize // 2

        nz, nx = np.shape(s)

        for z in range(half, nz - half):
            for x in range(half, nx - half):
                new_value = 0.0
                for i in range(ksize):
                    for j in range(ksize):
                        new_value += (kernel[i, j] * s[z + i - half, x + j - half])
                s[z, x] = new_value

        for z in range(half):
            s[z, :] = s[half, :]
            s[nz - 1 - z, :] = s[nz - 1 - half, :]
        for x in range(half):
            s[:, x] = s[:, half]
            s[:, nx - 1 - x] = s[:, nx - 1 - half]

        return (1.0 / s)
                   
    def load_checkpoint(self, shot, k):
        checkpointFile = (f"{self.pmt.checkpointFolder}{self.pmt.approximation}{self.pmt.ABC}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")
        with open(checkpointFile, "rb") as file:
            count = self.pmt.nx_abc * self.pmt.nz_abc
            self.wf.current = np.fromfile(file, np.float32, count).reshape(self.pmt.nz_abc, self.pmt.nx_abc)
            self.wf.future  = np.fromfile(file, np.float32, count).reshape(self.pmt.nz_abc, self.pmt.nx_abc)

            if self.pmt.ABC == "CPML":
                count_x = self.pmt.nz_abc * (self.pmt.N_abc+4)
                count_z = (self.pmt.N_abc+4) * self.pmt.nx_abc

                self.wf.PsixFR  = np.fromfile(file, np.float32, count_x).reshape(self.pmt.nz_abc, self.pmt.N_abc+4)
                self.wf.PsixFL  = np.fromfile(file, np.float32, count_x).reshape(self.pmt.nz_abc, self.pmt.N_abc+4)
                self.wf.PsizFU  = np.fromfile(file, np.float32, count_z).reshape(self.pmt.N_abc+4, self.pmt.nx_abc)
                self.wf.PsizFD  = np.fromfile(file, np.float32, count_z).reshape(self.pmt.N_abc+4, self.pmt.nx_abc)

                self.wf.ZetaxFR = np.fromfile(file, np.float32, count_x).reshape(self.pmt.nz_abc, self.pmt.N_abc+4)
                self.wf.ZetaxFL = np.fromfile(file, np.float32, count_x).reshape(self.pmt.nz_abc, self.pmt.N_abc+4)
                self.wf.ZetazFU = np.fromfile(file, np.float32, count_z).reshape(self.pmt.N_abc+4, self.pmt.nx_abc)
                self.wf.ZetazFD = np.fromfile(file, np.float32, count_z).reshape(self.pmt.N_abc+4, self.pmt.nx_abc)
    
    def save_checkpoint(self, shot, k):
        if self.pmt.migration != "checkpoint":
            return
        if k not in self.ckpt_frames:
            return
        
        checkpointFile = (f"{self.pmt.checkpointFolder}{self.pmt.approximation}{self.pmt.ABC}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")

        save = [self.wf.current, self.wf.future]
        if self.pmt.ABC == "CPML":
            save += [self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD]

        with open(checkpointFile, "wb") as file:
            for field in save:
                field.astype(np.float32).tofile(file)

        print(f"info: Checkpoint saved to {checkpointFile}")
    
    def forward_step(self,k):
        if self.pmt.migration in ["onthefly", "checkpoint", "SB"]:
            if self.pmt.approximation == "acoustic" and self.pmt.ABC == "cerjan":
                self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.current, self.A)

            elif self.pmt.approximation == "acoustic" and self.pmt.ABC == "CPML":
                self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.future = updateWaveEquationCPML(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)             
                self.wf.future[self.sz,self.sx] += self.wf.source[k]

            elif self.pmt.approximation == "VTI" and self.pmt.ABC == "cerjan":
                self.wf.future= updateWaveEquationVTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.current, self.A)

            elif self.pmt.approximation == "VTI" and self.pmt.ABC == "CPML":
                self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.future = updateWaveEquationVTICPML(self.wf.future, self.wf.current, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.pmt.nx_abc, self.pmt.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]

            elif self.pmt.approximation == "TTI" and self.pmt.ABC == "cerjan":
                self.wf.future= updateWaveEquationTTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.current, self.A)
        if self.pmt.migration == "RBC":
            if self.pmt.approximation == "acoustic":
                self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
            elif self.pmt.approximation == "VTI":
                self.wf.future= updateWaveEquationVTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
            elif self.pmt.approximation == "TTI":
                self.wf.future= updateWaveEquationTTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]

    def backward_step(self,k):
        if self.pmt.approximation == "acoustic" and self.pmt.ABC == "cerjan":
            self.wf.futurebck = updateWaveEquation(self.wf.futurebck, self.wf.currentbck, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]
            # Apply absorbing boundary condition
            self.wf.futurebck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.futurebck, self.A)
            self.wf.currentbck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.currentbck, self.A)

        elif self.pmt.approximation == "acoustic" and self.pmt.ABC == "CPML":
            self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            self.wf.futurebck = updateWaveEquationCPML(self.wf.futurebck, self.wf.currentbck, self.vp_exp, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)             
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]

        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "cerjan":
            self.wf.futurebck= updateWaveEquationVTI(self.wf.futurebck, self.wf.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]
            # Apply absorbing boundary condition
            self.wf.futurebck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.futurebck, self.A)
            self.wf.currentbck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.currentbck, self.A)

        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "CPML":
            self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            self.wf.futurebck = updateWaveEquationVTICPML(self.wf.futurebck, self.wf.currentbck, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.pmt.nx_abc, self.pmt.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]

        elif self.pmt.approximation == "TTI" and self.pmt.ABC == "cerjan":
            self.wf.futurebck= updateWaveEquationTTI(self.wf.futurebck, self.wf.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]
            # Apply absorbing boundary condition
            self.wf.futurebck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.futurebck, self.A)
            self.wf.currentbck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.currentbck, self.A)

    def save_boundaries(self,k): 
        if self.pmt.migration == "SB":
            self.wf.top[k,:,:]   = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + 4, self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]
            self.wf.bot[k,:,:]   = self.wf.future[self.pmt.N_abc + self.pmt.nz - 4: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]
            self.wf.left[k,:,:]  = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc+4]
            self.wf.right[k,:,:] = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz,self.pmt.N_abc + self.pmt.nx - 4: self.pmt.N_abc + self.pmt.nx]

    def apply_boundaries(self,k):
        if self.pmt.migration == "SB":
            self.wf.future[self.pmt.N_abc: self.pmt.N_abc + 4, self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx] = self.wf.top[k,:,:]
            self.wf.future[self.pmt.N_abc + self.pmt.nz - 4: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]  = self.wf.bot[k,:,:]
            self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc+4] = self.wf.left[k,:,:] 
            self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz,self.pmt.N_abc + self.pmt.nx - 4: self.pmt.N_abc + self.pmt.nx] = self.wf.right[k,:,:]

    def build_ckpts_steps(self):
        self.ckpts_steps = []
        for t0 in range (self.stop,self.pmt.nt-1,self.pmt.step):
            t1 = min(t0 + self.pmt.step,self.pmt.nt-1)
            self.ckpts_steps.append((t0,t1))
        self.ckpt_frames = {t1 for (t0, t1) in self.ckpts_steps}
   
    def reset_field(self):
        self.wf.current.fill(0)
        self.wf.future.fill(0)
        self.wf.currentbck.fill(0)
        self.wf.futurebck.fill(0)
        if self.pmt.ABC == "CPML":
            self.wf.PsixFR.fill(0)
            self.wf.PsixFL.fill(0)
            self.wf.PsizFU.fill(0)  
            self.wf.PsizFD.fill(0) 
            self.wf.ZetaxFR.fill(0)
            self.wf.ZetaxFL.fill(0)
            self.wf.ZetazFU.fill(0)
            self.wf.ZetazFD.fill(0)

    def get_randomvalue(self,velocity, func, par):
        point = np.random.normal(velocity, par*func)    
        value = par if point < par else (par + velocity 
                    if point > par + velocity else point)
        return value

    def poisson_disk_sampling(self, x_max, z_max, radius, k=30):
        cell_size = radius / np.sqrt(2)
        grid_width = int(x_max / cell_size) + 1
        grid_height = int(z_max / cell_size) + 1
        grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
        
        def get_cell_coords(p):
            return int(p[0] / cell_size), int(p[1] / cell_size)

        def is_valid(p):
            gx, gy = get_cell_coords(p)
            for i in range(max(gx - 2, 0), min(gx + 3, grid_width)):
                for j in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                    neighbor = grid[i][j]
                    if neighbor is not None:
                        dx, dy = neighbor[0] - p[0], neighbor[1] - p[1]
                        if dx * dx + dy * dy < radius * radius:
                            return False
            return True

        def generate_random_point_around(p):
            r = random.uniform(radius, 2 * radius)
            angle = random.uniform(0, 2 * np.pi)
            return (p[0] + r * np.cos(angle), p[1] + r * np.sin(angle))

        # Initial point
        p0 = (random.uniform(0, x_max), random.uniform(0, z_max))
        process_list = [p0]
        samples = [p0]
        gx, gy = get_cell_coords(p0)
        grid[gx][gy] = p0

        while process_list:
            idx = random.randint(0, len(process_list) - 1)
            p = process_list[idx]
            found = False
            for _ in range(k):
                q = generate_random_point_around(p)
                if 0 <= q[0] < x_max and 0 <= q[1] < z_max and is_valid(q):
                    process_list.append(q)
                    samples.append(q)
                    gx, gy = get_cell_coords(q)
                    grid[gx][gy] = q
                    found = True
            if not found:
                process_list.pop(idx)

        return samples

    def create_random_boundary(self):
        f1d = np.linspace(0, 1, self.pmt.N_abc)
        vmax = self.vp_exp.max()
        vmin = self.vp_exp.min()
        dvel = 1500
        ratio = 300
        boundary_x = self.pmt.N_abc * self.pmt.dx
        boundary_z = self.pmt.N_abc * self.pmt.dz
        L_abc = (self.pmt.nx_abc * self.pmt.dx) - self.pmt.dx
        D_abc = (self.pmt.nz_abc * self.pmt.dz) - self.pmt.dz
        N_abc = self.pmt.N_abc * self.pmt.dx
        rectangle = np.array([[boundary_x, boundary_x], 
                      [boundary_x, D_abc - boundary_x], 
                      [L_abc - boundary_x, D_abc - boundary_x],
                      [L_abc - boundary_x, boundary_x],
                      [boundary_x, boundary_x]])
        for i in range(self.pmt.nz):
            for j in range(self.pmt.N_abc):
                self.vp_exp[self.pmt.N_abc+i,j] = self.get_randomvalue(self.vp_exp[self.pmt.N_abc+i,self.pmt.N_abc], f1d[self.pmt.N_abc-j-1], dvel) 
                self.vp_exp[self.pmt.N_abc+i,self.pmt.nx_abc-j-1] = self.get_randomvalue(self.vp_exp[self.pmt.N_abc+i,self.pmt.nx_abc-self.pmt.N_abc], f1d[self.pmt.N_abc-j-1], dvel)

        for i in range(self.pmt.N_abc):
            for j in range(self.pmt.nx):
                self.vp_exp[i,self.pmt.N_abc+j] = self.get_randomvalue(self.vp_exp[self.pmt.N_abc,self.pmt.N_abc+j], f1d[self.pmt.N_abc-i-1], dvel)
                self.vp_exp[self.pmt.nz_abc-i-1,self.pmt.N_abc+j] = self.get_randomvalue(self.vp_exp[self.pmt.nz_abc-self.pmt.N_abc,self.pmt.N_abc+j], f1d[self.pmt.N_abc-i-1], dvel)

        for i in range(self.pmt.N_abc):
            for j in range(i,self.pmt.N_abc):
                self.vp_exp[j,i] = self.get_randomvalue(self.vp_exp[self.pmt.N_abc,self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], dvel)
                self.vp_exp[i,j] = self.get_randomvalue(self.vp_exp[self.pmt.N_abc,self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], dvel)

                self.vp_exp[j,self.pmt.nx_abc-i-1] = self.get_randomvalue(self.vp_exp[self.pmt.N_abc,self.pmt.nx_abc-self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], dvel)
                self.vp_exp[i,self.pmt.nx_abc-j-1] = self.get_randomvalue(self.vp_exp[self.pmt.N_abc,self.pmt.nx_abc-self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], dvel)

                self.vp_exp[self.pmt.nz_abc-j-1,i] = self.get_randomvalue(self.vp_exp[self.pmt.nz_abc-self.pmt.N_abc,self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], dvel)
                self.vp_exp[self.pmt.nz_abc-i-1,j] = self.get_randomvalue(self.vp_exp[self.pmt.nz_abc-self.pmt.N_abc,self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], dvel)

                self.vp_exp[self.pmt.nz_abc-j-1,self.pmt.nx_abc-i-1] = self.get_randomvalue(self.vp_exp[self.pmt.nz_abc-self.pmt.N_abc,self.pmt.nx_abc-self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], dvel)
                self.vp_exp[self.pmt.nz_abc-i-1,self.pmt.nx_abc-j-1] = self.get_randomvalue(self.vp_exp[self.pmt.nz_abc-self.pmt.N_abc,self.pmt.nx_abc-self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], dvel)

        self.vp_exp[np.where(self.vp_exp > vmax + dvel)] = vmax + dvel
        self.vp_exp[np.where(self.vp_exp < vmin - dvel)] = vmin - dvel

        points = self.poisson_disk_sampling(L_abc, D_abc, ratio)

        points = np.array(points)

        x_mask = np.logical_or(points[:,0] < 0.5*N_abc, points[:,0] > self.pmt.L + N_abc + 0.5*N_abc)
        z_mask = np.logical_or(points[:,1] < 0.5*N_abc, points[:,1] > self.pmt.D + N_abc + 0.5*N_abc)

        mask = np.logical_or(x_mask, z_mask)

        x, z = np.meshgrid(np.arange(self.pmt.nx_abc)*self.pmt.dx, np.arange(self.pmt.nz_abc)*self.pmt.dz) 

        points = points[mask]

        for index in range(len(points)):

            xc = points[index, 0]  
            zc = points[index, 1] 

            r = np.random.uniform(0.1*ratio, ratio)
            A = np.random.uniform(0.5*dvel, dvel)

            factor = np.random.choice([-1,1])

            self.vp_exp = self.vp_exp + factor*A*np.exp(-0.5*(((x - xc) / r)**2 + ((z - zc) / r)**2))
            
        self.vp_exp[np.where(self.vp_exp > vmax + dvel)] = vmax + dvel
        self.vp_exp[np.where(self.vp_exp < vmin - dvel)] = vmin - dvel

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(self.vp_exp, aspect = "auto", cmap = "jet", vmax = vmax + dvel, vmin = vmin - dvel, extent = [0, L_abc, D_abc, 0])
        plt.plot(rectangle[:,0], rectangle[:,1], "--k")
        plt.show()

    #On the fly
    def solveBackwardWaveEquationOntheFly(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.d0, self.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)
        
        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        save_field = np.zeros([self.pmt.nt,self.pmt.nz,self.pmt.nx],dtype=np.float32)
        self.stop = int(self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")

            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.3,window = 0.3,v0=1500)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.forward_step(k)
                save_field[k,:,:] = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.backward_step(t)
                self.migrated_partial += (save_field[t,:,:] * self.wf.futurebck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.ilum += save_field[t,:,:] * save_field[t,:,:]
                #swap
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum + 1e-12)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
     
        # Apply laplacian_filter filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    # REGULAR CHECKPOINTING
    def solveBackwardWaveEquationCheckpointing(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.d0, self.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)
        
        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        self.stop = int(self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.3,window = 0.3,v0=1500)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            self.build_ckpts_steps()
            for k in range(self.pmt.nt):
                self.forward_step(k)
                self.save_checkpoint(shot, k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for (t0,t1) in reversed(self.ckpts_steps):
                self.load_checkpoint(shot,t1)
                for t in range(t1, t0, -1):
                    self.forward_step(t)
                    self.backward_step(t)
                    self.migrated_partial += (self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.futurebck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                    self.ilum += self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current
                    self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum + 1e-12)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
     
        # Apply laplacian_filter filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    #Saving Boundaries
    def solveBackwardWaveEquationSavingBoundaries(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.d0, self.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)

        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        self.stop = int(self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.2 ,window = 0.3,v0=1500)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.save_boundaries(k)
                self.forward_step(k)            
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current 
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.forward_step(t)
                self.apply_boundaries(t)           
                self.backward_step(t)
                self.migrated_partial += (self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.futurebck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])  
                self.ilum += self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck

            self.wf.migrated_image += self.migrated_partial / (self.ilum + 1e-12)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
     
        # Apply laplacian_filter filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    #Random Boundary Condintion
    def solveBackwardWaveEquationRBC(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.d0, self.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)

        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        self.stop = int(self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()
            self.create_random_boundary()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.2 ,window = 0.3,v0=1500)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.forward_step(k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            self.wf.current, self.wf.future = self.wf.future, self.wf.current    
            for t in range(self.pmt.nt - 1, self.stop, -1): 
                self.forward_step(t)
                self.backward_step(t)     
                self.migrated_partial += (self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.futurebck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.ilum += self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck

            self.wf.migrated_image += self.migrated_partial / (self.ilum + 1e-12)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
     
        # Apply Laplacian filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    def SolveBackwardWaveEquation(self):
        if self.pmt.migration == "onthefly":
            self.solveBackwardWaveEquationOntheFly()
        elif self.pmt.migration == "checkpoint":
            self.solveBackwardWaveEquationCheckpointing()
        elif self.pmt.migration == "SB":
            self.solveBackwardWaveEquationSavingBoundaries()
        elif self.pmt.migration == "RBC":
            self.solveBackwardWaveEquationRBC()
        else:
            raise ValueError("Unknown migration method. Choose 'onthefly','checkpoint' or 'SB'.")
        print(f"info: Migration solved")


