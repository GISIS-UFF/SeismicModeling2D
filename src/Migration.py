import numpy as np
import time
import random
import cupy as cp

from utils import updateWaveEquation
from utils import updateWaveEquationCPML
from utils import updateWaveEquationVTI
from utils import updateWaveEquationVTICPML
from utils import updateWaveEquationTTI
from utils import AbsorbingBoundary
from utils import updatePsi
from utils import updateZeta
from utils import Mute
from utils import updateWaveEquationGPU
from utils import updateWaveEquationCPMLGPU
from utils import updateWaveEquationVTIGPU
from utils import updateWaveEquationVTICPMLGPU
from utils import updateWaveEquationTTIGPU
from utils import AbsorbingBoundaryGPU
from utils import updatePsiGPU
from utils import updateZetaGPU
from mpl_toolkits.axes_grid1 import make_axes_locatable

class migration: 

    def __init__(self,wavefield,parameters):
        self.pmt = parameters
        self.wf = wavefield

    def adjustColorBar(self,fig,ax,im):
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append an axes to the right of the current axes, with the same height
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im,cax=cax)
        return cbar
    
    def loadSeismogram(self, shot):
        seismogramFile = f"{self.pmt.seismogramFolder}seismogram_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        seismogram = np.fromfile(seismogramFile, dtype=np.float32).reshape(self.pmt.nt,self.pmt.Nrec) 
        return seismogram
    
    def save_snapshotBCK(self,shot, k):        
        if not self.pmt.snap:
            return
        if k > self.pmt.last_save:
            return
        if k % self.pmt.step != 0:
            return

        snapshot = self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]

        snapshotFile = (f"{self.pmt.snapshotFolder}{self.pmt.approximation}backward_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")
        snapshot.tofile(snapshotFile)
        print(f"info: Snapshot saved to {snapshotFile}")
    
    def store_snapshotBCKGPU(self, k):        
        if not self.pmt.snap:
            return
        if k > self.pmt.last_save:
            return
        if k % self.pmt.step != 0:
            return
        
        snapshot = self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
        self.wf.snapshots_gpu[self.wf.snap_idx, :, :] = snapshot
        self.wf.snap_idx += 1

    def save_snapshotBCKGPU(self,shot):        
        if not self.pmt.snap:
            return
        snapshots_cpu = cp.asnumpy(self.wf.snapshots_gpu[:self.snap_idx,:,:])
        for i, k in enumerate(self.wf.snap_times[:self.snap_idx]):
            snapshotFile = (f"{self.pmt.snapshotFolder}{self.pmt.approximation}backward_shot_{shot+1}"f"_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")
            snapshots_cpu[i].tofile(snapshotFile)
            print(f"info: Snapshot saved to {snapshotFile}")

    def save_image(self,shot, k):        
        if not self.pmt.snap:
            return
        if k > self.pmt.last_save:
            return
        if k % self.pmt.step != 0:
            return

        img = self.migrated_partial

        imageFile = (f"{self.pmt.migratedimageFolder}{self.pmt.approximation}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_frame_{k}.bin")
        img.tofile(imageFile)
        print(f"info: Snapshot saved to {imageFile}")
    
    def store_imageGPU(self, k):        
        if not self.pmt.snap:
            return
        if k > self.pmt.last_save:
            return
        if k % self.pmt.step != 0:
            return
        
        img = self.migrated_partial
        self.wf.img_gpu[self.wf.img_idx, :, :] = img
        self.wf.img_idx += 1

    def save_imageGPU(self,shot):        
        if not self.pmt.snap:
            return
        img_cpu = cp.asnumpy(self.wf.img_gpu[:self.img_idx,:,:])
        for i, k in enumerate(self.wf.img_times[:self.img_idx]):
            imageFile = (f"{self.pmt.migratedimageFolder}{self.pmt.approximation}_shot_{shot+1}"f"_Nx{self.pmt.nx}_Nz{self.pmt.nz}_frame_{k}.bin")
            img_cpu[i].tofile(imageFile)
            print(f"info: Snapshot saved to {imageFile}")
    
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
        if self.pmt.unit == "GPU":
            self.wf.current = cp.asarray(self.wf.current)
            self.wf.future = cp.asarray(self.wf.future)

    def save_checkpoint(self, shot, k):
        if self.pmt.migration != "checkpoint":
            return
        if k not in self.ckpt_frames:
            return
        
        checkpointFile = (f"{self.pmt.checkpointFolder}{self.pmt.approximation}{self.pmt.ABC}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")
        if self.pmt.unit == "GPU":
            self.wf.current = cp.asnumpy(self.wf.current)
            self.wf.future = cp.asnumpy(self.wf.future)
        save = [self.wf.current, self.wf.future]
        with open(checkpointFile, "wb") as file:
            for field in save:
                field.astype(np.float32).tofile(file)

        print(f"info: Checkpoint saved to {checkpointFile}")

    def forward_step(self,k):
        if self.pmt.migration in ["onthefly", "checkpoint", "SB"]:
            if self.pmt.approximation == "acoustic" and self.pmt.ABC == "cerjan":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.current, self.A)
            elif self.pmt.approximation == "acoustic" and self.pmt.ABC == "CPML":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.future = updateWaveEquationCPML(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)             
            elif self.pmt.approximation == "VTI" and self.pmt.ABC == "cerjan":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                self.wf.future= updateWaveEquationVTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.current, self.A)
            elif self.pmt.approximation == "VTI" and self.pmt.ABC == "CPML":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.future = updateWaveEquationVTICPML(self.wf.future, self.wf.current, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.pmt.nx_abc, self.pmt.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)
            elif self.pmt.approximation == "TTI" and self.pmt.ABC == "cerjan":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                self.wf.future= updateWaveEquationTTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.current, self.A)
        if self.pmt.migration == "RBC":
            if self.pmt.approximation == "acoustic":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            elif self.pmt.approximation == "VTI":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                self.wf.future = updateWaveEquationVTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            elif self.pmt.approximation == "TTI":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                self.wf.future = updateWaveEquationTTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)

    def forward_stepGPU(self,k):
        if self.pmt.migration in ["onthefly", "checkpoint", "SB"]:
            if self.pmt.approximation == "acoustic" and self.pmt.ABC == "cerjan":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                updateWaveEquationGPU(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
                # Apply absorbing boundary condition
                self.wf.future, self.wf.current = AbsorbingBoundaryGPU(self.wf.future,self.wf.current,self.pmt.N_abc,self.pmt.nx_abc,self.pmt.nz_abc, self.A)
            elif self.pmt.approximation == "acoustic" and self.pmt.ABC == "CPML":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                updatePsiGPU(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                updateZetaGPU(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                updateWaveEquationCPMLGPU(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)             
            elif self.pmt.approximation == "VTI" and self.pmt.ABC == "cerjan":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                updateWaveEquationVTIGPU(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                # Apply absorbing boundary condition
                self.wf.future, self.wf.current = AbsorbingBoundaryGPU(self.wf.future,self.wf.current,self.pmt.N_abc,self.pmt.nx_abc,self.pmt.nz_abc, self.A)
            elif self.pmt.approximation == "VTI" and self.pmt.ABC == "CPML":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                updatePsiGPU(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                updateZetaGPU(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                updateWaveEquationVTICPMLGPU(self.wf.future, self.wf.current, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.pmt.nx_abc, self.pmt.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)
            elif self.pmt.approximation == "TTI" and self.pmt.ABC == "cerjan":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                updateWaveEquationTTIGPU(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
                # Apply absorbing boundary condition
                self.wf.future,self.wf.current = AbsorbingBoundaryGPU(self.wf.future,self.wf.current,self.pmt.N_abc,self.pmt.nx_abc,self.pmt.nz_abc, self.A)
        if self.pmt.migration == "RBC":
            if self.pmt.approximation == "acoustic":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                updateWaveEquationGPU(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            elif self.pmt.approximation == "VTI":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                updateWaveEquationVTIGPU(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            elif self.pmt.approximation == "TTI":
                self.wf.current[self.sz,self.sx] += self.wf.source[k]
                updateWaveEquationTTIGPU(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
     
    def reconstructed_step(self):
            if self.pmt.approximation == "acoustic":
                self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            elif self.pmt.approximation == "VTI":
                self.wf.future = updateWaveEquationVTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            elif self.pmt.approximation == "TTI":
                self.wf.future = updateWaveEquationTTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)

    def reconstructed_stepGPU(self):
            if self.pmt.approximation == "acoustic":
                updateWaveEquationGPU(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            elif self.pmt.approximation == "VTI":
                updateWaveEquationVTIGPU(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            elif self.pmt.approximation == "TTI":
                updateWaveEquationTTIGPU(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)

    def backward_step(self,k):
        if self.pmt.approximation == "acoustic" and self.pmt.ABC == "cerjan":
            self.wf.currentbck[self.rz, self.rx] += self.muted_seismogram[k, :]
            self.wf.futurebck = updateWaveEquation(self.wf.futurebck, self.wf.currentbck, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            # Apply absorbing boundary condition
            self.wf.futurebck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.futurebck, self.A)
            self.wf.currentbck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.currentbck, self.A)
        elif self.pmt.approximation == "acoustic" and self.pmt.ABC == "CPML":
            self.wf.currentbck[self.rz, self.rx] += self.muted_seismogram[k, :]
            self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            self.wf.futurebck = updateWaveEquationCPML(self.wf.futurebck, self.wf.currentbck, self.vp_exp, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)                         
        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "cerjan":
            self.wf.currentbck[self.rz, self.rx] += self.muted_seismogram[k, :]
            self.wf.futurebck = updateWaveEquationVTI(self.wf.futurebck, self.wf.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            # Apply absorbing boundary condition
            self.wf.futurebck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.futurebck, self.A)
            self.wf.currentbck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.currentbck, self.A)
        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "CPML":
            self.wf.currentbck[self.rz, self.rx] += self.muted_seismogram[k, :]
            self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            self.wf.futurebck = updateWaveEquationVTICPML(self.wf.futurebck, self.wf.currentbck, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.pmt.nx_abc, self.pmt.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)
        elif self.pmt.approximation == "TTI" and self.pmt.ABC == "cerjan":
            self.wf.currentbck[self.rz, self.rx] += self.muted_seismogram[k, :]
            self.wf.futurebck = updateWaveEquationTTI(self.wf.futurebck, self.wf.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
            # Apply absorbing boundary condition
            self.wf.futurebck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.futurebck, self.A)
            self.wf.currentbck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.currentbck, self.A)

    def backward_stepGPU(self,k):
        if self.pmt.approximation == "acoustic" and self.pmt.ABC == "cerjan":
            self.wf.currentbck[self.rz, self.rx] += self.muted_seismogram[k, :]
            updateWaveEquationGPU(self.wf.futurebck, self.wf.currentbck, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            # Apply absorbing boundary condition
            self.wf.futurebck, self.wf.currentbck = AbsorbingBoundaryGPU(self.wf.futurebck,self.wf.currentbck,self.pmt.N_abc,self.pmt.nx_abc,self.pmt.nz_abc, self.A)
        elif self.pmt.approximation == "acoustic" and self.pmt.ABC == "CPML":
            self.wf.currentbck[self.rz, self.rx] += self.muted_seismogram[k, :]
            updatePsiGPU(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            updateZetaGPU(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            updateWaveEquationCPMLGPU(self.wf.futurebck, self.wf.currentbck, self.vp_exp, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)                         
        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "cerjan":
            self.wf.currentbck[self.rz, self.rx] += self.muted_seismogram[k, :]
            updateWaveEquationVTIGPU(self.wf.futurebck, self.wf.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            # Apply absorbing boundary condition
            self.wf.futurebck, self.wf.currentbck = AbsorbingBoundaryGPU(self.wf.futurebck,self.wf.currentbck,self.pmt.N_abc,self.pmt.nx_abc,self.pmt.nz_abc, self.A)
        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "CPML":
            self.wf.currentbck[self.rz, self.rx] += self.muted_seismogram[k, :]
            updatePsiGPU(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            updateZetaGPU(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.currentbck, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
            updateWaveEquationVTICPMLGPU(self.wf.futurebck, self.wf.currentbck, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.pmt.nx_abc, self.pmt.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)
        elif self.pmt.approximation == "TTI" and self.pmt.ABC == "cerjan":
            self.wf.currentbck[self.rz, self.rx] += self.muted_seismogram[k, :]
            updateWaveEquationTTIGPU(self.wf.futurebck, self.wf.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
            # Apply absorbing boundary condition
            self.wf.futurebck, self.wf.currentbck = AbsorbingBoundaryGPU(self.wf.futurebck,self.wf.currentbck,self.pmt.N_abc,self.pmt.nx_abc,self.pmt.nz_abc, self.A)

    def save_boundaries(self,k): 
        self.wf.top[k,:,:]   = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + 4, self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]
        self.wf.bot[k,:,:]   = self.wf.future[self.pmt.N_abc + self.pmt.nz - 4: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]
        self.wf.left[k,:,:]  = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc+4]
        self.wf.right[k,:,:] = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz,self.pmt.N_abc + self.pmt.nx - 4: self.pmt.N_abc + self.pmt.nx]

    def apply_boundaries(self,k):
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
        if self.pmt.unit == "GPU":
            if self.pmt.snap == True:
                self.wf.snapshots_gpu.fill(0) 
                self.wf.snap_idx = 0

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
        dvel = 1000
        ratio = 200
        boundary_x = self.pmt.N_abc * self.pmt.dx
        boundary_z = self.pmt.N_abc * self.pmt.dz
        L_abc = (self.pmt.nx_abc * self.pmt.dx) - self.pmt.dx
        D_abc = (self.pmt.nz_abc * self.pmt.dz) - self.pmt.dz
        N_abc = self.pmt.N_abc * self.pmt.dx
        rectangle = np.array([
            [boundary_x, boundary_z],                       
            [boundary_x, D_abc - boundary_z],                
            [L_abc - boundary_x, D_abc - boundary_z],        
            [L_abc - boundary_x, boundary_z],               
            [boundary_x, boundary_z]])
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
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(self.vp_exp, aspect = "auto", cmap = "jet", vmax = vmax + dvel, vmin = vmin - dvel, extent = [0, L_abc, D_abc, 0])
        plt.plot(rectangle[:,0], rectangle[:,1], "--k")
        cbar = self.adjustColorBar(fig, ax, im)
        cbar.set_label("Amplitude")
        plt.show()

## CPU Migration Types
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
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.8,window = 0.3,v0=1500)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.forward_step(k)
                self.wf.save_snapshot(shot, k)
                save_field[k,:,:] = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                self.ilum += save_field[k,:,:] * save_field[k,:,:] 
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.backward_step(t)
                self.save_snapshotBCK(shot,t)
                self.migrated_partial += (save_field[t,:,:] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.save_image(shot,t)
                #swap
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
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
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.75,window = 0.25,v0=1500)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            self.build_ckpts_steps()
            for k in range(self.pmt.nt):
                self.forward_step(k)
                self.wf.save_snapshot(shot, k)
                self.ilum += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] 
                self.save_checkpoint(shot, k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for (t0,t1) in reversed(self.ckpts_steps):
                self.load_checkpoint(shot,t1)
                for t in range(t1, t0, -1):
                    self.reconstructed_step()
                    self.backward_step(t)
                    self.save_snapshotBCK(shot,t)
                    self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                    self.save_image(shot,t)
                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current
                    self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
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
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.75,window = 0.25,v0=1500)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.save_boundaries(k)
                self.forward_step(k)
                self.wf.save_snapshot(shot, k)
                self.ilum += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]  
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current 
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.reconstructed_step()
                self.apply_boundaries(t)           
                self.backward_step(t)
                self.save_snapshotBCK(shot,t)
                self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])  
                self.save_image(shot,t)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")      
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    #Random Boundary Condintion
    def solveBackwardWaveEquationRBC(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        vp_exp_base = self.wf.ExpandModel(self.vp)
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
            self.vp_exp = vp_exp_base.copy()
            self.create_random_boundary()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.75,window = 0.25,v0=1500)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.forward_step(k)
                self.wf.save_snapshot(shot, k)
                self.ilum += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] 
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            self.wf.current, self.wf.future = self.wf.future, self.wf.current    
            for t in range(self.pmt.nt - 1, self.stop, -1): 
                self.reconstructed_step()
                self.backward_step(t) 
                self.save_snapshotBCK(shot,t)
                self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])    
                self.save_image(shot,t)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

## GPU Migration Types
    #On the fly
    def solveBackwardWaveEquationOntheFlyGPU(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        self.vp_exp = cp.asarray(self.vp_exp, dtype=cp.float32)
        if self.pmt.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
            self.A = cp.asarray(self.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.d0, self.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            self.epsilon_exp  = cp.asarray(self.epsilon_exp, dtype=cp.float32)
            self.delta_exp  = cp.asarray(self.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)
                self.theta_exp  = cp.asarray(self.theta_exp, dtype=cp.float32)
        
        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        self.rx = cp.asarray(self.rx)
        self.rz = cp.asarray(self.rz)
        save_field = cp.zeros([self.pmt.nt,self.pmt.nz,self.pmt.nx],dtype=np.float32)
        self.stop = int(self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.75,window = 0.25,v0=1500)
            self.muted_seismogram = cp.asarray(self.muted_seismogram,dtype=cp.float32)
            self.migrated_partial = cp.zeros_like(self.wf.migrated_image)
            self.ilum = cp.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.forward_stepGPU(k)
                self.wf.store_snapshotGPU(k) 
                save_field[k,:,:] = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                self.ilum += save_field[k,:,:] * save_field[k,:,:] 
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.backward_stepGPU(t)
                self.store_snapshotBCKGPU(t)
                self.migrated_partial += (save_field[t,:,:] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.store_imageGPU(t)
                #swap
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum)
            self.wf.save_snapshotGPU(shot)
            self.save_snapshotBCKGPU(shot)
            self.save_imageGPU(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        migrated_imagecpu = cp.asnumpy(self.wf.migrated_image)
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        migrated_imagecpu.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")
        
    # REGULAR CHECKPOINTING
    def solveBackwardWaveEquationCheckpointingGPU(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        self.vp_exp = cp.asarray(self.vp_exp, dtype=cp.float32)
        if self.pmt.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
            self.A = cp.asarray(self.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.d0, self.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            self.epsilon_exp  = cp.asarray(self.epsilon_exp, dtype=cp.float32)
            self.delta_exp  = cp.asarray(self.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)
                self.theta_exp  = cp.asarray(self.theta_exp, dtype=cp.float32)
        
        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        self.rx = cp.asarray(self.rx)
        self.rz = cp.asarray(self.rz)
        self.stop = int(self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.75,window = 0.25,v0=1500)
            self.muted_seismogram = cp.asarray(self.muted_seismogram,dtype=cp.float32)
            self.migrated_partial = cp.zeros_like(self.wf.migrated_image)
            self.ilum = cp.zeros_like(self.wf.migrated_image)
            self.build_ckpts_steps()
            for k in range(self.pmt.nt):
                self.forward_stepGPU(k)
                self.wf.store_snapshotGPU(k)
                self.ilum += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] 
                self.save_checkpoint(shot, k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for (t0,t1) in reversed(self.ckpts_steps):
                self.load_checkpoint(shot,t1)
                for t in range(t1, t0, -1):
                    self.reconstructed_stepGPU()
                    self.backward_stepGPU(t)
                    self.store_snapshotBCKGPU(t)
                    self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                    self.store_imageGPU(t)
                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current
                    self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum)
            self.wf.save_snapshotGPU(shot)
            self.save_snapshotBCKGPU(shot)
            self.save_imageGPU(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        migrated_imagecpu = cp.asnumpy(self.wf.migrated_image)
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        migrated_imagecpu.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")
    
    #Saving Boundaries
    def solveBackwardWaveEquationSavingBoundariesGPU(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        self.vp_exp = cp.asarray(self.vp_exp, dtype=cp.float32)
        if self.pmt.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
            self.A = cp.asarray(self.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.d0, self.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            self.epsilon_exp  = cp.asarray(self.epsilon_exp, dtype=cp.float32)
            self.delta_exp  = cp.asarray(self.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)
                self.theta_exp  = cp.asarray(self.theta_exp, dtype=cp.float32)

        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        self.rx = cp.asarray(self.rx)
        self.rz = cp.asarray(self.rz)
        self.stop = int(self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.75 ,window = 0.25,v0=1500)
            self.muted_seismogram = cp.asarray(self.muted_seismogram,dtype=cp.float32)
            self.migrated_partial = cp.zeros_like(self.wf.migrated_image)
            self.ilum = cp.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.save_boundaries(k)
                self.forward_stepGPU(k)
                self.wf.store_snapshotGPU(k)
                self.ilum += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] 
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current 
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.reconstructed_stepGPU()
                self.apply_boundaries(t)           
                self.backward_stepGPU(t)
                self.store_snapshotBCKGPU(t)
                self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])  
                self.store_imageGPU(t)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum)
            self.wf.save_snapshotGPU(shot)
            self.save_snapshotBCKGPU(shot)
            self.save_imageGPU(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        migrated_imagecpu = cp.asnumpy(self.wf.migrated_image)      
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        migrated_imagecpu.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    #Random Boundary Condintion
    def solveBackwardWaveEquationRBCGPU(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        vp_exp_base = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
            self.A = cp.asarray(self.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.d0, self.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            self.epsilon_exp  = cp.asarray(self.epsilon_exp, dtype=cp.float32)
            self.delta_exp  = cp.asarray(self.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)
                self.theta_exp  = cp.asarray(self.theta_exp, dtype=cp.float32)

        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        self.rx = cp.asarray(self.rx)
        self.rz = cp.asarray(self.rz)
        self.stop = int(self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()
            self.vp_exp = vp_exp_base.copy()
            self.create_random_boundary()
            self.vp_exp = cp.asarray(self.vp_exp, dtype=cp.float32)

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt, shift = 0.75,window = 0.25,v0=1500)
            self.muted_seismogram = cp.asarray(self.muted_seismogram,dtype=cp.float32)
            self.migrated_partial = cp.zeros_like(self.wf.migrated_image)
            self.ilum = cp.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.forward_stepGPU(k)
                self.wf.store_snapshotGPU(k)
                self.ilum += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            self.wf.current, self.wf.future = self.wf.future, self.wf.current    
            for t in range(self.pmt.nt - 1, self.stop, -1): 
                self.reconstructed_stepGPU()
                self.backward_stepGPU(t)
                self.store_snapshotBCKGPU(t) 
                self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])    
                self.store_imageGPU(t)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum)
            self.wf.save_snapshotGPU(shot)
            self.save_snapshotBCKGPU(shot)
            self.save_imageGPU(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        migrated_imagecpu = cp.asnumpy(self.wf.migrated_image)
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        migrated_imagecpu.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    def SolveBackwardWaveEquation(self):
        if self.pmt.unit == "CPU":
            if self.pmt.migration == "onthefly":
                self.solveBackwardWaveEquationOntheFly()
            elif self.pmt.migration == "checkpoint":
                self.solveBackwardWaveEquationCheckpointing()
            elif self.pmt.migration == "SB":
                self.solveBackwardWaveEquationSavingBoundaries()
            elif self.pmt.migration == "RBC":
                self.solveBackwardWaveEquationRBC()
        elif self.pmt.unit == "GPU":
            if self.pmt.migration == "onthefly":
                self.solveBackwardWaveEquationOntheFlyGPU()
            elif self.pmt.migration == "checkpoint":
                self.solveBackwardWaveEquationCheckpointingGPU()
            elif self.pmt.migration == "SB":
                self.solveBackwardWaveEquationSavingBoundariesGPU()
            elif self.pmt.migration == "RBC":
                self.solveBackwardWaveEquationRBCGPU()
        else:
            raise ValueError("Unknown migration method. Choose 'onthefly','checkpoint' or 'SB'.")
        print(f"info: Migration solved")