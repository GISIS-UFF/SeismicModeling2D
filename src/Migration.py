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
from utils import smooth_model
from mpl_toolkits.axes_grid1 import make_axes_locatable

class migration: 
    def __init__(self,wavefield,parameters):
        self.pmt = parameters
        self.wf = wavefield
    
    def initializeMigrationfields(self):
        self.migrated_image = np.zeros((self.pmt.nz, self.pmt.nx), dtype=np.float32)
        self.ilum = np.zeros((self.pmt.nz, self.pmt.nx), dtype=np.float32)
        self.currentbck  = np.zeros([self.pmt.nz_abc,self.pmt.nx_abc],dtype=np.float32)
        self.futurebck   = np.zeros([self.pmt.nz_abc,self.pmt.nx_abc],dtype=np.float32)
        if self.pmt.migration == "SB":
            self.top   = np.zeros((self.pmt.nt, 4, self.pmt.nx), dtype=np.float32)
            self.bot   = np.zeros((self.pmt.nt, 4, self.pmt.nx), dtype=np.float32)
            self.left  = np.zeros((self.pmt.nt, self.pmt.nz, 4), dtype=np.float32)
            self.right = np.zeros((self.pmt.nt, self.pmt.nz, 4), dtype=np.float32)
            if self.pmt.unit == "GPU":
                self.top   = cp.zeros((self.pmt.nt, 4, self.pmt.nx), dtype=np.float32)
                self.bot   = cp.zeros((self.pmt.nt, 4, self.pmt.nx), dtype=np.float32)
                self.left  = cp.zeros((self.pmt.nt, self.pmt.nz, 4), dtype=np.float32)
                self.right = cp.zeros((self.pmt.nt, self.pmt.nz, 4), dtype=np.float32)
        if self.pmt.unit == "GPU":
            self.migrated_image = cp.zeros((self.pmt.nz, self.pmt.nx), dtype=np.float32)
            self.currentbck  = cp.zeros([self.pmt.nz_abc,self.pmt.nx_abc],dtype=np.float32)
            self.futurebck   = cp.zeros([self.pmt.nz_abc,self.pmt.nx_abc],dtype=np.float32)
            if self.pmt.snap == True:
                self.snapshots_gpubck = cp.zeros((self.wf.nsnaps, self.pmt.nz, self.pmt.nx), dtype=cp.float32)
                self.snap_idxbck = 0
                self.img_times = list(range(0, self.pmt.last_save + 1, self.pmt.step))
                self.nimg = len(self.wf.snap_times)
                self.img_gpu = cp.zeros((self.wf.nsnaps, self.pmt.nz, self.pmt.nx), dtype=cp.float32)
            if self.pmt.migration == "SB":
                self.top   = cp.zeros((self.pmt.nt, 4, self.pmt.nx), dtype=np.float32)
                self.bot   = cp.zeros((self.pmt.nt, 4, self.pmt.nx), dtype=np.float32)
                self.left  = cp.zeros((self.pmt.nt, self.pmt.nz, 4), dtype=np.float32)
                self.right = cp.zeros((self.pmt.nt, self.pmt.nz, 4), dtype=np.float32)
            if self.pmt.migration == "checkpoint":
                self.ckpt_list = []
                self.ckpt_file = []
                self.ckpt_list_size = 16
        print(f"info: Wavefields initialized: {self.pmt.nx}x{self.pmt.nz}x{self.pmt.nt}")

    def adjustColorBar(self,fig,ax,im):
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append an axes to the right of the current axes, with the same height
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im,cax=cax)
        return cbar

    def loadSeismogram(self, shot):
        if self.pmt.fwi  == True:
            seismogramFile = f"{self.pmt.seismogramFolder}residual_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        else:
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

        snapshot = self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]

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
        
        snapshot = self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
        self.snapshots_gpubck[self.snap_idxbck, :, :] = snapshot
        self.snap_idxbck += 1

    def save_snapshotBCKGPU(self,shot):        
        if not self.pmt.snap:
            return
        self.snap_times_reversed = self.wf.snap_times[::-1]
        snapshots_cpu = cp.asnumpy(self.snapshots_gpubck[:self.snap_idxbck,:,:])
        for i, k in enumerate(self.snap_times_reversed[:self.snap_idxbck]):
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
        print(f"info: Image saved to {imageFile}")
    
    def store_imageGPU(self, k):        
        if not self.pmt.snap:
            return
        if k > self.pmt.last_save:
            return
        if k % self.pmt.step != 0:
            return
        
        img = self.migrated_partial
        self.img_gpu[self.wf.img_idx, :, :] = img
        self.wf.img_idx += 1

    def save_imageGPU(self,shot):        
        if not self.pmt.snap:
            return
        img_cpu = cp.asnumpy(self.img_gpu[:self.wf.img_idx,:,:])
        for i, k in enumerate(self.img_times[:self.wf.img_idx]):
            imageFile = (f"{self.pmt.migratedimageFolder}{self.pmt.approximation}_shot_{shot+1}"f"_Nx{self.pmt.nx}_Nz{self.pmt.nz}_frame_{k}.bin")
            img_cpu[i].tofile(imageFile)
            print(f"info: Image saved to {imageFile}")
                   
    def load_checkpoint(self, shot, k):
        checkpointFile = (f"{self.pmt.checkpointFolder}{self.pmt.approximation}{self.pmt.ABC}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")
        with open(checkpointFile, "rb") as file:
            count = self.pmt.nx_abc * self.pmt.nz_abc
            self.wf.current = np.fromfile(file, np.float32, count).reshape(self.pmt.nz_abc, self.pmt.nx_abc)
            self.wf.future  = np.fromfile(file, np.float32, count).reshape(self.pmt.nz_abc, self.pmt.nx_abc)
        
        if self.pmt.unit == "GPU":
            self.wf.current = cp.asarray(self.wf.current, dtype=cp.float32)
            self.wf.future  = cp.asarray(self.wf.future, dtype=cp.float32)

    def flush_checkpointGPU(self):
        if len(self.ckpt_list) == 0:
            return

        for i in range(len(self.ckpt_list)):
            current_gpu, future_gpu = self.ckpt_list[i]
            checkpointFile = self.ckpt_file[i]

            with open(checkpointFile, "wb") as file:
                cp.asnumpy(current_gpu).astype(np.float32).tofile(file)
                cp.asnumpy(future_gpu).astype(np.float32).tofile(file)

            print(f"info: Checkpoint saved to {checkpointFile}")

        self.ckpt_list = []
        self.ckpt_file = []

    def save_checkpointGPU(self, shot, k):
        if self.pmt.migration != "checkpoint":
            return
        if k not in self.ckpt_frames:
            return

        checkpointFile = (f"{self.pmt.checkpointFolder}{self.pmt.approximation}{self.pmt.ABC}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")

        self.ckpt_list.append((self.wf.current.copy(), self.wf.future.copy()))
        self.ckpt_file.append(checkpointFile)
        if len(self.ckpt_list) >= self.ckpt_list_size:
            self.flush_checkpointGPU()


    def save_checkpoint(self, shot, k):
        if self.pmt.migration != "checkpoint":
            return
        if k not in self.ckpt_frames:
            return

        checkpointFile = (f"{self.pmt.checkpointFolder}{self.pmt.approximation}{self.pmt.ABC}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")
        save = [self.wf.current, self.wf.future]
        with open(checkpointFile, "wb") as file:
            for field in save:
                field.astype(np.float32).tofile(file)
                                
        print(f"info: Checkpoint saved to {checkpointFile}")

    def forward_step_RBC(self,k):
        if self.pmt.approximation == "acoustic":
            self.wf.current[self.wf.isz,self.wf.isx] += self.wf.source[k]
            self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.wf.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
        elif self.pmt.approximation == "VTI":
            self.wf.current[self.wf.isz,self.wf.isx] += self.wf.source[k]
            self.wf.future = updateWaveEquationVTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp)
        elif self.pmt.approximation == "TTI":
            self.wf.current[self.wf.isz,self.wf.isx] += self.wf.source[k]
            self.wf.future = updateWaveEquationTTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp, self.wf.theta_exp)

    def forward_stepGPU_RBC(self,k):
        if self.pmt.approximation == "acoustic":
            self.wf.current[self.wf.isz,self.wf.isx] += self.wf.source[k]
            updateWaveEquationGPU(self.wf.future, self.wf.current, self.wf.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
        elif self.pmt.approximation == "VTI":
            self.wf.current[self.wf.isz,self.wf.isx] += self.wf.source[k]
            updateWaveEquationVTIGPU(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp)
        elif self.pmt.approximation == "TTI":
            self.wf.current[self.wf.isz,self.wf.isx] += self.wf.source[k]
            updateWaveEquationTTIGPU(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp, self.wf.theta_exp)
     
    def reconstructed_step(self):
            if self.pmt.approximation == "acoustic":
                self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.wf.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            elif self.pmt.approximation == "VTI":
                self.wf.future = updateWaveEquationVTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp)
            elif self.pmt.approximation == "TTI":
                self.wf.future = updateWaveEquationTTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp, self.wf.theta_exp)

    def reconstructed_stepGPU(self):
            if self.pmt.approximation == "acoustic":
                updateWaveEquationGPU(self.wf.future, self.wf.current, self.wf.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            elif self.pmt.approximation == "VTI":
                updateWaveEquationVTIGPU(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp)
            elif self.pmt.approximation == "TTI":
                updateWaveEquationTTIGPU(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp, self.wf.theta_exp)

    def backward_step(self,k):
        if self.pmt.approximation == "acoustic" and self.pmt.ABC == "cerjan":
            self.currentbck[self.pmt.rz, self.pmt.rx] += (self.muted_seismogram[k, :] / (self.pmt.dx*self.pmt.dz))
            self.futurebck = updateWaveEquation(self.futurebck, self.currentbck, self.wf.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            # Apply absorbing boundary condition
            self.futurebck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.futurebck, self.wf.A)
            self.currentbck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.currentbck, self.wf.A)
        elif self.pmt.approximation == "acoustic" and self.pmt.ABC == "CPML":
            self.currentbck[self.pmt.rz, self.pmt.rx] += (self.muted_seismogram[k, :] / (self.pmt.dx*self.pmt.dz))
            self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.currentbck, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.wf.f_pico, self.wf.d0, self.pmt.dt, self.wf.vp_exp)
            self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.currentbck, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.wf.f_pico, self.wf.d0, self.pmt.dt, self.wf.vp_exp)
            self.futurebck = updateWaveEquationCPML(self.futurebck, self.currentbck, self.wf.vp_exp, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)                         
        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "cerjan":
            self.currentbck[self.pmt.rz, self.pmt.rx] += (self.muted_seismogram[k, :] / (self.pmt.dx*self.pmt.dz))
            self.futurebck = updateWaveEquationVTI(self.futurebck, self.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp)
            # Apply absorbing boundary condition
            self.futurebck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.futurebck, self.wf.A)
            self.currentbck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.currentbck, self.wf.A)
        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "CPML":
            self.currentbck[self.pmt.rz, self.pmt.rx] += (self.muted_seismogram[k, :] / (self.pmt.dx*self.pmt.dz))
            self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.currentbck, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.wf.f_pico, self.wf.d0, self.pmt.dt, self.wf.vp_exp)
            self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.currentbck, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.wf.f_pico, self.wf.d0, self.pmt.dt, self.wf.vp_exp)
            self.futurebck = updateWaveEquationVTICPML(self.futurebck, self.currentbck, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp,self.pmt.nx_abc, self.pmt.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)
        elif self.pmt.approximation == "TTI" and self.pmt.ABC == "cerjan":
            self.currentbck[self.pmt.rz, self.pmt.rx] += (self.muted_seismogram[k, :] / (self.pmt.dx*self.pmt.dz))
            self.futurebck = updateWaveEquationTTI(self.futurebck, self.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp, self.wf.theta_exp)
            # Apply absorbing boundary condition
            self.futurebck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.futurebck, self.wf.A)
            self.currentbck = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.currentbck, self.wf.A)

    def backward_stepGPU(self,k):
        if self.pmt.approximation == "acoustic" and self.pmt.ABC == "cerjan":
            self.currentbck[self.pmt.rz, self.pmt.rx] += (self.muted_seismogram[k, :] / (self.pmt.dx*self.pmt.dz))
            updateWaveEquationGPU(self.futurebck, self.currentbck, self.wf.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            # Apply absorbing boundary condition
            self.futurebck, self.currentbck = AbsorbingBoundaryGPU(self.futurebck,self.currentbck,self.pmt.N_abc,self.pmt.nx_abc,self.pmt.nz_abc, self.wf.A)
        elif self.pmt.approximation == "acoustic" and self.pmt.ABC == "CPML":
            self.currentbck[self.pmt.rz, self.pmt.rx] += (self.muted_seismogram[k, :] / (self.pmt.dx*self.pmt.dz))
            updatePsiGPU(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.currentbck, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.wf.f_pico, self.wf.d0, self.pmt.dt, self.wf.vp_exp)
            updateZetaGPU(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.currentbck, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.wf.f_pico, self.wf.d0, self.pmt.dt, self.wf.vp_exp)
            updateWaveEquationCPMLGPU(self.futurebck, self.currentbck, self.wf.vp_exp, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)                         
        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "cerjan":
            self.currentbck[self.pmt.rz, self.pmt.rx] += (self.muted_seismogram[k, :] / (self.pmt.dx*self.pmt.dz))
            updateWaveEquationVTIGPU(self.futurebck, self.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp)
            # Apply absorbing boundary condition
            self.futurebck, self.currentbck = AbsorbingBoundaryGPU(self.futurebck,self.currentbck,self.pmt.N_abc,self.pmt.nx_abc,self.pmt.nz_abc, self.wf.A)
        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "CPML":
            self.currentbck[self.pmt.rz, self.pmt.rx] += (self.muted_seismogram[k, :] / (self.pmt.dx*self.pmt.dz))
            updatePsiGPU(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.currentbck, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.wf.f_pico, self.wf.d0, self.pmt.dt, self.wf.vp_exp)
            updateZetaGPU(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.currentbck, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.wf.f_pico, self.wf.d0, self.pmt.dt, self.wf.vp_exp)
            updateWaveEquationVTICPMLGPU(self.futurebck, self.currentbck, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp,self.pmt.nx_abc, self.pmt.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)
        elif self.pmt.approximation == "TTI" and self.pmt.ABC == "cerjan":
            self.currentbck[self.pmt.rz, self.pmt.rx] += (self.muted_seismogram[k, :] / (self.pmt.dx*self.pmt.dz))
            updateWaveEquationTTIGPU(self.futurebck, self.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.wf.vp_exp, self.wf.epsilon_exp, self.wf.delta_exp, self.wf.theta_exp)
            # Apply absorbing boundary condition
            self.futurebck, self.currentbck = AbsorbingBoundaryGPU(self.futurebck,self.currentbck,self.pmt.N_abc,self.pmt.nx_abc,self.pmt.nz_abc, self.wf.A)

    def save_boundaries(self,k): 
        self.top[k,:,:]   = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + 4, self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]
        self.bot[k,:,:]   = self.wf.future[self.pmt.N_abc + self.pmt.nz - 4: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]
        self.left[k,:,:]  = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc+4]
        self.right[k,:,:] = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz,self.pmt.N_abc + self.pmt.nx - 4: self.pmt.N_abc + self.pmt.nx]

    def apply_boundaries(self,k):
        self.wf.future[self.pmt.N_abc: self.pmt.N_abc + 4, self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx] = self.top[k,:,:]
        self.wf.future[self.pmt.N_abc + self.pmt.nz - 4: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]  = self.bot[k,:,:]
        self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc+4] = self.left[k,:,:] 
        self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz,self.pmt.N_abc + self.pmt.nx - 4: self.pmt.N_abc + self.pmt.nx] = self.right[k,:,:]

    def build_ckpts_steps(self):
        self.ckpts_steps = []
        for t0 in range (self.stop,self.pmt.nt-1,self.pmt.step):
            t1 = min(t0 + self.pmt.step,self.pmt.nt-1)
            self.ckpts_steps.append((t0,t1))
        self.ckpt_frames = {t1 for (t0, t1) in self.ckpts_steps}
   
    def reset_field(self):
        self.wf.current.fill(0)
        self.wf.future.fill(0)
        self.currentbck.fill(0)
        self.futurebck.fill(0)
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
                self.snapshots_gpubck.fill(0)
                self.wf.snap_idx = 0
                self.snap_idxbck = 0
                self.img_gpu.fill(0)
                self.wf.img_idx = 0

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
        vmax = self.wf.vp_exp.max()
        vmin = self.wf.vp_exp.min()
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
                self.wf.vp_exp[self.pmt.N_abc+i,j] = self.get_randomvalue(self.wf.vp_exp[self.pmt.N_abc+i,self.pmt.N_abc], f1d[self.pmt.N_abc-j-1], self.pmt.dvel) 
                self.wf.vp_exp[self.pmt.N_abc+i,self.pmt.nx_abc-j-1] = self.get_randomvalue(self.wf.vp_exp[self.pmt.N_abc+i,self.pmt.nx_abc-self.pmt.N_abc], f1d[self.pmt.N_abc-j-1], self.pmt.dvel)

        for i in range(self.pmt.N_abc):
            for j in range(self.pmt.nx):
                self.wf.vp_exp[i,self.pmt.N_abc+j] = self.get_randomvalue(self.wf.vp_exp[self.pmt.N_abc,self.pmt.N_abc+j], f1d[self.pmt.N_abc-i-1], self.pmt.dvel)
                self.wf.vp_exp[self.pmt.nz_abc-i-1,self.pmt.N_abc+j] = self.get_randomvalue(self.wf.vp_exp[self.pmt.nz_abc-self.pmt.N_abc,self.pmt.N_abc+j], f1d[self.pmt.N_abc-i-1], self.pmt.dvel)

        for i in range(self.pmt.N_abc):
            for j in range(i,self.pmt.N_abc):
                self.wf.vp_exp[j,i] = self.get_randomvalue(self.wf.vp_exp[self.pmt.N_abc,self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], self.pmt.dvel)
                self.wf.vp_exp[i,j] = self.get_randomvalue(self.wf.vp_exp[self.pmt.N_abc,self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], self.pmt.dvel)

                self.wf.vp_exp[j,self.pmt.nx_abc-i-1] = self.get_randomvalue(self.wf.vp_exp[self.pmt.N_abc,self.pmt.nx_abc-self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], self.pmt.dvel)
                self.wf.vp_exp[i,self.pmt.nx_abc-j-1] = self.get_randomvalue(self.wf.vp_exp[self.pmt.N_abc,self.pmt.nx_abc-self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], self.pmt.dvel)

                self.wf.vp_exp[self.pmt.nz_abc-j-1,i] = self.get_randomvalue(self.wf.vp_exp[self.pmt.nz_abc-self.pmt.N_abc,self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], self.pmt.dvel)
                self.wf.vp_exp[self.pmt.nz_abc-i-1,j] = self.get_randomvalue(self.wf.vp_exp[self.pmt.nz_abc-self.pmt.N_abc,self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], self.pmt.dvel)

                self.wf.vp_exp[self.pmt.nz_abc-j-1,self.pmt.nx_abc-i-1] = self.get_randomvalue(self.wf.vp_exp[self.pmt.nz_abc-self.pmt.N_abc,self.pmt.nx_abc-self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], self.pmt.dvel)
                self.wf.vp_exp[self.pmt.nz_abc-i-1,self.pmt.nx_abc-j-1] = self.get_randomvalue(self.wf.vp_exp[self.pmt.nz_abc-self.pmt.N_abc,self.pmt.nx_abc-self.pmt.N_abc], f1d[self.pmt.N_abc-i-1], self.pmt.dvel)

        self.wf.vp_exp[np.where(self.wf.vp_exp > vmax + self.pmt.dvel)] = vmax + self.pmt.dvel
        self.wf.vp_exp[np.where(self.wf.vp_exp < vmin - self.pmt.dvel)] = vmin - self.pmt.dvel

        points = self.poisson_disk_sampling(L_abc, D_abc, self.pmt.ratio)

        points = np.array(points)

        x_mask = np.logical_or(points[:,0] < 0.5*N_abc, points[:,0] > self.pmt.L + N_abc + 0.5*N_abc)
        z_mask = np.logical_or(points[:,1] < 0.5*N_abc, points[:,1] > self.pmt.D + N_abc + 0.5*N_abc)

        mask = np.logical_or(x_mask, z_mask)

        x, z = np.meshgrid(np.arange(self.pmt.nx_abc)*self.pmt.dx, np.arange(self.pmt.nz_abc)*self.pmt.dz) 

        points = points[mask]

        for index in range(len(points)):

            xc = points[index, 0]  
            zc = points[index, 1] 

            r = np.random.uniform(0.1*self.pmt.ratio, self.pmt.ratio)
            A = np.random.uniform(0.5*self.pmt.dvel, self.pmt.dvel)

            factor = np.random.choice([-1,1])

            self.wf.vp_exp = self.wf.vp_exp + factor*A*np.exp(-0.5*(((x - xc) / r)**2 + ((z - zc) / r)**2))
            
        self.wf.vp_exp[np.where(self.wf.vp_exp > vmax + self.pmt.dvel)] = vmax + self.pmt.dvel
        self.wf.vp_exp[np.where(self.wf.vp_exp < vmin - self.pmt.dvel)] = vmin - self.pmt.dvel

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(self.wf.vp_exp, aspect = "auto", cmap = "jet", vmax = vmax + self.pmt.dvel, vmin = vmin - self.pmt.dvel, extent = [0, L_abc, D_abc, 0])
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
        if self.pmt.fwi == False:
            water_mask = np.abs(self.wf.vp - 1500.0) < 1e-3
            self.vp = smooth_model(self.wf.vp, self.pmt.sigma, water_mask)
        self.wf.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.wf.theta)
        

        save_field = np.zeros([self.pmt.nt,self.pmt.nz,self.pmt.nx],dtype=np.float32)
        self.stop = int(2*self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")

            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.wf.isx = self.pmt.sx[shot]
            self.wf.isz = self.pmt.sz[shot]        

            # Top muting
            seismogram = self.loadSeismogram(shot)
            if self.pmt.fwi  == True:
                self.muted_seismogram = seismogram
            else:
                self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,self.pmt.tlag, self.pmt.shift,self.pmt.dx,self.pmt.N_abc,self.pmt.window,self.pmt.v0) 
            self.migrated_partial = np.zeros_like(self.migrated_image)
            self.ilum_partial = np.zeros_like(self.ilum)
            for k in range(self.pmt.nt):
                self.wf.forward_step(k)
                self.wf.save_snapshot(shot, k)
                save_field[k,:,:] = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                self.ilum_partial += save_field[k,:,:] * save_field[k,:,:] 
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.backward_step(t)
                self.save_snapshotBCK(shot,t)
                if self.pmt.fwi  == True:
                    d2Udt2 = (save_field[t+1,:,:] - 2.0*save_field[t,:,:] + save_field[t-1,:,:]) / (self.pmt.dt*self.pmt.dt)
                    self.migrated_partial += d2Udt2 * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                else:
                    self.migrated_partial += (save_field[t,:,:] * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.save_image(shot,t)
                #swap
                self.currentbck, self.futurebck = self.futurebck, self.currentbck

            self.migrated_image += self.migrated_partial
            self.ilum += self.ilum_partial
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        
        self.migrated_image = self.migrated_image / self.ilum
        if self.pmt.fwi  == True:
            self.outputFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
            self.migrated_image.astype(np.float32).tofile(self.outputFile)
        else:
            self.outputFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
            self.migrated_image.astype(np.float32).tofile(self.outputFile)
        print(f"info: Final image saved to {self.outputFile}")
    
    # REGULAR CHECKPOINTING
    def solveBackwardWaveEquationCheckpointing(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        if self.pmt.fwi == False:
            water_mask = np.abs(self.wf.vp - 1500.0) < 1e-3
            self.vp = smooth_model(self.wf.vp, self.pmt.sigma, water_mask)
        self.wf.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.wf.theta)
        

        self.stop = int(2*self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.wf.isx = self.pmt.sx[shot]
            self.wf.isz = self.pmt.sz[shot]        

            # Top muting
            seismogram = self.loadSeismogram(shot)
            if self.pmt.fwi  == True:
                self.muted_seismogram = seismogram
            else:
                self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,self.pmt.tlag, self.pmt.shift,self.pmt.dx,self.pmt.N_abc,self.pmt.window,self.pmt.v0) 
            self.migrated_partial = np.zeros_like(self.migrated_image)
            self.ilum_partial = np.zeros_like(self.ilum)
            self.build_ckpts_steps()
            for k in range(self.pmt.nt):
                self.wf.forward_step(k)
                self.wf.save_snapshot(shot, k)
                self.save_checkpoint(shot, k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for (t0,t1) in reversed(self.ckpts_steps):
                self.load_checkpoint(shot,t1)
                for t in range(t1, t0, -1):
                    self.ilum_partial += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] 
                    if self.pmt.fwi == True:
                        u_next = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()
                        u_curr = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()
                    self.reconstructed_step()
                    if self.pmt.fwi == True:
                        u_prev = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()
                    self.backward_step(t)
                    self.save_snapshotBCK(shot,t)
                    if self.pmt.fwi  == True:
                        d2Udt2 = (u_next - 2.0*u_curr + u_prev) / (self.pmt.dt*self.pmt.dt) 
                        self.migrated_partial += d2Udt2 * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                    else:
                        self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                    self.save_image(shot,t)
                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current
                    self.currentbck, self.futurebck = self.futurebck, self.currentbck

            self.migrated_image += self.migrated_partial
            self.ilum += self.ilum_partial 
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        
        self.migrated_image = self.migrated_image / self.ilum
        if self.pmt.fwi  == True:
            self.outputFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
            self.migrated_image.astype(np.float32).tofile(self.outputFile)
        else:
            self.outputFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
            self.migrated_image.astype(np.float32).tofile(self.outputFile)
        print(f"info: Final image saved to {self.outputFile}")
    
    #Saving Boundaries
    def solveBackwardWaveEquationSavingBoundaries(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        if self.pmt.fwi == False:
            water_mask = np.abs(self.wf.vp - 1500.0) < 1e-3
            self.vp = smooth_model(self.wf.vp, self.pmt.sigma, water_mask)
        self.wf.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.wf.theta)


        self.stop = int(2*self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.wf.isx = self.pmt.sx[shot]
            self.wf.isz = self.pmt.sz[shot]        

            # Top muting
            seismogram = self.loadSeismogram(shot)
            if self.pmt.fwi  == True:
                self.muted_seismogram = seismogram
            else:
                self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,self.pmt.tlag, self.pmt.shift,self.pmt.dx,self.pmt.N_abc,self.pmt.window,self.pmt.v0) 
            self.migrated_partial = np.zeros_like(self.migrated_image)
            self.ilum_partial = np.zeros_like(self.ilum)
            for k in range(self.pmt.nt):
                self.wf.forward_step(k)
                self.save_boundaries(k)
                self.wf.save_snapshot(shot, k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current 
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.ilum_partial += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]  
                if self.pmt.fwi == True:
                    u_next = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()
                    u_curr = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()   
                self.reconstructed_step()
                if self.pmt.fwi == True:
                    u_prev = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()
                self.apply_boundaries(t)           
                self.backward_step(t)
                self.save_snapshotBCK(shot,t)
                if self.pmt.fwi  == True:
                    d2Udt2 = (u_next - 2.0*u_curr + u_prev) / (self.pmt.dt*self.pmt.dt) 
                    self.migrated_partial += d2Udt2 * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                else:
                    self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.save_image(shot,t)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.currentbck, self.futurebck = self.futurebck, self.currentbck

            self.migrated_image += self.migrated_partial
            self.ilum += self.ilum_partial
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        
        self.migrated_image = self.migrated_image / self.ilum      
        if self.pmt.fwi  == True:
            self.outputFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
            self.migrated_image.astype(np.float32).tofile(self.outputFile)
        else:
            self.outputFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
            self.migrated_image.astype(np.float32).tofile(self.outputFile)
        print(f"info: Final image saved to {self.outputFile}")

    #Random Boundary Condintion
    def solveBackwardWaveEquationRBC(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        if self.pmt.fwi == False:
            water_mask = np.abs(self.wf.vp - 1500.0) < 1e-3
            self.vp = smooth_model(self.wf.vp, self.pmt.sigma, water_mask)
        vp_exp_base = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.wf.theta)


        self.stop = int(2*self.pmt.tlag/self.pmt.dt)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()
            self.wf.vp_exp = vp_exp_base.copy()
            self.create_random_boundary()

            # convert acquisition geometry coordinates to grid points
            self.wf.isx = self.pmt.sx[shot]
            self.wf.isz = self.pmt.sz[shot]        

            # Top muting
            seismogram = self.loadSeismogram(shot)
            if self.pmt.fwi  == True:
                self.muted_seismogram = seismogram
            else:
                self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,self.pmt.tlag, self.pmt.shift,self.pmt.dx,self.pmt.N_abc,self.pmt.window,self.pmt.v0) 
            self.migrated_partial = np.zeros_like(self.migrated_image)
            self.ilum_partial = np.zeros_like(self.ilum)
            for k in range(self.pmt.nt):
                self.forward_step_RBC(k)
                self.wf.save_snapshot(shot, k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            self.wf.current, self.wf.future = self.wf.future, self.wf.current    
            for t in range(self.pmt.nt - 1, self.stop, -1): 
                self.ilum_partial += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] 
                if self.pmt.fwi == True:
                    u_next = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()
                    u_curr = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()   
                self.reconstructed_step()
                if self.pmt.fwi == True:
                    u_prev = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()
                self.backward_step(t) 
                self.save_snapshotBCK(shot,t)
                if self.pmt.fwi  == True:
                    d2Udt2 = (u_next - 2.0*u_curr + u_prev) / (self.pmt.dt*self.pmt.dt) 
                    self.migrated_partial += d2Udt2 * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                else:
                    self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.save_image(shot,t)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.currentbck, self.futurebck = self.futurebck, self.currentbck

            self.migrated_image += self.migrated_partial
            self.ilum += self.ilum_partial
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        
        self.migrated_image = self.migrated_image / self.ilum
        if self.pmt.fwi  == True:
            self.outputFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
            self.migrated_image.astype(np.float32).tofile(self.outputFile)
        else:
            self.outputFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
            self.migrated_image.astype(np.float32).tofile(self.outputFile)
        print(f"info: Final image saved to {self.outputFile}")

## GPU Migration Types
    #On the fly
    def solveBackwardWaveEquationOntheFlyGPU(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        if self.pmt.fwi == False:
            water_mask = np.abs(self.wf.vp - 1500.0) < 1e-3
            self.vp = smooth_model(self.wf.vp, self.pmt.sigma, water_mask)
        self.wf.vp_exp = self.wf.ExpandModel(self.vp)
        self.wf.vp_exp = cp.asarray(self.wf.vp_exp, dtype=cp.float32)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
            self.wf.A = cp.asarray(self.wf.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.wf.delta)
            self.wf.epsilon_exp  = cp.asarray(self.wf.epsilon_exp, dtype=cp.float32)
            self.wf.delta_exp  = cp.asarray(self.wf.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.wf.theta)
                self.wf.theta_exp  = cp.asarray(self.wf.theta_exp, dtype=cp.float32)
        
        self.pmt.rx = cp.asarray(self.pmt.rx)
        self.pmt.rz = cp.asarray(self.pmt.rz)
        save_field = cp.zeros([self.pmt.nt,self.pmt.nz,self.pmt.nx],dtype=cp.float32)
        self.stop = int(2*self.pmt.tlag/self.pmt.dt)
        self.ilum = cp.asarray(self.ilum)
        self.migrated_image = cp.asarray(self.migrated_image)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            self.wf.isx = self.pmt.sx[shot]
            self.wf.isz = self.pmt.sz[shot]        

            # Top muting
            seismogram = self.loadSeismogram(shot)
            if self.pmt.fwi == True:
                self.muted_seismogram = seismogram
            else:
                self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,self.pmt.tlag, self.pmt.shift,self.pmt.dx,self.pmt.N_abc,self.pmt.window,self.pmt.v0) 
            self.muted_seismogram = cp.asarray(self.muted_seismogram,dtype=cp.float32)
            self.migrated_partial = cp.zeros_like(self.migrated_image)
            self.ilum_partial = cp.zeros_like(self.migrated_image)
            for k in range(self.pmt.nt):
                self.wf.forward_stepGPU(k)
                self.wf.store_snapshotGPU(k) 
                save_field[k,:,:] = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                self.ilum_partial += save_field[k,:,:] * save_field[k,:,:] 
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.wf.forward_stepGPU(t)
                self.backward_stepGPU(t)
                self.store_snapshotBCKGPU(t)
                if self.pmt.fwi  == True:
                    d2Udt2 = (save_field[t+1,:,:] - 2.0*save_field[t,:,:] + save_field[t-1,:,:]) / (self.pmt.dt*self.pmt.dt)
                    self.migrated_partial += d2Udt2 * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                else:
                    self.migrated_partial += (save_field[t,:,:] * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.store_imageGPU(t)
                #swap
                self.currentbck, self.futurebck = self.futurebck, self.currentbck
            self.ilum += self.ilum_partial
            self.migrated_image += self.migrated_partial 
            self.wf.save_snapshotGPU(shot)
            self.save_snapshotBCKGPU(shot)
            self.save_imageGPU(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        self.migrated_image = self.migrated_image / self.ilum
        migrated_imagecpu = cp.asnumpy(self.migrated_image)
        migrated_imagecpu[water_mask] = 0
        if self.pmt.fwi  == True:
            self.outputFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        else:
            self.outputFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        migrated_imagecpu.astype(np.float32).tofile(self.outputFile)
        print(f"info: Final image saved to {self.outputFile}")
        
    # REGULAR CHECKPOINTING
    def solveBackwardWaveEquationCheckpointingGPU(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        if self.pmt.fwi == False:
            water_mask = np.abs(self.wf.vp - 1500.0) < 1e-3
            self.vp = smooth_model(self.wf.vp, self.pmt.sigma, water_mask)
        self.wf.vp_exp = self.wf.ExpandModel(self.vp)
        self.wf.vp_exp = cp.asarray(self.wf.vp_exp, dtype=cp.float32)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
            self.wf.A = cp.asarray(self.wf.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.wf.delta)
            self.wf.epsilon_exp  = cp.asarray(self.wf.epsilon_exp, dtype=cp.float32)
            self.wf.delta_exp  = cp.asarray(self.wf.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.wf.theta)
                self.wf.theta_exp  = cp.asarray(self.wf.theta_exp, dtype=cp.float32)
        
        self.pmt.rx = cp.asarray(self.pmt.rx)
        self.pmt.rz = cp.asarray(self.pmt.rz)
        self.stop = int(2*self.pmt.tlag/self.pmt.dt)
        self.ilum = cp.asarray(self.ilum)
        self.migrated_image = cp.asarray(self.migrated_image)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.wf.isx = self.pmt.sx[shot]
            self.wf.isz = self.pmt.sz[shot]        

            # Top muting
            seismogram = self.loadSeismogram(shot)
            if self.pmt.fwi  == True:
                self.muted_seismogram = seismogram
            else:
                self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,self.pmt.tlag, self.pmt.shift,self.pmt.dx,self.pmt.N_abc,self.pmt.window,self.pmt.v0) 
            self.muted_seismogram = cp.asarray(self.muted_seismogram,dtype=cp.float32)
            self.migrated_partial = cp.zeros_like(self.migrated_image)
            self.ilum_partial = cp.zeros_like(self.migrated_image)
            self.build_ckpts_steps()
            for k in range(self.pmt.nt):
                self.wf.forward_stepGPU(k)
                self.wf.store_snapshotGPU(k)
                self.save_checkpointGPU(shot, k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            self.flush_checkpointGPU()
            for (t0,t1) in reversed(self.ckpts_steps):
                self.load_checkpoint(shot,t1)
                for t in range(t1, t0, -1):
                    self.ilum_partial += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] 
                    if self.pmt.fwi == True:
                        u_next = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()
                        u_curr = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()                  
                    self.reconstructed_stepGPU()
                    if self.pmt.fwi == True:
                        u_prev = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()                
                    self.backward_stepGPU(t)
                    self.store_snapshotBCKGPU(t)
                    if self.pmt.fwi  == True:
                        d2Udt2 = (u_next - 2.0*u_curr + u_prev) / (self.pmt.dt*self.pmt.dt) 
                        self.migrated_partial += d2Udt2 * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                    else:
                        self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                    self.store_imageGPU(t)
                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current
                    self.currentbck, self.futurebck = self.futurebck, self.currentbck

            self.migrated_image += self.migrated_partial
            self.ilum += self.ilum_partial 
            self.wf.save_snapshotGPU(shot)
            self.save_snapshotBCKGPU(shot)
            self.save_imageGPU(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        
        self.migrated_image = self.migrated_image / self.ilum
        migrated_imagecpu = cp.asnumpy(self.migrated_image)
        if self.pmt.fwi  == True:
            self.outputFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        else:
            self.outputFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        migrated_imagecpu.astype(np.float32).tofile(self.outputFile)
        print(f"info: Final image saved to {self.outputFile}")
    
    #Saving Boundaries
    def solveBackwardWaveEquationSavingBoundariesGPU(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        if self.pmt.fwi == False:
            water_mask = np.abs(self.wf.vp - 1500.0) < 1e-3
            # self.vp = smooth_model(self.wf.vp, self.pmt.sigma, water_mask)
        self.wf.vp_exp = self.wf.ExpandModel(self.wf.vp)
        self.wf.vp_exp = cp.asarray(self.wf.vp_exp, dtype=cp.float32)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
            self.wf.A = cp.asarray(self.wf.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.wf.delta)
            self.wf.epsilon_exp  = cp.asarray(self.wf.epsilon_exp, dtype=cp.float32)
            self.wf.delta_exp  = cp.asarray(self.wf.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.wf.theta)
                self.wf.theta_exp  = cp.asarray(self.wf.theta_exp, dtype=cp.float32)


        self.pmt.rx = cp.asarray(self.pmt.rx)
        self.pmt.rz = cp.asarray(self.pmt.rz)
        self.stop = int(2*self.pmt.tlag/self.pmt.dt)
        self.ilum = cp.asarray(self.ilum)
        self.migrated_image = cp.asarray(self.migrated_image)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.wf.isx = self.pmt.sx[shot]
            self.wf.isz = self.pmt.sz[shot]        

            # Top muting
            seismogram = self.loadSeismogram(shot)
            if self.pmt.fwi  == True:
                self.muted_seismogram = seismogram
            else:
                self.muted_seismogram = seismogram#Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,self.pmt.tlag, self.pmt.shift,self.pmt.dx,self.pmt.N_abc,self.pmt.window,self.pmt.v0) 
            self.muted_seismogram = cp.asarray(self.muted_seismogram,dtype=cp.float32)
            self.migrated_partial = cp.zeros_like(self.migrated_image)
            self.ilum_partial = cp.zeros_like(self.migrated_image)
            for k in range(self.pmt.nt):
                self.wf.forward_stepGPU(k)
                self.save_boundaries(k)
                self.wf.store_snapshotGPU(k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current 
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.ilum_partial += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] 
                if self.pmt.fwi == True:
                    u_next = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()
                    u_curr = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()                  
                self.reconstructed_stepGPU()
                if self.pmt.fwi == True:
                    u_prev = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()                
                self.apply_boundaries(t)           
                self.backward_stepGPU(t)
                self.store_snapshotBCKGPU(t)
                if self.pmt.fwi  == True:
                    d2Udt2 = (u_next - 2.0*u_curr + u_prev) / (self.pmt.dt*self.pmt.dt) 
                    self.migrated_partial += d2Udt2 * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                else:
                    self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.store_imageGPU(t)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.currentbck, self.futurebck = self.futurebck, self.currentbck

            self.migrated_image += self.migrated_partial
            self.ilum += self.ilum_partial
            self.wf.save_snapshotGPU(shot)
            self.save_snapshotBCKGPU(shot)
            self.save_imageGPU(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds") 
        self.migrated_image = self.migrated_image / self.ilum
        migrated_imagecpu = cp.asnumpy(self.migrated_image)      
        if self.pmt.fwi  == True:
            self.outputFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        else:
            self.outputFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        migrated_imagecpu.astype(np.float32).tofile(self.outputFile)
        print(f"info: Final image saved to {self.outputFile}")

    #Random Boundary Condintion
    def solveBackwardWaveEquationRBCGPU(self):
        start_time = time.time()
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        if self.pmt.fwi == False:
            water_mask = np.abs(self.wf.vp - 1500.0) < 1e-3
            self.vp = smooth_model(self.wf.vp, self.pmt.sigma, water_mask)
        vp_exp_base = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
            self.wf.A = cp.asarray(self.wf.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.wf.delta)
            self.wf.epsilon_exp  = cp.asarray(self.wf.epsilon_exp, dtype=cp.float32)
            self.wf.delta_exp  = cp.asarray(self.wf.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.wf.theta)
                self.wf.theta_exp  = cp.asarray(self.wf.theta_exp, dtype=cp.float32)


        self.pmt.rx = cp.asarray(self.pmt.rx)
        self.pmt.rz = cp.asarray(self.pmt.rz)
        self.stop = int(2*self.pmt.tlag/self.pmt.dt)
        self.ilum = cp.asarray(self.ilum)
        self.migrated_image = cp.asarray(self.migrated_image)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()
            self.wf.vp_exp = vp_exp_base.copy()
            self.create_random_boundary()
            self.wf.vp_exp = cp.asarray(self.wf.vp_exp, dtype=cp.float32)

            # convert acquisition geometry coordinates to grid points
            self.wf.isx = self.pmt.sx[shot]
            self.wf.isz = self.pmt.sz[shot]        

            # Top muting
            seismogram = self.loadSeismogram(shot)
            if self.pmt.fwi  == True:
                self.muted_seismogram = seismogram
            else:
                self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,self.pmt.tlag, self.pmt.shift,self.pmt.dx,self.pmt.N_abc,self.pmt.window,self.pmt.v0) 
            self.muted_seismogram = cp.asarray(self.muted_seismogram,dtype=cp.float32)
            self.migrated_partial = cp.zeros_like(self.migrated_image)
            self.ilum_partial = cp.zeros_like(self.migrated_image)
            for k in range(self.pmt.nt):
                self.forward_stepGPU_RBC(k)
                self.wf.store_snapshotGPU(k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            self.wf.current, self.wf.future = self.wf.future, self.wf.current    
            for t in range(self.pmt.nt - 1, self.stop, -1):
                self.ilum_partial += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                if self.pmt.fwi == True:
                    u_next = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()
                    u_curr = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()                   
                self.reconstructed_stepGPU()
                if self.pmt.fwi == True:
                    u_prev = self.wf.future[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc].copy()                
                self.backward_stepGPU(t)
                self.store_snapshotBCKGPU(t) 
                if self.pmt.fwi  == True:
                    d2Udt2 = (u_next - 2.0*u_curr + u_prev) / (self.pmt.dt*self.pmt.dt) 
                    self.migrated_partial += d2Udt2 * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                else:
                    self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.store_imageGPU(t)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.currentbck, self.futurebck = self.futurebck, self.currentbck

            self.migrated_image += self.migrated_partial
            self.ilum += self.ilum_partial
            self.wf.save_snapshotGPU(shot)
            self.save_snapshotBCKGPU(shot)
            self.save_imageGPU(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        
        self.migrated_image = self.migrated_image / self.ilum
        migrated_imagecpu = cp.asnumpy(self.migrated_image)
        if self.pmt.fwi  == True:
            self.outputFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        else:
            self.outputFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        migrated_imagecpu.astype(np.float32).tofile(self.outputFile)
        print(f"info: Final image saved to {self.outputFile}")

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