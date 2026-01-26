import numpy as np
import pandas as pd
import json
from Modeling2D import wavefield
import os
import matplotlib.pyplot as plt

from utils import ricker
from utils import updateWaveEquation
from utils import updateWaveEquationCPML
from utils import updateWaveEquationVTI
from utils import updateWaveEquationVTICPML
from utils import updateWaveEquationTTI
from utils import updateWaveEquationTTICPML
from utils import AbsorbingBoundary
from utils import updatePsi
from utils import updateZeta
from utils import updatePsiTTI
from utils import updateZetaTTI

class migration: 

    def __init__(self, parameters_path, wavefield):
        self.parameters_path = parameters_path
        self.readParameters()
        self.readAcquisitionGeometry()
        self.wf = wavefield

    def readParameters(self):
        with open(self.parameters_path) as f:
            self.parameters = json.load(f)

        # Approximation type
        self.approximation = self.parameters["approximation"]
        self.migration = self.parameters["migration"]
        self.ABC = self.parameters["ABC"]
        
        # Discretization self.parameters
        self.dx   = self.parameters["dx"]
        self.dz   = self.parameters["dz"]
        self.dt   = self.parameters["dt"]
        
        # Model size
        self.L    = self.parameters["L"]
        self.D    = self.parameters["D"]
        self.T    = self.parameters["T"]

        # Number of point for absorbing boundary condition
        self.N_abc = self.parameters["N_abc"]

        # Number of points in each direction
        self.nx = int(self.L/self.dx)+1
        self.nz = int(self.D/self.dz)+1
        self.nt = int(self.T/self.dt)+1

        self.nx_abc = self.nx + 2*self.N_abc
        self.nz_abc = self.nz + 2*self.N_abc

        # Define arrays for space and time
        self.x = np.linspace(0, self.L, self.nx)
        self.z = np.linspace(0, self.D, self.nz)
        self.t = np.linspace(0, self.T, self.nt)

        # Max frequency
        self.fcut = self.parameters["fcut"]

        # Output folders
        self.seismogramFolder = self.parameters["seismogramFolder"]
        self.migratedimageFolder = self.parameters["migratedimageFolder"]
        self.snapshotFolder = self.parameters["snapshotFolder"]
        self.modelFolder = self.parameters["modelFolder"]
        self.checkpointFolder = self.parameters["checkpointFolder"]

        # Source and receiver files
        self.rec_file = self.parameters["rec_file"]
        self.src_file = self.parameters["src_file"]

        # Velocity model file
        self.vpFile = self.parameters["vpFile"]
        self.vsFile = self.parameters["vsFile"]
        self.thetaFile = self.parameters["thetaFile"]

        # Snapshot flag
        self.snap = self.parameters["snap"]
        self.step = self.parameters["step"]
        self.last_save = self.parameters["last_save"]

        # Anisotropy parameters files
        self.epsilonFile = self.parameters["epsilonFile"]  
        self.deltaFile   = self.parameters["deltaFile"]  

        #Anisotropy parameters for Layered model
        self.vpLayer1 = self.parameters["vpLayer1"]
        self.vpLayer2 = self.parameters["vpLayer2"]
        self.thetaLayer1 = self.parameters["thetaLayer1"]
        self.thetaLayer2 = self.parameters["thetaLayer2"]
        self.epsilonLayer1 = self.parameters["epsilonLayer1"]
        self.epsilonLayer2 = self.parameters["epsilonLayer2"]
        self.deltaLayer1   = self.parameters["deltaLayer1"]
        self.deltaLayer2  = self.parameters["deltaLayer2"]

    def readAcquisitionGeometry(self):        
        # Read receiver and source coordinates from CSV files
        receiverTable = pd.read_csv(self.rec_file)
        print(f"info: Imported: {self.rec_file}")     
        sourceTable = pd.read_csv(self.src_file)
        print(f"info: Imported: {self.src_file}")

        # Read receiver and source coordinates
        self.rec_x = receiverTable['coordx'].to_numpy()
        self.rec_z = receiverTable['coordz'].to_numpy()
        self.shot_x = sourceTable['coordx'].to_numpy()
        self.shot_z = sourceTable['coordz'].to_numpy()

        self.Nrec = len(self.rec_x)
        self.Nshot = len(self.shot_x) 
    
    def loadSeismogram(self, shot):
        seismogramFile = f"{self.seismogramFolder}{self.approximation}{self.ABC}_seismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
        seismogram = np.fromfile(seismogramFile, dtype=np.float32).reshape(self.nt,self.Nrec) 
        return seismogram

    def Mute(self, seismogram, shot): 
        muted = seismogram.copy() 
        v0 = self.wf.vp[0, :]
        rec_idx = (self.rec_x / self.dx).astype(int)
        v0_rec = v0[rec_idx]
        distz = self.rec_z - self.shot_z[shot]   
        distx = self.rec_x - self.shot_x[shot]   
        dist = np.sqrt(distx**2 + distz**2)
        t_lag = 2 * np.sqrt(np.pi) / self.fcut
        traveltimes = dist / v0_rec + 2.5 * t_lag 
        
        for r in range(self.Nrec): 
            mute_samples = int(traveltimes[r] / self.dt)
            muted[:mute_samples, r] = 0 
                
        return muted

    def laplacian_filter(self, f):
        dim1,dim2 = np.shape(f)
        g = np.zeros([dim1,dim2])
        lap_z = 0
        lap_x = 0
        for ix in range(1, dim2 - 1):
            for iz in range(1, dim1 - 1):
                lap_z = f[iz+1, ix] + f[iz-1, ix] - 2 * f[iz, ix]
                lap_x = f[iz, ix+1] + f[iz, ix-1] - 2 * f[iz, ix]
                g[iz, ix] = lap_z/(self.dz*self.dz) + lap_x/(self.dx*self.dx)

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
        kernel_size = np.ceil(2 * sigma + 1)
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
        checkpointFile = (f"{self.checkpointFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin")
        with open(checkpointFile, "rb") as file:
            count = self.nx_abc * self.nz_abc
            self.wf.current = np.fromfile(file, np.float32, count).reshape(self.nz_abc, self.nx_abc)
            self.wf.future  = np.fromfile(file, np.float32, count).reshape(self.nz_abc, self.nx_abc)
    
    def save_checkpoint(self, shot, k):
        if self.migration != "checkpoint":
            return
        if k > self.last_save:
            return
        if k % self.step != 0:
            return

        if self.approximation == "TTI" and self.ABC == "CPML":
            raise ValueError("Checkpoint saving for TTI CPML not implemented yet.")
        
        checkpointFile = (f"{self.checkpointFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin")

        save = [self.wf.current, self.wf.future]

        with open(checkpointFile, "wb") as file:
            for field in save:
                field.astype(np.float32).tofile(file)

        print(f"info: Checkpoint saved to {checkpointFile}")
    
    def forward_step(self,k):
        if self.migration in ["onthefly","checkpoint","RBC"]:
            if self.approximation == "acoustic":
                self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
            if self.approximation == "VTI":
                self.wf.future= updateWaveEquationVTI(self.wf.future, self.wf.current, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
            elif self.approximation == "TTI":
                self.wf.future= updateWaveEquationTTI(self.wf.future, self.wf.current, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
        
        if self.migration == "SB":
            if self.approximation == "acoustic" and self.ABC == "cerjan":
                self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.current, self.A)

            elif self.approximation == "acoustic" and self.ABC == "CPML":
                self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.nx_abc, self.nz_abc, self.wf.current, self.dx, self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.nx_abc, self.nz_abc, self.wf.current, self.dx,self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.wf.future = updateWaveEquationCPML(self.wf.future, self.wf.current, self.vp_exp, self.nx_abc, self.nz_abc, self.dz, self.dx, self.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.N_abc)             
                self.wf.future[self.sz,self.sx] += self.wf.source[k]

            elif self.approximation == "VTI" and self.ABC == "cerjan":
                self.wf.future= updateWaveEquationVTI(self.wf.future, self.wf.current, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.current, self.A)

            elif self.approximation == "VTI" and self.ABC == "CPML":
                self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.nx_abc, self.nz_abc, self.wf.current, self.dx, self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.nx_abc, self.nz_abc, self.wf.current, self.dx,self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.wf.future = updateWaveEquationVTICPML(self.wf.future, self.wf.current, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.nx_abc, self.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.N_abc)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]

            elif self.approximation == "TTI" and self.ABC == "cerjan":
                self.wf.future= updateWaveEquationTTI(self.wf.future, self.wf.current, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.current, self.A)

    def reconstructed_step(self,t):
        if self.approximation == "acoustic":
            self.wf.future[self.sz, self.sx] -= self.wf.source[t]
            self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)
        elif self.approximation == "VTI":
            self.wf.future[self.sz, self.sx] -= self.wf.source[t]
            self.wf.future= updateWaveEquationVTI(self.wf.future, self.wf.current, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
        elif self.approximation == "TTI":
            self.wf.future[self.sz, self.sx] -= self.wf.source[t]
            self.wf.future= updateWaveEquationTTI(self.wf.future, self.wf.current, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
        
    def save_boundaries(self,k): 
        if self.migration == "SB":
            self.wf.top[k,:,:]   = self.wf.future[self.N_abc: self.N_abc + 4, self.N_abc: self.N_abc + self.nx]
            self.wf.bot[k,:,:]   = self.wf.future[self.N_abc + self.nz - 4: self.N_abc + self.nz , self.N_abc: self.N_abc + self.nx]
            self.wf.left[k,:,:]  = self.wf.future[self.N_abc: self.N_abc + self.nz , self.N_abc: self.N_abc+4]
            self.wf.right[k,:,:] = self.wf.future[self.N_abc: self.N_abc + self.nz,self.N_abc + self.nx - 4: self.N_abc + self.nx]

    def apply_boundaries(self,k):
        if self.migration == "SB":
            self.wf.future[self.N_abc: self.N_abc + 4, self.N_abc: self.N_abc + self.nx] = self.wf.top[k,:,:]
            self.wf.future[self.N_abc + self.nz - 4: self.N_abc + self.nz , self.N_abc: self.N_abc + self.nx]  = self.wf.bot[k,:,:]
            self.wf.future[self.N_abc: self.N_abc + self.nz , self.N_abc: self.N_abc+4] = self.wf.left[k,:,:] 
            self.wf.future[self.N_abc: self.N_abc + self.nz,self.N_abc + self.nx - 4: self.N_abc + self.nx] = self.wf.right[k,:,:]
       
    def backward_step(self,k):
        if self.approximation == "acoustic":
            self.wf.futurebck = updateWaveEquation(self.wf.futurebck, self.wf.currentbck, self.vp_exp,self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]
            
        elif self.approximation == "VTI":
            self.wf.futurebck= updateWaveEquationVTI(self.wf.futurebck, self.wf.currentbck, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]
            
        elif self.approximation == "TTI":
            self.wf.futurebck= updateWaveEquationTTI(self.wf.futurebck, self.wf.currentbck, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]

    def build_ckpts_steps(self):
        self.ckpts_steps = []
        for t0 in range (0,self.nt-1,self.step):
            t1 = min(t0 + self.step,self.nt)
            self.ckpts_steps.append((t0,t1))
    
    def reset_field(self):
        self.wf.current.fill(0)
        self.wf.future.fill(0)
        self.wf.currentbck.fill(0)
        self.wf.futurebck.fill(0)
        if self.ABC == "CPML":
            self.wf.PsixFR.fill(0)
            self.wf.PsixFL.fill(0)
            self.wf.PsizFU.fill(0)  
            self.wf.PsizFD.fill(0) 
            self.wf.ZetaxFR.fill(0)
            self.wf.ZetaxFL.fill(0)
            self.wf.ZetazFU.fill(0)
            self.wf.ZetazFD.fill(0)

    def create_random_boundary(self):
        cmax = 0.5
        v_limite = (cmax*self.dx)/self.dt
        A = self.vp_exp.max()*2 
        print(A)
        for i in range(self.nx_abc):
            for j in range(self.nz_abc):
                if i < self.N_abc:
                    dx = self.N_abc - i
                elif i >= self.nx_abc - self.N_abc:
                    dx = i - (self.nx_abc - self.N_abc)
                else:
                    dx = 0

                if j < self.N_abc:
                    dz = self.N_abc - j
                elif j >= self.nz_abc - self.N_abc:
                    dz = j - (self.nz_abc - self.N_abc)
                else:
                    dz = 0

                d = np.sqrt(dx*dx + dz*dz)
                
                d = d / self.N_abc
                
                found = False
                while found == False:
                    r = A * 2.*np.random.rand()-1.
                    vtest = self.vp_exp[j,i] + r * d
                    if vtest <= v_limite:
                        self.vp_exp[j,i] = vtest
                        found = True
        plt.figure()
        plt.imshow(self.vp_exp)
        plt.show()

    #On the fly
    def solveBackwardWaveEquationOntheFly(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)
        
        self.rx = np.int32(self.rec_x/self.dx) + self.N_abc
        self.rz = np.int32(self.rec_z/self.dz) + self.N_abc
        save_field = np.zeros([self.nt,self.nz,self.nx],dtype=np.float32)
        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")

            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            self.sz = int(self.shot_z[shot]/self.dz) + self.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = self.Mute(seismogram, shot)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)

            for k in range(self.nt):
                self.forward_step(k)
                save_field[k,:,:] = self.wf.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc]
                # if k%500 == 0:
                #     plt.figure()
                #     plt.imshow(self.wf.current)
                #     plt.show()
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for t in range(self.nt - 1, -1, -1):
                self.backward_step(t)
                # if k%500 == 0:
                #     plt.figure()
                #     plt.imshow(self.wf.currentbck)
                #     plt.show()
                self.migrated_partial += (save_field[t,:,:] * self.wf.currentbck[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc])
                #swap
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")
     
        # Apply laplacian_filter filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_{self.approximation}_Nx{self.nx}_Nz{self.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    # REGULAR CHECKPOINTING
    def solveBackwardWaveEquationCheckpointing(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)
        
        self.rx = np.int32(self.rec_x/self.dx) + self.N_abc
        self.rz = np.int32(self.rec_z/self.dz) + self.N_abc
        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            self.sz = int(self.shot_z[shot]/self.dz) + self.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = self.Mute(seismogram, shot)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.build_ckpts_steps()
            for k in range(self.nt):
                self.forward_step(k)
                self.save_checkpoint(shot, k)
                if k == 1200:
                    plt.figure()
                    plt.imshow(self.wf.current)
                    plt.show()
                    snapshot = self.wf.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc]
                    snapshotFile = (f"{self.snapshotFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}FORWARD.bin")
                    snapshot.tofile(snapshotFile)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for (t0,t1) in reversed(self.ckpts_steps):
                self.load_checkpoint(shot,t1)
                for t in range(t1, t0, -1):
                    self.reconstructed_step(t)
                    self.backward_step(t)
                    if t == 1200:
                        plt.figure()
                        plt.imshow(self.wf.current)
                        plt.show()
                        snapshot = self.wf.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc]
                        snapshotFile = (f"{self.snapshotFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}CHECKPOINT.bin")
                        snapshot.tofile(snapshotFile)
                    self.migrated_partial += (self.wf.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc] * self.wf.currentbck[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc])
                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current
                    self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")
     
        # Apply laplacian_filter filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_{self.approximation}_Nx{self.nx}_Nz{self.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    #Saving Boundaries
    def solveBackwardWaveEquationSavingBoundaries(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
        elif self.ABC == "CPML":
            self.d0, self.f_pico = self.wf.dampening_const()
        if self.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)

        self.rx = np.int32(self.rec_x/self.dx) + self.N_abc
        self.rz = np.int32(self.rec_z/self.dz) + self.N_abc
        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            self.sz = int(self.shot_z[shot]/self.dz) + self.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = self.Mute(seismogram, shot)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            for k in range(self.nt):
                self.forward_step(k)
                self.save_boundaries(k)
                # if k==1200:
                #     snapshot = self.wf.future[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc]
                #     snapshotFile = (f"{self.snapshotFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}FORWARD.bin")
                #     snapshot.tofile(snapshotFile)               
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current

            for t in range(self.nt - 1, -1, -1):
               self.reconstructed_step(t)
               self.apply_boundaries(t)
            #    if t%50 == 0:
            #     plt.figure()
            #     plt.imshow(self.wf.current)
            #     plt.show()

               self.backward_step(t)
               self.migrated_partial += (self.wf.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc] * self.wf.currentbck[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc])
                
               #swap
               self.wf.current, self.wf.future = self.wf.future, self.wf.current
               self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck

            self.wf.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")
     
        # Apply laplacian_filter filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_{self.approximation}_Nx{self.nx}_Nz{self.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    #Random Boundary Condintion
    def solveBackwardWaveEquationRBC(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        self.create_random_boundary()
        if self.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)

        self.rx = np.int32(self.rec_x/self.dx) + self.N_abc
        self.rz = np.int32(self.rec_z/self.dz) + self.N_abc
        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            self.sz = int(self.shot_z[shot]/self.dz) + self.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = self.Mute(seismogram, shot)

            self.migrated_partial = np.zeros_like(self.wf.migrated_image)

            for k in range(self.nt):
                self.forward_step(k)
                if k%500 == 0:
                    plt.figure()
                    plt.imshow(self.wf.current)
                    plt.show()
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            
            for t in range(self.nt - 1, -1, -1):
                self.reconstructed_step(t)
                self.backward_step(t)
                if t%500 == 0:
                    plt.figure()
                    plt.imshow(self.wf.current)
                    plt.show()
                self.migrated_partial += (self.wf.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc] * self.wf.currentbck[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc])
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")
     
        # Apply Laplacian filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_{self.approximation}_Nx{self.nx}_Nz{self.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    def SolveBackwardWaveEquation(self):
        if self.migration == "onthefly":
            self.solveBackwardWaveEquationOntheFly()
        elif self.migration == "checkpoint":
            self.solveBackwardWaveEquationCheckpointing()
        elif self.migration == "SB":
            self.solveBackwardWaveEquationSavingBoundaries()
        elif self.migration == "RBC":
            self.solveBackwardWaveEquationRBC()
        else:
            raise ValueError("Unknown migration method. Choose 'onthefly','checkpoint' or 'SB'.")
        print(f"info: Migration solved")


