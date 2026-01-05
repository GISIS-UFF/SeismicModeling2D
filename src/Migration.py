import numpy as np
import pandas as pd
import json
from Modeling2D import wavefield
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
        traveltimes = dist / v0_rec + 3 * t_lag 
        
        for r in range(self.Nrec): 
            mute_samples = int(traveltimes[r] / self.dt)
            muted[:mute_samples, r] = 0 
                
        return muted
        
    def laplacian(self, f):
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
    
    def load_checkpoint(self, checkpointFile):
        with open(checkpointFile, "rb") as file:
            count = self.nx_abc * self.nz_abc
            self.wf.current = np.fromfile(file, np.float32, count).reshape(self.nz_abc, self.nx_abc)
            self.wf.future  = np.fromfile(file, np.float32, count).reshape(self.nz_abc, self.nx_abc)

            if self.ABC == "CPML":
                count_x = self.nz_abc * (self.N_abc+4)
                count_z = (self.N_abc+4) * self.nx_abc

                self.wf.PsixFR  = np.fromfile(file, np.float32, count_x).reshape(self.nz_abc, self.N_abc+4)
                self.wf.PsixFL  = np.fromfile(file, np.float32, count_x).reshape(self.nz_abc, self.N_abc+4)
                self.wf.PsizFU  = np.fromfile(file, np.float32, count_z).reshape(self.N_abc+4, self.nx_abc)
                self.wf.PsizFD  = np.fromfile(file, np.float32, count_z).reshape(self.N_abc+4, self.nx_abc)

                self.wf.ZetaxFR = np.fromfile(file, np.float32, count_x).reshape(self.nz_abc, self.N_abc+4)
                self.wf.ZetaxFL = np.fromfile(file, np.float32, count_x).reshape(self.nz_abc, self.N_abc+4)
                self.wf.ZetazFU = np.fromfile(file, np.float32, count_z).reshape(self.N_abc+4, self.nx_abc)
                self.wf.ZetazFD = np.fromfile(file, np.float32, count_z).reshape(self.N_abc+4, self.nx_abc)
    
    def build_windows(self):
        windows = []
        for t0 in range(0, self.nt, self.step):
            t1 = min(t0 + self.step - 1, self.nt - 1)
            windows.append((t0, t1))
        return windows[::-1]

    def solveBackwardAcousticWaveEquation(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.wf.ExpandModel(self.wf.vp)
        self.A = self.wf.createCerjanVector()
        
        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        windows = self.build_windows()
        
        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.wf.current.fill(0)
            self.wf.future.fill(0)
            self.wf.currentbck.fill(0)
            self.wf.futurebck.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = self.Mute(seismogram, shot)

            self.migrated_partial = np.zeros_like(self.wf.migrated_image)          

            for (t0, t1) in windows:

                checkpointFile = (f"{self.checkpointFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{t0}.bin")
                self.load_checkpoint(checkpointFile)

                for k in range(t0, t1 + 1):        
                    self.wf.current[sz,sx] += self.wf.source[k]
                    self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)

                    # Apply absorbing boundary condition
                    self.wf.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.future, self.A)
                    self.wf.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.current, self.A)

                    self.wf.savefield[k-t0, :, :] = self.wf.current[self.N_abc:self.nz_abc - self.N_abc, self.N_abc:self.nx_abc - self.N_abc]
        
                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current  

                # Begin backward propagation
                for t in range(t1, t0 - 1, -1):
                    self.wf.currentbck[rz, rx] += self.muted_seismogram[t, :]
                    
                    self.wf.futurebck = updateWaveEquation(self.wf.futurebck, self.wf.currentbck, self.vp_exp,self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)

                    self.wf.futurebck = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.futurebck, self.A)
                    self.wf.currentbck = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.currentbck, self.A)
                    self.migrated_partial += self.wf.savefield[t-t0, :, :] * self.wf.currentbck[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc] 

                    self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck

            self.wf.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")
        
        # Apply Laplacian filter 
        self.wf.migrated_image = self.laplacian(self.wf.migrated_image)
        
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_{self.approximation}_Nx{self.nx}_Nz{self.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")


    def solveBackwardAcousticWaveEquationCPML(self):
        print(f"info: Solving backward acoustic CPML wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.wf.ExpandModel(self.wf.vp)
        self.d0, self.f_pico = self.wf.dampening_const()

        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        windows = self.build_windows()

        for shot in range(self.Nshot):
            print(f"info: Starting backward migration for shot {shot+1}")
            self.wf.current.fill(0)
            self.wf.future.fill(0)
            self.wf.currentbck.fill(0)
            self.wf.futurebck.fill(0)
            self.wf.PsixFR.fill(0)
            self.wf.PsixFL.fill(0)
            self.wf.PsizFU.fill(0)  
            self.wf.PsizFD.fill(0) 
            self.wf.ZetaxFR.fill(0)
            self.wf.ZetaxFL.fill(0)
            self.wf.ZetazFU.fill(0)
            self.wf.ZetazFD.fill(0)
            self.wf.PsixFRbck.fill(0)
            self.wf.PsixFLbck.fill(0)
            self.wf.PsizFUbck.fill(0)  
            self.wf.PsizFDbck.fill(0) 
            self.wf.ZetaxFRbck.fill(0)
            self.wf.ZetaxFLbck.fill(0)
            self.wf.ZetazFUbck.fill(0)
            self.wf.ZetazFDbck.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc  

            self.migrated_partial = np.zeros_like(self.wf.migrated_image)

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = self.Mute(seismogram, shot)  
                    
            # Begin backward propagation
            for (t0, t1) in windows:

                checkpointFile = (f"{self.checkpointFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{t0}.bin")
                self.load_checkpoint(checkpointFile)

                for k in range(t0, t1+1):
                    self.wf.current[sz,sx] += self.wf.source[k]
                    self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.nx_abc, self.nz_abc, self.wf.current, self.dx, self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                    self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.nx_abc, self.nz_abc, self.wf.current, self.dx,self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                    self.wf.future = updateWaveEquationCPML(self.wf.future, self.wf.current, self.vp_exp, self.nx_abc, self.nz_abc, self.dz, self.dx, self.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.N_abc)
                    
                    self.wf.savefield[k-t0, :, :] = self.wf.current[self.N_abc:self.nz_abc - self.N_abc, self.N_abc:self.nx_abc - self.N_abc]

                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current

                for t in range(t1, t0 - 1, -1):
                    self.wf.currentbck[rz, rx] += self.muted_seismogram[t, :]
                    self.wf.PsixFRbck, self.wf.PsixFLbck, self.wf.PsizFUbck, self.wf.PsizFDbck = updatePsi(self.wf.PsixFRbck, self.wf.PsixFLbck,self.wf.PsizFUbck, self.wf.PsizFDbck, self.nx_abc, self.nz_abc, self.wf.currentbck, self.dx, self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                    self.wf.ZetaxFRbck, self.wf.ZetaxFLbck, self.wf.ZetazFUbck, self.wf.ZetazFDbck = updateZeta(self.wf.PsixFRbck, self.wf.PsixFLbck, self.wf.ZetaxFRbck, self.wf.ZetaxFLbck,self.wf.PsizFUbck, self.wf.PsizFDbck, self.wf.ZetazFUbck, self.wf.ZetazFDbck, self.nx_abc, self.nz_abc, self.wf.currentbck, self.dx,self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                    self.wf.futurebck = updateWaveEquationCPML(self.wf.futurebck, self.wf.currentbck, self.vp_exp, self.nx_abc, self.nz_abc, self.dz, self.dx, self.dt, self.wf.PsixFRbck, self.wf.PsixFLbck, self.wf.PsizFUbck, self.wf.PsizFDbck, self.wf.ZetaxFRbck, self.wf.ZetaxFLbck, self.wf.ZetazFUbck, self.wf.ZetazFDbck, self.N_abc)
                    self.migrated_partial += self.wf.savefield[t-t0, :, :] * self.wf.currentbck[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc] 

                    self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck

            self.wf.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")

        # Apply Laplacian filter 
        self.wf.migrated_image = self.laplacian(self.wf.migrated_image)
            
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_{self.approximation}_Nx{self.nx}_Nz{self.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")
    
    def solveBackwardAcousticVTIWaveEquation(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.wf.ExpandModel(self.wf.vp)
        self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
        self.delta_exp = self.wf.ExpandModel(self.wf.delta)
        self.A = self.wf.createCerjanVector()
        
        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        windows = self.build_windows()
        
        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.wf.current.fill(0)
            self.wf.future.fill(0)
            self.wf.currentbck.fill(0)
            self.wf.futurebck.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = self.Mute(seismogram, shot)

            self.migrated_partial = np.zeros_like(self.wf.migrated_image)          

            for (t0, t1) in windows:

                checkpointFile = (f"{self.checkpointFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{t0}.bin")
                self.load_checkpoint(checkpointFile)

                for k in range(t0, t1 + 1):        
                    self.wf.current[sz,sx] += self.wf.source[k]
                    self.wf.future = updateWaveEquationVTI(self.wf.future, self.wf.current, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)

                    # Apply absorbing boundary condition
                    self.wf.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.future, self.A)
                    self.wf.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.current, self.A)

                    self.wf.savefield[k-t0, :, :] = self.wf.current[self.N_abc:self.nz_abc - self.N_abc, self.N_abc:self.nx_abc - self.N_abc]
        
                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current  

                # Begin backward propagation
                for t in range(t1, t0 - 1, -1):
                    self.wf.currentbck[rz, rx] += self.muted_seismogram[t, :]
                    
                    self.wf.futurebck = updateWaveEquationVTI(self.wf.futurebck, self.wf.currentbck, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)

                    self.wf.futurebck = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.futurebck, self.A)
                    self.wf.currentbck = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.currentbck, self.A)
                    self.migrated_partial += self.wf.savefield[t-t0, :, :] * self.wf.currentbck[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc] 

                    self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck

            self.wf.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")
        
        # Apply Laplacian filter 
        self.wf.migrated_image = self.laplacian(self.wf.migrated_image)
        
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_{self.approximation}_Nx{self.nx}_Nz{self.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    def solveBackwardAcousticVTIWaveEquationCPML(self):
        print(f"info: Solving backward acoustic CPML wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.wf.ExpandModel(self.wf.vp)
        self.d0, self.f_pico = self.wf.dampening_const()

        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        windows = self.build_windows()

        for shot in range(self.Nshot):
            print(f"info: Starting backward migration for shot {shot+1}")
            self.wf.current.fill(0)
            self.wf.future.fill(0)
            self.wf.currentbck.fill(0)
            self.wf.futurebck.fill(0)
            self.wf.PsixFR.fill(0)
            self.wf.PsixFL.fill(0)
            self.wf.PsizFU.fill(0)  
            self.wf.PsizFD.fill(0) 
            self.wf.ZetaxFR.fill(0)
            self.wf.ZetaxFL.fill(0)
            self.wf.ZetazFU.fill(0)
            self.wf.ZetazFD.fill(0)
            self.wf.PsixFRbck.fill(0)
            self.wf.PsixFLbck.fill(0)
            self.wf.PsizFUbck.fill(0)  
            self.wf.PsizFDbck.fill(0) 
            self.wf.ZetaxFRbck.fill(0)
            self.wf.ZetaxFLbck.fill(0)
            self.wf.ZetazFUbck.fill(0)
            self.wf.ZetazFDbck.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc  

            self.migrated_partial = np.zeros_like(self.wf.migrated_image)

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = self.Mute(seismogram, shot)  
                    
            # Begin backward propagation
            for (t0, t1) in windows:

                checkpointFile = (f"{self.checkpointFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{t0}.bin")
                self.load_checkpoint(checkpointFile)

                for k in range(t0, t1+1):
                    self.wf.current[sz,sx] += self.wf.source[k]
                    self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.nx_abc, self.nz_abc, self.wf.current, self.dx, self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                    self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.nx_abc, self.nz_abc, self.wf.current, self.dx,self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                    self.wf.future = updateWaveEquationVTICPML(self.wf.future, self.wf.current, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.nx_abc, self.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.N_abc)
                    self.wf.savefield[k-t0, :, :] = self.wf.current[self.N_abc:self.nz_abc - self.N_abc, self.N_abc:self.nx_abc - self.N_abc]

                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current

                for t in range(t1, t0 - 1, -1):
                    self.wf.currentbck[rz, rx] += self.muted_seismogram[t, :]
                    self.wf.PsixFRbck, self.wf.PsixFLbck, self.wf.PsizFUbck, self.wf.PsizFDbck = updatePsi(self.wf.PsixFRbck, self.wf.PsixFLbck,self.wf.PsizFUbck, self.wf.PsizFDbck, self.nx_abc, self.nz_abc, self.wf.currentbck, self.dx, self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                    self.wf.ZetaxFRbck, self.wf.ZetaxFLbck, self.wf.ZetazFUbck, self.wf.ZetazFDbck = updateZeta(self.wf.PsixFRbck, self.wf.PsixFLbck, self.wf.ZetaxFRbck, self.wf.ZetaxFLbck,self.wf.PsizFUbck, self.wf.PsizFDbck, self.wf.ZetazFUbck, self.wf.ZetazFDbck, self.nx_abc, self.nz_abc, self.wf.currentbck, self.dx,self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                    self.wf.futurebck = updateWaveEquationVTICPML(self.wf.futurebck, self.wf.currentbck, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.nx_abc, self.nz_abc, self.wf.PsixFRbck, self.wf.PsixFLbck, self.wf.PsizFUbck, self.wf.PsizFDbck, self.wf.ZetaxFRbck, self.wf.ZetaxFLbck, self.wf.ZetazFUbck, self.wf.ZetazFDbck, self.N_abc)
                    self.migrated_partial += self.wf.savefield[t-t0, :, :] * self.wf.currentbck[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc] 

                    self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck

            self.wf.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")
    
    def solveBackwardAcousticTTIWaveEquation(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.wf.ExpandModel(self.wf.vp)
        self.theta_exp = self.wf.ExpandModel(self.wf.theta)
        self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
        self.delta_exp = self.wf.ExpandModel(self.wf.delta)
        self.A = self.wf.createCerjanVector()
        
        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        windows = self.build_windows()
        
        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.wf.current.fill(0)
            self.wf.future.fill(0)
            self.wf.currentbck.fill(0)
            self.wf.futurebck.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = self.Mute(seismogram, shot)

            self.migrated_partial = np.zeros_like(self.wf.migrated_image)          

            for (t0, t1) in windows:

                checkpointFile = (f"{self.checkpointFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{t0}.bin")
                self.load_checkpoint(checkpointFile)

                for k in range(t0, t1 + 1):        
                    self.wf.current[sz,sx] += self.wf.source[k]
                    self.wf.future = updateWaveEquationTTI(self.wf.future, self.wf.current, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)

                    # Apply absorbing boundary condition
                    self.wf.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.future, self.A)
                    self.wf.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.current, self.A)

                    self.wf.savefield[k-t0, :, :] = self.wf.current[self.N_abc:self.nz_abc - self.N_abc, self.N_abc:self.nx_abc - self.N_abc]
        
                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current  

                # Begin backward propagation
                for t in range(t1, t0 - 1, -1):
                    self.wf.currentbck[rz, rx] += self.muted_seismogram[t, :]
                    
                    self.wf.futurebck = updateWaveEquationTTI(self.wf.futurebck, self.wf.currentbck, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)

                    self.wf.futurebck = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.futurebck, self.A)
                    self.wf.currentbck = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.wf.currentbck, self.A)
                    self.migrated_partial += self.wf.savefield[t-t0, :, :] * self.wf.currentbck[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc] 

                    self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck

            self.wf.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")
        
        # Apply Laplacian filter 
        self.wf.migrated_image = self.laplacian(self.wf.migrated_image)
        
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_{self.approximation}_Nx{self.nx}_Nz{self.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    def SolveBackwardWaveEquation(self):
        if self.approximation == "acoustic" and self.ABC == "cerjan":
            self.solveBackwardAcousticWaveEquation()
        elif self.approximation == "acoustic" and self.ABC == "CPML":
            self.solveBackwardAcousticWaveEquationCPML()
        elif self.approximation == "VTI" and self.ABC == "cerjan":
            self.solveBackwardAcousticVTIWaveEquation()
        elif self.approximation == "VTI" and self.ABC == "CPML":
            self.solveBackwardAcousticVTIWaveEquationCPML()
        elif self.approximation == "TTI" and self.ABC == "cerjan":
            self.solveBackwardAcousticTTIWaveEquation()
        # elif self.approximation == "TTI" and self.ABC == "CPML":
        #     self.solveBackwardAcousticTTIWaveEquationCPML()
        else:
            raise ValueError("ERROR: Unknown approximation. Choose 'acoustic', 'VTI' or 'TTI'. Otherwise, unknown Absorbing Boundary Condition. Choose 'cerjan' or 'CPML'.")
        print(f"info: Wave equation solved")