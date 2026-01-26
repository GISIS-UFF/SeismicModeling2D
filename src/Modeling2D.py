import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time

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

class wavefield: 

    def __init__(self, parameters_path):
        self.parameters_path = parameters_path
        self.readParameters()
        self.readAcquisitionGeometry()

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

    def createSourceWavelet(self):
        # Create Ricker wavelet
        self.source = ricker(self.fcut, self.t, self.dt, self.dx, self.dz)
        print(f"info: Ricker Source wavelet created: {self.nt} samples")
        
    def ImportModel(self, filename):
        data = np.fromfile(filename, dtype=np.float32).reshape(self.nx, self.nz)
        print(f"info: Imported: {filename}")
        return data.T
        
    def ExpandModel(self, model_data):
        N = self.N_abc
        nz_abc, nx_abc = self.nz_abc, self.nx_abc
        
        model_exp = np.zeros((nz_abc, nx_abc),dtype=np.float32)
        model_exp[N:nz_abc-N, N:nx_abc-N] = model_data
        model_exp[0:N, N:nx_abc-N] = model_data[0, :]  
        model_exp[nz_abc-N:nz_abc, N:nx_abc-N] = model_data[-1, :]  
        model_exp[N:nz_abc-N, 0:N] = model_data[:, 0:1] 
        model_exp[N:nz_abc-N, nx_abc-N:nx_abc] = model_data[:, -1:]  
        model_exp[0:N, 0:N] = model_data[0, 0]  
        model_exp[0:N, nx_abc-N:nx_abc] = model_data[0, -1] 
        model_exp[nz_abc-N:nz_abc, 0:N] = model_data[-1, 0]  
        model_exp[nz_abc-N:nz_abc, nx_abc-N:nx_abc] = model_data[-1, -1] 
        print(f"info: Model expanded to {nz_abc}x{nx_abc}")
        # plt.figure()
        # plt.imshow(model_exp, cmap='jet', aspect='auto')
        
        return model_exp
    
    def initializeWavefields(self):
        # Initialize velocity model and wavefields
        self.vp         = np.zeros([self.nz,self.nx],dtype=np.float32)
        self.current    = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
        self.future     = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
        self.seismogram = np.zeros([self.nt,self.Nrec],dtype=np.float32)
        self.migrated_image = np.zeros((self.nz, self.nx), dtype=np.float32)

        if self.ABC == "CPML":
            # Initialize absorbing layers       
            self.PsixFR      = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
            self.PsixFL      = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)     
            self.PsizFU      = np.zeros([self.N_abc+4, self.nx_abc], dtype=np.float32) 
            self.PsizFD      = np.zeros([self.N_abc+4, self.nx_abc], dtype=np.float32)       
            self.ZetaxFR     = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
            self.ZetaxFL     = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
            self.ZetazFU     = np.zeros([self.N_abc+4, self.nx_abc], dtype=np.float32)
            self.ZetazFD     = np.zeros([self.N_abc+4, self.nx_abc], dtype=np.float32)
        if self.migration in ["checkpoint", "SB", "RBC", "onthefly"] :
            self.currentbck  = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
            self.futurebck   = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
            if self.migration == "SB":
                self.top   = np.zeros((self.nt, 4, self.nx), dtype=np.float32)
                self.bot   = np.zeros((self.nt, 4, self.nx), dtype=np.float32)
                self.left  = np.zeros((self.nt, self.nz, 4), dtype=np.float32)
                self.right = np.zeros((self.nt, self.nz, 4), dtype=np.float32)

        print(f"info: Wavefields initialized: {self.nx}x{self.nz}x{self.nt}")
        #create or import velocity model
        if (self.vpFile==None):
            self.vpFile = "VpModel"
            self.createLayeredVpModel(self.vpLayer1,self.vpLayer2)
        else:
            self.vp = self.ImportModel(self.vpFile)
        
        if self.approximation in ["VTI", "TTI"]:
            # Initialize epsilon and delta models
            self.epsilon = np.zeros([self.nz,self.nx],dtype=np.float32)
            self.delta = np.zeros([self.nz,self.nx],dtype=np.float32)

            #import epsilon and delta model
            if (self.epsilonFile == None):
                self.epsilonFile = "EpsilonModel"
                self.createLayeredEpsilonModel(self.epsilonLayer1,self.epsilonLayer2)
            else:
                self.epsilon = self.ImportModel(self.epsilonFile)

            if (self.deltaFile == None):
                self.deltaFile = "DeltaModel"
                self.createLayeredDeltaModel(self.deltaLayer1,self.deltaLayer2)
            else:
                self.delta = self.ImportModel(self.deltaFile)
                
        if self.approximation == "TTI":
            # Initialize vs and theta model
            self.vs = np.zeros([self.nz,self.nx], dtype=np.float32)
            self.theta = np.zeros([self.nz,self.nx],dtype=np.float32)

            #import vs and theta models
            if (self.vsFile == None):
                self.vsFile = "VsModel"
                self.createLayeredVsModel()
            else: 
                self.vs = self.ImportModel(self.vsFile)

            if (self.thetaFile == None):
                self.thetaFile = "ThetaModel"
                self.createLayeredThetaModel(np.radians(self.thetaLayer1), np.radians(self.thetaLayer2))
            else:
                self.theta = self.ImportModel(self.thetaFile)
                self.theta = np.radians(self.theta)
        
            if self.approximation == "TTI" and self.ABC == "CPML":
                # Initialize velocity model and wavefields
                self.Qc = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
                self.Qf = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
                # Initialize absorbing layers
                self.PsizqFU     = np.zeros([self.N_abc+4, self.nx_abc], dtype=np.float32)
                self.PsizqFD     = np.zeros([self.N_abc+4, self.nx_abc], dtype=np.float32)
                self.ZetazqFU    = np.zeros([self.N_abc+4, self.nx_abc], dtype=np.float32)
                self.ZetazqFD    = np.zeros([self.N_abc+4, self.nx_abc], dtype=np.float32)
                # Initialize absorbing layers
                # self.PsixqFR     = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                # self.PsixqFL     = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                # self.PsiauxFL     = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                # self.PsiauxFR     = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                # self.PsiauxqFL     = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                # self.PsiauxqFR    = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                

                # self.ZetaxqFR   = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)  
                # self.ZetaxqFL   = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)

                # self.ZetaxzFL    = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzFR    = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzqFL   = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzqFR   = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)

                # self.ZetaxzFLU    = np.zeros([self.N_abc+4, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzFLD    = np.zeros([self.N_abc+4, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzFRU    = np.zeros([self.N_abc+4, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzFRD    = np.zeros([self.N_abc+4, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzqFLU   = np.zeros([self.N_abc+4, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzqFLD   = np.zeros([self.N_abc+4, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzqFRU   = np.zeros([self.N_abc+4, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzqFRD   = np.zeros([self.N_abc+4, self.N_abc+4], dtype=np.float32)

                self.PsixF      = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)
                self.PsizF      = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32) 
                self.ZetaxF     = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)
                self.ZetazF     = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)
                # self.PsixqF     = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)
                # self.PsizqF     = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)
                # self.ZetaxqF    = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)
                # self.ZetazqF    = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)
                # self.ZetaxzF    = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)
                # self.ZetaxzqF   = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)
                self.ZetaxzF    = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)
                self.ZetazxF    = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)

                # self.ZetaxzFL   = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                # self.ZetaxzFR   = np.zeros([self.nz_abc, self.N_abc+4], dtype=np.float32)
                # self.ZetazxFU   = np.zeros([self.N_abc+4, self.nx_abc], dtype=np.float32)
                # self.ZetazxFD   = np.zeros([self.N_abc+4, self.nx_abc], dtype=np.float32)

    def createLayeredVpModel(self,v1, v2):
        self.vp[0:self.nz//2, :] = v1
        self.vp[self.nz//2:self.nz, :] = v2

        self.modelFile = f"{self.modelFolder}layeredvp_Nz{self.nz}_Nx{self.nx}.bin"
        self.vp.T.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")

    def createLayeredVsModel(self):
        vs1 = np.sqrt(self.vpLayer1*self.vpLayer1*(self.epsilonLayer1 - self.deltaLayer1)/0.8)
        vs2 = np.sqrt(self.vpLayer2*self.vpLayer2*(self.epsilonLayer2 - self.deltaLayer2)/0.8)
        self.vs[0:self.nz//2, :] = vs1
        self.vs[self.nz//2:self.nz, :] = vs2

        self.modelFile = f"{self.modelFolder}layeredvs_Nz{self.nz}_Nx{self.nx}.bin"
        self.vs.T.tofile(self.modelFile)
        print(f"info: Vs saved to {self.modelFile}")

    def createLayeredThetaModel(self, t1, t2):
        self.theta[0:self.nz//2, :] = t1
        self.theta[self.nz//2:self.nz, :] = t2

        self.modelFile = f"{self.modelFolder}layeredtheta_Nz{self.nz}_Nx{self.nx}.bin"
        self.theta.T.tofile(self.modelFile)
        print(f"info: Theta saved to {self.modelFile}")

    def createLayeredEpsilonModel(self,e1, e2):
        self.epsilon[0:self.nz//2, :] = e1
        self.epsilon[self.nz//2:self.nz, :] = e2

        self.modelFile = f"{self.modelFolder}layeredepsilon_Nz{self.nz}_Nx{self.nx}.bin"
        self.epsilon.T.tofile(self.modelFile)
        print(f"info: Epsilon saved to {self.modelFile}")

    def createLayeredDeltaModel(self, d1, d2):
        self.delta[0:self.nz//2, :] = d1
        self.delta[self.nz//2:self.nz, :] = d2

        self.modelFile = f"{self.modelFolder}layereddelta_Nz{self.nz}_Nx{self.nx}.bin"
        self.delta.T.tofile(self.modelFile)
        print(f"info: Delta saved to {self.modelFile}")

    def createModelFromVp(self):
        if not self.approximation in ["VTI", "TTI"]:
            raise ValueError("ERROR: Change approximation parameter to 'VTI'or 'TTI'.")
        
        if self.vpFile == None:
            raise ValueError("ERROR: Import or create a velocity model first.")
            
        idx_water = np.where(self.vp <= 1500)

        # create density model with Gardner's equation
        self.rho = np.zeros([self.nz,self.nx],dtype=np.float32)
        a, b = 0.23, 0.25
        self.rho = a * np.power(self.vp/0.3048,b)*1000 # Gardner relation - Rosa (2010) apud Gardner et al. (1974) pag. 496 rho = a * v^b
        self.rho[idx_water] = 1000.0 # water density

        # create epsilon model epsilon = 0.25 rho - 0.3 - Petrov et al. (2021) 
        self.epsilon = np.zeros([self.nz,self.nx],dtype=np.float32)
        self.epsilon = 0.25 * self.rho/1000 - 0.3 # rho in g/cm3
        self.epsilon[idx_water] = 0.0 # water epsilon
        self.epsilon.T.tofile(self.vpFile.replace(".bin","_epsilon.bin"))	
        print(f"info: Epsilon model saved to {self.vpFile.replace('.bin','_epsilon.bin')}")

        # create delta model delta = 0.125 rho - 0.1 - Petrov et al. (2021)
        self.delta = np.zeros([self.nz,self.nx],dtype=np.float32)
        self.delta = 0.125 * self.rho/1000 - 0.1 # rho in g/cm3
        self.delta[idx_water] = 0.0 # water delta
        self.delta.T.tofile(self.vpFile.replace(".bin","_delta.bin"))
        print(f"info: Delta model saved to {self.vpFile.replace('.bin','_delta.bin')}")

        #create vs model
        self.vs = np.zeros([self.nz,self.nx], dtype=np.float32)
        self.vs = np.sqrt(self.vp*self.vp*(self.epsilon - self.delta)/0.8)
        self.vs[idx_water] = 0.0
        self.vs.T.tofile(self.vpFile.replace(".bin","_vs.bin"))	
        print(f"info: Vs model saved to {self.vpFile.replace('.bin','_vs.bin')}")

    def Reflectioncoefficient(self):
        borda_ref = 10
        R_ref = 1e-3
        R = R_ref ** (self.N_abc/borda_ref)

        if self.N_abc >= 200:
            R = R_ref ** (150/borda_ref)

        return R

    def dampening_const(self):     
        M = 2
        Rcoef = self.Reflectioncoefficient()
        f_pico = self.fcut/3
        d0 = - (M + 1)* np.log(Rcoef) 

        return d0, f_pico

    def checkDispersionAndStability(self):
        if self.approximation == "acoustic":
            vp_min = np.min(self.vp)
            vp_max = np.max(self.vp)
            lambda_min = vp_min / self.fcut
            dx_lim = lambda_min / 4.28
            dt_lim = dx_lim / (4 * vp_max)
            print(f"info: Dispersion and stability check")
            print(f"info: Minimum velocity: {vp_min:.2f} m/s")
            print(f"info: Maximum velocity: {vp_max:.2f} m/s")
            print(f"info: Maximum frequency: {self.fcut:.2f} Hz")
            print(f"info: Current dx: {self.dx:.2f} m")
            print(f"info: Current dt: {self.dt:.5f} s")
            print(f"info: Critical dx: {dx_lim:.2f} m")
            print(f"info: Critical dt: {dt_lim:.5f} s")
            if self.dx <= dx_lim and self.dt <= dt_lim:
                print("info: Dispersion and stability conditions satisfied.")
            else:
                print("WARNING: Dispersion or stability conditions not satisfied.")
        
        elif self.approximation in ["VTI", "TTI"]:
            vp_min = np.min(self.vp)
            vpx = self.vp*np.sqrt(1+2*self.epsilon)
            vpx_max = np.max(vpx)
            lambda_min = vp_min / self.fcut
            dx_lim = lambda_min / 4.28
            dt_lim = dx_lim / (4 * vpx_max)
            print(f"info: Dispersion and stability check")
            print(f"info: Minimum velocity: {vp_min:.2f} m/s")
            print(f"info: Maximum velocity: {vpx_max:.2f} m/s")
            print(f"info: Maximum frequency: {self.fcut:.2f} Hz")
            print(f"info: Current dx: {self.dx:.2f} m")
            print(f"info: Current dt: {self.dt:.5f} s")
            print(f"info: Critical dx: {dx_lim:.2f} m")
            print(f"info: Critical dt: {dt_lim:.5f} s")
            if self.dx <= dx_lim and self.dt <= dt_lim:
                print("info: Dispersion and stability conditions satisfied.")
            else:
                print("WARNING: Dispersion or stability conditions not satisfied.")
    
    def createCerjanVector(self):
        sb = 4. * self.N_abc
        A = np.ones(self.N_abc)
        for i in range(self.N_abc):
                fb = (self.N_abc - i) / (np.sqrt(2.) * sb)
                A[i] = np.exp(-fb * fb)
                
        return A 
    
    def save_snapshot(self,shot, k):        
        if not self.snap:
            return
        if k > self.last_save:
            return
        if k % self.step != 0:
            return

        snapshot = self.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc]

        snapshotFile = (f"{self.snapshotFolder}{self.approximation}{self.ABC}_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin")
        snapshot.tofile(snapshotFile)
        print(f"info: Snapshot saved to {snapshotFile}")
    
    def save_seismogram(self,shot):        
        self.seismogramFile = f"{self.seismogramFolder}{self.approximation}{self.ABC}_seismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
        self.seismogram.tofile(self.seismogramFile)
        print(f"info: Seismogram saved to {self.seismogramFile}")

    def solveAcousticWaveEquation(self):
        start_time = time.time()
        print(f"info: Solving acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.A = self.createCerjanVector()
        
        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc           

            for k in range(self.nt):        
                self.future = updateWaveEquation(self.future, self.current, self.vp_exp, self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)
                self.future[sz,sx] += self.source[k]

                # Apply absorbing boundary condition
                self.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                self.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                self.save_snapshot(shot, k)
                
                #swap
                self.current, self.future = self.future, self.current
          
            self.save_seismogram(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def solveAcousticWaveEquationCPML(self):
        start_time = time.time()
        print(f"info: Solving acoustic CPML wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.d0, self.f_pico = self.dampening_const()

        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.PsixFR.fill(0)
            self.PsixFL.fill(0)
            self.PsizFU.fill(0)  
            self.PsizFD.fill(0) 
            self.ZetaxFR.fill(0)
            self.ZetaxFL.fill(0)
            self.ZetazFU.fill(0)
            self.ZetazFD.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc

            for k in range(self.nt):
                self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD = updatePsi(self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.nx_abc, self.nz_abc, self.current, self.dx, self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD = updateZeta(self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.nx_abc, self.nz_abc, self.current, self.dx,self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.future = updateWaveEquationCPML(self.future, self.current, self.vp_exp, self.nx_abc, self.nz_abc, self.dz, self.dx, self.dt, self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD, self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD, self.N_abc)
                
                self.future[sz,sx] += self.source[k]

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]
                self.save_snapshot(shot, k)

                #swap
                self.current, self.future = self.future, self.current

            self.save_seismogram(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def solveAcousticVTIWaveEquation(self):
        start_time = time.time()
        print(f"info: Solving acoustic VTI wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.A = self.createCerjanVector()

        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc 

            for k in range(self.nt):
                self.future= updateWaveEquationVTI(self.future, self.current, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                self.future[sz,sx] += self.source[k]
                # Apply absorbing boundary condition
                self.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                self.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                self.save_snapshot(shot, k)
                
                #swap
                self.current, self.future = self.future, self.current
            self.save_seismogram(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        
    def solveAcousticVTIWaveEquationCPML(self):
        start_time = time.time()
        print(f"info: Solving acoustic VTI CPML wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.d0, self.f_pico = self.dampening_const()

        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.PsixFR.fill(0)
            self.PsixFL.fill(0)
            self.PsizFU.fill(0)
            self.PsizFD.fill(0)
            self.ZetaxFR.fill(0)
            self.ZetaxFL.fill(0)
            self.ZetazFU.fill(0)
            self.ZetazFD.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc 

            for k in range(self.nt):
                self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD = updatePsi(self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.nx_abc, self.nz_abc, self.current, self.dx, self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD = updateZeta(self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.nx_abc, self.nz_abc, self.current, self.dx,self.dz, self.N_abc, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.future = updateWaveEquationVTICPML(self.future, self.current, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.nx_abc, self.nz_abc, self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD, self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD, self.N_abc)
                
                self.future[sz,sx] += self.source[k]

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                self.save_snapshot(shot, k)
                
                #swap
                self.current, self.future = self.future, self.current

            self.save_seismogram(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def solveAcousticTTIWaveEquation(self):
        start_time = time.time()
        print(f"info: Solving acoustic TTI wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.theta_exp = self.ExpandModel(self.theta)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.A = self.createCerjanVector()

        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

            for k in range(self.nt):
                self.future= updateWaveEquationTTI(self.future, self.current, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
                self.future[sz,sx] += self.source[k]
                # Apply absorbing boundary condition
                self.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                self.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]
                self.save_snapshot(shot, k)
                
                #swap
                self.current, self.future = self.future, self.current


            self.save_seismogram(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def solveAcousticTTIWaveEquationCPML(self):
        start_time = time.time()
        print(f"info: Solving acoustic TTI CPML wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.vs_exp = self.ExpandModel(self.vs)
        self.theta_exp = self.ExpandModel(self.theta)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.d0, self.f_pico = self.dampening_const()

        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.PsixF.fill(0)
            self.PsizF.fill(0)
            self.ZetaxF.fill(0)
            self.ZetazF.fill(0)
            self.ZetaxzF.fill(0)
            self.ZetazxF.fill(0)
            #colocar os sem ser por regiao
            # self.PsixFL.fill(0)
            # self.PsixFR.fill(0)
            # self.PsizFU.fill(0)
            # self.PsizFD.fill(0)
            # self.ZetaxFL.fill(0)
            # self.ZetaxFR.fill(0)
            # self.ZetazFU.fill(0)
            # self.ZetazFD.fill(0)
            # self.ZetaxzFL.fill(0)
            # self.ZetaxzFR.fill(0)
            # self.ZetazxFU.fill(0)
            # self.ZetazxFD.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

            for k in range(self.nt):
                self.PsixF, self.PsizF = updatePsiTTI(self.PsixF, self.PsizF, self.nx_abc, self.nz_abc,self.N_abc,self.vp_exp,self.f_pico,self.d0, self.current, self.dz, self.dx,self.dt)
                self.ZetaxF, self.ZetazF, self.ZetaxzF, self.ZetazxF = updateZetaTTI(self.PsixF, self.PsizF, self.ZetaxF, self.ZetazF, self.ZetaxzF, self.ZetazxF, self.nx_abc, self.nz_abc,self.N_abc,self.vp_exp,self.f_pico,self.d0,self.dt, self.current, self.dz, self.dx)
                self.future = updateWaveEquationTTICPML(self.future, self.current, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.theta_exp,self.nx_abc, self.nz_abc,self.PsixF,self.PsizF, self.ZetaxF, self.ZetazF, self.ZetaxzF,self.ZetazxF)
                
                self.future[sz,sx] += self.source[k]

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]
                self.save_snapshot(shot, k)
                
                #swap
                self.current, self.future = self.future, self.current

            self.save_seismogram(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    # def solveAcousticTTIWaveEquationCPML(self):
    #     start_time = time.time()
    #     print(f"info: Solving acoustic TTI CPML wave equation")
    #     # Expand models and Create absorbing layers
    #     self.vp_exp = self.ExpandModel(self.vp)
    #     self.vs_exp = self.ExpandModel(self.vs)
    #     self.theta_exp = self.ExpandModel(self.theta)
    #     self.epsilon_exp = self.ExpandModel(self.epsilon)
    #     self.delta_exp = self.ExpandModel(self.delta)
    #     self.ax, self.bx, self.az, self.bz =  self.dampening_profiles(self.vp_exp)
    #     self.d0, self.f_pico = self.dampening_const()

    #     rx = np.int32(self.rec_x/self.dx) + self.N_abc
    #     rz = np.int32(self.rec_z/self.dz) + self.N_abc

    #     for shot in range(self.Nshot):
    #         print(f"info: Shot {shot+1} of {self.Nshot}")
    #         self.current.fill(0)
    #         self.future.fill(0)
    #         self.seismogram.fill(0)
    #         self.snapshot.fill(0)
    #         self.PsixFL.fill(0)
    #         self.PsixFR.fill(0)
    #         self.PsizFU.fill(0)
    #         self.PsizFD.fill(0)
    #         self.ZetaxFL.fill(0)
    #         self.ZetaxFR.fill(0)
    #         self.ZetazFU.fill(0)
    #         self.ZetazFD.fill(0)
    #         self.ZetaxzFL.fill(0)
    #         self.ZetaxzFR.fill(0)
    #         self.ZetazxFU.fill(0)
    #         self.ZetazxFD.fill(0)


    #         # convert acquisition geometry coordinates to grid points
    #         sx = int(self.shot_x[shot]/self.dx) + self.N_abc
    #         sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

    #         for k in range(self.nt):
    #             self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD = updatePsi(self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.nx_abc, self.nz_abc, self.current, self.dx, self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
    #             self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD = updateZeta(self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.nx_abc, self.nz_abc, self.current, self.dx,self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
    #             self.ZetaxzFL, self.ZetaxzFR, self.ZetazxFU, self.ZetazxFD = updateZetaTTI(self.PsixFL, self.PsixFR, self.PsizFU, self.PsizFD, self.ZetaxzFL, self.ZetaxzFR, self.ZetazxFU, self.ZetazxFD, self.nx_abc, self.nz_abc, self.ax, self.bx, self.az, self.bz, self.current, self.dx, self.dz, self.N_abc)
    #             self.future = updateWaveEquationTTICPML(self.future, self.current, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.theta_exp,self.nx_abc, self.nz_abc, self.PsixFR, self.PsixFL,self.PsizFU,self.PsizFD, self.ZetaxFR, self.ZetaxFL,self.ZetazFU, self.ZetazFD, self.ZetaxzFL,self.ZetaxzFR,self.ZetazxFU,self.ZetazxFD, self.N_abc)
                # self.future[sz,sx] += self.source[k]

        #         Register seismogram
    #             self.seismogram[k, :] = self.current[rz, rx]
    #             self.save_snapshot(shot, k)
    #             
    #             #swap
    #             self.current, self.future = self.future, self.current

            # self.save_seismogram(shot)
    #         print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    # def solveAcousticTTIWaveEquationCPML(self):
    #     start_time = time.time()
    #     print(f"info: Solving acoustic TTI CPML wave equation")
    #     # Expand models and Create absorbing layers
    #     self.vp_exp = self.ExpandModel(self.vp)
    #     self.vs_exp = self.ExpandModel(self.vs)
    #     self.theta_exp = self.ExpandModel(self.theta)
    #     self.epsilon_exp = self.ExpandModel(self.epsilon)
    #     self.delta_exp = self.ExpandModel(self.delta)
    #     self.ax, self.bx, self.az, self.bz =  self.dampening_profiles(self.vp_exp)

    #     rx = np.int32(self.rec_x/self.dx) + self.N_abc
    #     rz = np.int32(self.rec_z/self.dz) + self.N_abc

    #     for shot in range(self.Nshot):
    #         print(f"info: Shot {shot+1} of {self.Nshot}")
    #         self.current.fill(0)
    #         self.future.fill(0)
    #         self.seismogram.fill(0)
    #         self.snapshot.fill(0)
    #         self.Qc.fill(0)
    #         self.Qf.fill(0)
    #         self.PsixF.fill(0)
    #         self.PsixqF.fill(0)
    #         self.PsizF.fill(0)
    #         self.PsizqF.fill(0)
    #         self.ZetaxF.fill(0)
    #         self.ZetazF.fill(0)
    #         self.ZetaxzF.fill(0)
    #         self.ZetaxqF.fill(0)
    #         self.ZetazqF.fill(0)
    #         self.ZetaxzqF.fill(0)

    #         # convert acquisition geometry coordinates to grid points
    #         sx = int(self.shot_x[shot]/self.dx) + self.N_abc
    #         sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

    #         for k in range(self.nt):
    #             # self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD = updatePsi(self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.nx_abc, self.nz_abc, self.current, self.dx, self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
    #             # self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD = updateZeta(self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.nx_abc, self.nz_abc, self.current, self.dx,self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
    #             # self.PsizqFU, self.PsizqFD = updatePsiVTI(self.PsizqFU, self.PsizqFD, self.nx_abc, self.nz_abc, self.az, self.bz, self.Qc, self.dz, self.N_abc) 
    #             # self.ZetazqFU, self.ZetazqFD = updateZetaVTI(self.PsizqFU, self.PsizqFD, self.ZetazqFU, self.ZetazqFD, self.nx_abc, self.nz_abc, self.az, self.bz, self.Qc, self.dz, self.N_abc)
    #             # self.PsixqFR, self.PsixqFL,self.PsiauxFL,self.PsiauxFR,self.PsiauxqFL,self.PsiauxqFR = updatePsiTTI(self.PsixqFR, self.PsixqFL,self.PsizFU,self.PsizFD,self.PsizqFU,self.PsizqFD,self.PsiauxFL,self.PsiauxFR,self.PsiauxqFL,self.PsiauxqFR, self.nx_abc,self.nz_abc, self.ax, self.bx, self.Qc,self.current, self.dx,self.dz, self.N_abc)
    #             # self.ZetaxqFL, self.ZetaxqFR, self.ZetaxzFLU, self.ZetaxzFLD, self.ZetaxzFRU, self.ZetaxzFRD, self.ZetaxzqFLU, self.ZetaxzqFLD, self.ZetaxzqFRU, self.ZetaxzqFRD, self.ZetaxzFL, self.ZetaxzFR, self.ZetaxzqFL, self.ZetaxzqFR = updateZetaTTI(self.PsixqFR, self.PsixqFL, self.PsizFU, self.PsizFD, self.PsizqFU, self.PsizqFD,self.PsiauxFL,self.PsiauxFR,self.PsiauxqFL,self.PsiauxqFR, self.ZetaxqFL, self.ZetaxqFR, self.ZetaxzFLU,self.ZetaxzFLD,self.ZetaxzFRU,self.ZetaxzFRD,self.ZetaxzqFLU,self.ZetaxzqFLD, self.ZetaxzqFRU,self.ZetaxzqFRD,self.ZetaxzFL, self.ZetaxzFR, self.ZetaxzqFL, self.ZetaxzqFR, self.nx_abc, self.nz_abc, self.ax, self.bx, self.Qc, self.current, self.dx, self.dz, self.N_abc)
    #             # self.future,self.Qf = updateWaveEquationTTICPML(self.future, self.current, self.Qc, self.Qf, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.vs_exp, self.epsilon_exp, self.delta_exp, self.theta_exp, self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD,self.PsixqFR, self.PsixqFL,self.PsizqFU, self.PsizqFD,self.PsiauxFL,self.PsiauxFR,self.PsiauxqFL,self.PsiauxqFR, self.ZetaxFR, self.ZetaxFL,self.ZetazFU, self.ZetazFD,self.ZetaxqFL, self.ZetaxqFR,self.ZetazqFU, self.ZetazqFD, self.ZetaxzFLU,self.ZetaxzFLD,self.ZetaxzFRU,self.ZetaxzFRD,self.ZetaxzqFLU,self.ZetaxzqFLD, self.ZetaxzqFRU,self.ZetaxzqFRD,self.ZetaxzFL, self.ZetaxzFR, self.ZetaxzqFL, self.ZetaxzqFR, self.N_abc)
    #             self.PsixF, self.PsixqF, self.PsizF, self.PsizqF = updatePsiTTI(self.PsixF, self.PsixqF,self.PsizF, self.PsizqF, self.nx_abc, self.nz_abc, self.az, self.ax, self.bz, self.bx, self.current, self.Qc, self.dz, self.dx)
    #             self.ZetaxF, self.ZetazF, self.ZetaxzF, self.ZetaxqF, self.ZetazqF, self.ZetaxzqF = updateZetaTTI(self.PsixF, self.PsizF,self.PsizqF,self.PsixqF, self.ZetaxF, self.ZetazF, self.ZetaxzF, self.ZetaxqF, self.ZetazqF, self.ZetaxzqF, self.nx_abc, self.nz_abc, self.az, self.ax, self.bz, self.bx, self.current, self.Qc, self.dz, self.dx)
    #             self.future,self.Qf = updateWaveEquationTTICPML(self.future, self.current, self.Qc, self.Qf, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.vs_exp, self.epsilon_exp, self.delta_exp, self.theta_exp, self.PsixF,self.PsizF,self.PsixqF,self.PsizqF,self.ZetaxF,self.ZetazF,self.ZetaxzF,self.ZetaxqF,self.ZetazqF,self.ZetaxzqF)
                # self.future[sz,sx] += self.source[k]
    #             self.Qf[sz,sx] += self.source[k]

        #       # Register seismogram
    #             self.seismogram[k, :] = self.current[rz, rx]
    #             self.save_snapshot(shot, k)
    #             
    #             #swap
    #             self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc

            # self.save_seismogram(shot)
    #         print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def SolveWaveEquation(self):
        if self.approximation == "acoustic" and self.ABC == "cerjan":
            self.solveAcousticWaveEquation()
        elif self.approximation == "acoustic" and self.ABC == "CPML":
            self.solveAcousticWaveEquationCPML()
        elif self.approximation == "VTI" and self.ABC == "cerjan":
            self.solveAcousticVTIWaveEquation()
        elif self.approximation == "VTI" and self.ABC == "CPML":
            self.solveAcousticVTIWaveEquationCPML()
        elif self.approximation == "TTI" and self.ABC == "cerjan":
            self.solveAcousticTTIWaveEquation()
        elif self.approximation == "TTI" and self.ABC == "CPML":
            self.solveAcousticTTIWaveEquationCPML()
        else:
            raise ValueError("ERROR: Unknown approximation. Choose 'acoustic', 'VTI' or 'TTI'. Otherwise, unknown Absorbing Boundary Condition. Choose 'cerjan' or 'CPML'.")
        print(f"info: Wave equation solved")
