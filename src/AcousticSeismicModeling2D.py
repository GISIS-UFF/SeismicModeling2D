import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time

from utils import ricker
from utils import AnalyticalModel
from utils import updateWaveEquation
from utils import updateWaveEquationCPML
from utils import updateWaveEquationVTI
from utils import updateWaveEquationVTICPML
from utils import updateWaveEquationTTI
from utils import updateWaveEquationTTICPML
from utils import AbsorbingBoundary
from utils import updatePsi
from utils import updateZeta
from utils import updatePsiVTI
from utils import updateZetaVTI
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

        # Source and receiver files
        self.rec_file = self.parameters["rec_file"]
        self.src_file = self.parameters["src_file"]

        # Velocity model file
        self.vpFile = self.parameters["vpFile"]
        self.vsFile = self.parameters["vsFile"]
        self.thetaFile = self.parameters["thetaFile"]

        # Snapshot flag
        self.frame      = self.parameters["frame"] # time steps to save snapshots
        self.shot_frame = self.parameters["shot_frame"] # shots to save snapshots

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
        self.source = ricker(self.fcut, self.t)
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
        self.snapshot    = np.zeros([self.nt,self.nz_abc,self.nx_abc],dtype=np.float32)
        self.migrated_image = np.zeros((self.nz, self.nx), dtype=np.float32)

        if self.approximation in ["acousticCPML", "acousticVTICPML", "acousticTTICPML"]:
            # Initialize absorbing layers       
            self.PsixFR      = np.zeros([self.nz_abc, self.N_abc+8], dtype=np.float32)
            self.PsixFL      = np.zeros([self.nz_abc, self.N_abc+8], dtype=np.float32)     
            self.PsizFU      = np.zeros([self.N_abc+8, self.nx_abc], dtype=np.float32) 
            self.PsizFD      = np.zeros([self.N_abc+8, self.nx_abc], dtype=np.float32)       
            self.ZetaxFR     = np.zeros([self.nz_abc, self.N_abc+8], dtype=np.float32)
            self.ZetaxFL     = np.zeros([self.nz_abc, self.N_abc+8], dtype=np.float32)
            self.ZetazFU     = np.zeros([self.N_abc+8, self.nx_abc], dtype=np.float32)
            self.ZetazFD     = np.zeros([self.N_abc+8, self.nx_abc], dtype=np.float32)
        

        print(f"info: Wavefields initialized: {self.nx}x{self.nz}x{self.nt}")

        #create or import velocity model
        if (self.vpFile==None):
            self.vpFile = "VpModel"
            self.createLayeredVpModel(self.vpLayer1,self.vpLayer2)
        else:
            self.vp = self.ImportModel(self.vpFile)
        
        if self.approximation in ["acousticVTI", "acousticTTI", "acousticVTICPML", "acousticTTICPML"]:
            # Initialize velocity model and wavefields
            self.Qc = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
            self.Qf = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
            # Initialize epsilon and delta models
            self.epsilon = np.zeros([self.nz,self.nx],dtype=np.float32)
            self.delta = np.zeros([self.nz,self.nx],dtype=np.float32)

            if self.approximation in ["acousticVTICPML", "acousticTTICPML"]:
                # Initialize absorbing layers
                self.PsizqFU     = np.zeros([self.N_abc+8, self.nx_abc], dtype=np.float32)
                self.PsizqFD     = np.zeros([self.N_abc+8, self.nx_abc], dtype=np.float32)
                self.ZetazqFU    = np.zeros([self.N_abc+8, self.nx_abc], dtype=np.float32)
                self.ZetazqFD    = np.zeros([self.N_abc+8, self.nx_abc], dtype=np.float32)

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
                
        if self.approximation in ["acousticTTI", "acousticTTICPML"]:
            # Initialize vs and theta model
            self.vs = np.zeros([self.nz,self.nx], dtype=np.float32)
            self.theta = np.zeros([self.nz,self.nx],dtype=np.float32)

            if self.approximation == "acousticTTICPML":
                # Initialize absorbing layers
                self.PsixqFR     = np.zeros([self.nz_abc, self.N_abc+8], dtype=np.float32)
                self.PsixqFL     = np.zeros([self.nz_abc, self.N_abc+8], dtype=np.float32)
                self.ZetaxqFR    = np.zeros([self.nz_abc, self.N_abc+8], dtype=np.float32)  
                self.ZetaxqFL    = np.zeros([self.nz_abc, self.N_abc+8], dtype=np.float32) 

                self.ZetaxzFUR    = np.zeros([self.N_abc+8, self.N_abc+8], dtype=np.float32)
                self.ZetaxzFUL    = np.zeros([self.N_abc+8, self.N_abc+8], dtype=np.float32)
                self.ZetaxzFDL    = np.zeros([self.N_abc+8, self.N_abc+8], dtype=np.float32)
                self.ZetaxzFDR    = np.zeros([self.N_abc+8, self.N_abc+8], dtype=np.float32)

                self.ZetaxzqFUL   = np.zeros([self.N_abc+8, self.N_abc+8], dtype=np.float32)
                self.ZetaxzqFUR   = np.zeros([self.N_abc+8, self.N_abc+8], dtype=np.float32)
                self.ZetaxzqFDL   = np.zeros([self.N_abc+8, self.N_abc+8], dtype=np.float32)           
                self.ZetaxzqFDR   = np.zeros([self.N_abc+8, self.N_abc+8], dtype=np.float32)

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
        
    def createLayeredVpModel(self,v1, v2):
        self.vp[0:self.nz//2, :] = v1
        self.vp[self.nz//2:self.nz, :] = v2

        self.modelFile = f"{self.modelFolder}layeredvp_Nz{self.nz}_Nx{self.nx}.bin"
        self.vp.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")

    def createLayeredVsModel(self):
        vs1 = np.sqrt(self.vpLayer1*self.vpLayer1*(self.epsilonLayer1 - self.deltaLayer1)/0.8)
        vs2 = np.sqrt(self.vpLayer2*self.vpLayer2*(self.epsilonLayer2 - self.deltaLayer2)/0.8)
        self.vs[0:self.nz//2, :] = vs1
        self.vs[self.nz//2:self.nz, :] = vs2

        self.modelFile = f"{self.modelFolder}layeredvs_Nz{self.nz}_Nx{self.nx}.bin"
        self.vs.tofile(self.modelFile)
        print(f"info: Vs saved to {self.modelFile}")

    def createLayeredThetaModel(self, t1, t2):
        self.theta[0:self.nz//2, :] = t1
        self.theta[self.nz//2:self.nz, :] = t2

        self.modelFile = f"{self.modelFolder}layeredtheta_Nz{self.nz}_Nx{self.nx}.bin"
        self.theta.tofile(self.modelFile)
        print(f"info: Theta saved to {self.modelFile}")

    def createLayeredEpsilonModel(self,e1, e2):
        self.epsilon[0:self.nz//2, :] = e1
        self.epsilon[self.nz//2:self.nz, :] = e2

        self.modelFile = f"{self.modelFolder}layeredepsilon_Nz{self.nz}_Nx{self.nx}.bin"
        self.epsilon.tofile(self.modelFile)
        print(f"info: Epsilon saved to {self.modelFile}")

    def createLayeredDeltaModel(self, d1, d2):
        self.delta[0:self.nz//2, :] = d1
        self.delta[self.nz//2:self.nz, :] = d2

        self.modelFile = f"{self.modelFolder}layereddelta_Nz{self.nz}_Nx{self.nx}.bin"
        self.delta.tofile(self.modelFile)
        print(f"info: Delta saved to {self.modelFile}")

    def createModelFromVp(self):
        if not self.approximation in ["acousticVTI", "acousticTTI", "acousticVTICPML", "acousticTTICPML"]:
            raise ValueError("ERROR: Change approximation parameter to 'acousticVTI'or 'acousticTTI'.")
        
        if self.vpFile == None:
            raise ValueError("ERROR: Import or create a velocity model first.")
        if self.epsilonFile != None:
            raise ValueError("ERROR: Epsilon model already exists. Make sure epsilonFile = None.")
        if self.deltaFile != None:
            raise ValueError("ERROR:Delta model already exists.Make sure deltaFile = None.")
            
        idx_water = np.where(self.vp <= 1500)

        # create density model with Gardner's equation
        self.rho = np.zeros([self.nz,self.nx],dtype=np.float32)
        a, b = 0.23, 0.25
        self.rho = a * np.power(self.vp/0.3048,b)*1000 # Gardner relation - Rosa (2010) apud Gardner et al. (1974) pag. 496 rho = a * v^b
        self.rho[idx_water] = 1000.0 # water density
        # self.viewModel(self.rho, "Density Model")

        # create epsilon model epsilon = 0.25 rho - 0.3 - Petrov et al. (2021) 
        self.epsilon = np.zeros([self.nz,self.nx],dtype=np.float32)
        self.epsilon = 0.25 * self.rho/1000 - 0.3 # rho in g/cm3
        self.epsilon[idx_water] = 0.0 # water epsilon
        # self.viewModel(self.epsilon, "Epsilon Model")
        self.epsilon.T.tofile(self.vpFile.replace(".bin","_epsilon.bin"))	
        print(f"info: Epsilon model saved to {self.vpFile.replace('.bin','_epsilon.bin')}")


        # create delta model delta = 0.125 rho - 0.1 - Petrov et al. (2021)
        self.delta = np.zeros([self.nz,self.nx],dtype=np.float32)
        self.delta = 0.125 * self.rho/1000 - 0.1 # rho in g/cm3
        self.delta[idx_water] = 0.0 # water delta
        # self.viewModel(self.delta, "Delta Model")
        self.delta.T.tofile(self.vpFile.replace(".bin","_delta.bin"))
        print(f"info: Delta model saved to {self.vpFile.replace('.bin','_delta.bin')}")

        #create vs model
        self.vs = np.zeros([self.nz,self.nx], dtype=np.float32)
        self.vs = np.sqrt(self.vp*self.vp*(self.epsilon - self.delta)/0.8)
        self.vs[idx_water] = 0.0
        # self.viewModel(self.vs, "Vs Model")
        self.vs.T.tofile(self.vpFile.replace(".bin","_vs.bin"))	
        print(f"info: Vs model saved to {self.vpFile.replace('.bin','_vs.bin')}")


        # plt.show()

    def viewSnapshotAnalyticalComparison(self,k):
        snapshot = np.fromfile(self.viewSnapshotFile, dtype=np.float32).reshape(self.nz, self.nx)
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot snapshot
        im = ax.imshow(snapshot[k, self.N_abc:-self.N_abc, self.N_abc:-self.N_abc], aspect='equal', cmap='gray', extent=[0, self.L, self.D, 0])
        ax.plot(self.rec_x, self.rec_z, 'bv', markersize=2, label='Receivers')
        ax.plot(self.shot_x, self.shot_z, 'r*', markersize=5, label='Sources')
        
        # Compute the analytical wavefront
        if self.approximation == "acoustic":
            vel = self.vp[ int(self.shot_z[0]/self.dz), int(self.shot_x[0]/self.dx) ]
            Rp = AnalyticalModel(vel, 0, 0, self.dt, self.fcut, self.frame[0])
        elif self.approximation in ["acousticVTI", "acousticTTI", "acousticVTICPML", "acousticTTICPML"]:
            vel = self.vp[ int(self.shot_z[0]/self.dz), int(self.shot_x[0]/self.dx) ]
            Rp = AnalyticalModel(vel, 0.2, 0.2, self.dt, self.fcut, self.frame[0])
        else:
            raise ValueError("Info: Unknown approximation.")

        # Source coordinates
        x0 = self.shot_x[0]
        z0 = self.shot_z[0]
        
        # coordenates of the analytical wavefront
        theta = np.linspace(0, 2*np.pi, 500)
        x_rp = x0 + Rp * np.sin(theta)
        z_rp = z0 + Rp * np.cos(theta)

        if self.approximation == "acousticTTI":
            angle = -60  
            angle_rad = np.radians(angle)

            x_shifted = x_rp - x0
            z_shifted = z_rp - z0

            x_rot = x_shifted * np.cos(angle_rad) - z_shifted * np.sin(angle_rad)
            z_rot = x_shifted * np.sin(angle_rad) + z_shifted * np.cos(angle_rad)

            x = x0 + x_rot
            z = z0 + z_rot

        else:
            x = x_rp
            z = z_rp

        # Plot the analytical wavefront    
        ax.plot(x, z, 'r', label='Analytical wavefront')
        ax.legend()
        ax.set_title(f"Snapshot at time step {self.frame[0]} (shot {1})")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True)
        plt.tight_layout()

        if self.approximation == "acoustic":
            plt.savefig(f"{self.snapshotFolder}SnapshotAnalyticalComparison_acoustic_{0}_shot{1}.png")
        if self.approximation == "acousticVTI":
            plt.savefig(f"{self.snapshotFolder}SnapshotAnalyticalComparison_acousticVTI_{0}_shot{1}.png")
        if self.approximation == "acousticTTI":
            plt.savefig(f"{self.snapshotFolder}SnapshotAnalyticalComparison_acousticTTI_{0}_shot{1}.png")

        plt.show()

    def Reflectioncoefficient(self):
        borda_ref = 10
        R_ref = 1e-3
        R = R_ref ** (self.N_abc/borda_ref)

        if self.N_abc >= 150:
            R = R_ref ** (150/borda_ref)

        return R

    def dampening_const(self):     
        M = 2
        Rcoef = self.Reflectioncoefficient()
        f_pico = self.fcut/3
        d0 = - (M + 1)* np.log(Rcoef) 

        return d0, f_pico
    
    def dampening_profiles(self,vp):     
        deltas=(self.dz, self.dx)
        M = 2
        Rcoef = self.Reflectioncoefficient()
        f_pico = self.fcut/3
        for iN, N in enumerate(vp.shape):  
            dk = deltas[iN]    
            bordaCPML = self.N_abc * dk
            d0 = - (M + 1) * np.log(Rcoef) / (2 * bordaCPML) 
            bx, ax, bz, az = np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32), np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32), np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32), np.zeros([self.nz_abc, self.nx_abc], dtype=np.float32)

            for j in range(self.nz_abc):
                for i in range(self.nx_abc):
                    if i >= 0 and i < self.N_abc or i >= self.nx_abc - self.N_abc:
                        d = 0
                        alpha = 0
                        if i < self.N_abc:
                            points_CPML = (self.N_abc - i - 1)*self.dx
                            posicao_relativa = points_CPML / bordaCPML
                            d = d0 * (posicao_relativa**M) * vp[j,i]
                            alpha = np.pi* f_pico * (1 - posicao_relativa**2)

                        elif i >= self.nx_abc - self.N_abc:
                            points_CPML = (i - self.nx_abc + self.N_abc)*self.dx
                            posicao_relativa = points_CPML / bordaCPML
                            d = d0 * (posicao_relativa**M) * vp[j,i]
                            alpha = np.pi* f_pico * (1 - posicao_relativa**2)

                        ax[j,i] = np.exp(-(d + alpha) * self.dt)
                        if (np.abs((d + alpha)) > 1e-6):
                            bx[j,i] = (d / (d + alpha)) * (ax[j,i] - 1)
                    
                    if j >= 0 and j < self.N_abc or j >= self.nz_abc - self.N_abc:
                        d = 0
                        alpha = 0
                        if j < self.N_abc:
                            points_CPML = (self.N_abc - j - 1)*self.dz
                            posicao_relativa = points_CPML / bordaCPML
                            d = d0 * (posicao_relativa**M) * vp[j,i]
                            alpha = np.pi* f_pico * (1 - posicao_relativa**2)

                        elif j >= self.nz_abc - self.N_abc:
                            points_CPML = (j - self.nz_abc + self.N_abc)*self.dz
                            posicao_relativa = points_CPML / bordaCPML
                            d = d0 * (posicao_relativa**M) * vp[j,i]
                            alpha = np.pi* f_pico * (1 - posicao_relativa**2)

                        az[j,i] = np.exp(-(d + alpha) * self.dt)
                        if (np.abs((d + alpha)) > 1e-6):
                            bz[j,i] = (d / (d + alpha)) * (az[j,i] - 1)
       
        return ax, bx, az, bz

    def Mute(self, seismogram, shot): 
        muted = seismogram.copy() 
        v0 = self.vp[0, :]
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
    
    # def Mute(self, seismogram, shot): 
    #     muted = seismogram.copy() 
    #     v0 = self.vp[0, :]
    #     rec_idx = (self.rec_x / self.dx).astype(int)
    #     v0_rec = v0[rec_idx]
    #     distz = self.rec_z - self.shot_z[shot]   
    #     distx = self.rec_x - self.shot_x[shot]   
    #     dist = np.sqrt(distx**2 + distz**2)
    #     t_lag = 2 * np.sqrt(np.pi) / self.fcut
    #     traveltimes = dist / v0_rec + 3 * t_lag 
        
    #     for r in range(self.Nrec): 
    #         mute_samples = int(traveltimes[r] / self.dt)
    #         hann = np.hanning(mute_samples)
    #         hann_invertido = 1 - hann
    #         muted[:mute_samples, r] = hann_invertido * muted[:mute_samples, r]

    #         plt.plot(hann_invertido)
    #         plt.plot(muted[:, r])
    #         plt.show()
                
    #     return muted
    
    def LastTimeStepWithSignificantSourceAmplitude(self):
        source_abs = np.abs(self.source)
        source_max = source_abs.max()
        for k in range(self.nt):
            if abs(self.source[k]) > 1e-3 * source_max:
                last_t = k

        return last_t

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
        
        elif self.approximation in ["acousticVTI", "acousticTTI", "acousticVTICPML", "acousticTTICPML"]:
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
        sb = 3 * self.N_abc
        A = np.ones(self.N_abc)
        for i in range(self.N_abc):
                fb = (self.N_abc - i) / (np.sqrt(2) * sb)
                A[i] = np.exp(-fb * fb)
                
        return A 
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
    
    def solveAcousticWaveEquationCPML(self):
        start_time = time.time()
        print(f"info: Solving acoustic CPML wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.d0, self.f_pico = self.dampening_const()
        self.ax, self.bx, self.az, self.bz =  self.dampening_profiles(self.vp_exp)

        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.snapshot.fill(0)
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
                self.current[sz,sx] += self.source[k]
                self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD = updatePsi(self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.nx_abc, self.nz_abc, self.current, self.dx, self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD = updateZeta(self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.nx_abc, self.nz_abc, self.current, self.dx,self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.future = updateWaveEquationCPML(self.future, self.current, self.vp_exp, self.nx_abc, self.nz_abc, self.dz, self.dx, self.dt, self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD, self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD, self.N_abc)
                
                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                self.snapshot[k,:,:] = self.current
                
                if (shot + 1) in self.shot_frame and k in self.frame:
                    snapshotFile = f"{self.snapshotFolder}Acoustic_CPML_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    self.snapshot[k,:,:].tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")     

                #swap
                self.current, self.future = self.future, self.current

            self.seismogramFile = f"{self.seismogramFolder}AcousticCPMLSeismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

             # backward propagation
            print(f"info: Starting backward migration for shot {shot+1}")
            self.current.fill(0)
            self.future.fill(0)
            self.PsixFR.fill(0)
            self.PsixFL.fill(0)
            self.PsizFU.fill(0)  
            self.PsizFD.fill(0) 
            self.ZetaxFR.fill(0)
            self.ZetaxFL.fill(0)
            self.ZetazFU.fill(0)
            self.ZetazFD.fill(0)
            self.migrated_partial = np.zeros_like(self.migrated_image)

            # Top muting
            self.muted_seismogram = self.Mute(self.seismogram, shot)

            # Last time step with significant source amplitude
            self.last_t = self.LastTimeStepWithSignificantSourceAmplitude()    
                    
            # Begin backward propagation
            for t in range(self.nt - 1, self.last_t, -1):
                for r in range(len(rx)):
                    self.current[rz[r], rx[r]] += self.muted_seismogram[t, r]

                self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD = updatePsi(self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.nx_abc, self.nz_abc, self.current, self.dx, self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD = updateZeta(self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.nx_abc, self.nz_abc, self.current, self.dx,self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.future = updateWaveEquationCPML(self.future, self.current, self.vp_exp, self.nx_abc, self.nz_abc, self.dz, self.dx, self.dt, self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD, self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD, self.N_abc)
                self.migrated_partial += self.snapshot[t, self.N_abc:self.nz_abc - self.N_abc, self.N_abc:self.nx_abc - self.N_abc] * self.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc] 

                self.current, self.future = self.future, self.current

            self.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")

        # Apply Laplacian filter 
        self.migrated_image = self.laplacian(self.migrated_image)
        
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_acoustic_CPML.bin"
        self.migrated_image.tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")
   
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
            self.snapshot.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc           

            for k in range(self.nt):        
                self.current[sz,sx] += self.source[k]
                self.future = updateWaveEquation(self.future, self.current, self.vp_exp, self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)

                # Apply absorbing boundary condition
                self.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                self.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                self.snapshot[k, :, :] = self.current

                if (shot + 1) in self.shot_frame and k in self.frame:
                    snapshotFile = f"{self.snapshotFolder}Acoustic_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    self.snapshot[k,:,:].tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")
                
                #swap
                self.current, self.future = self.future, self.current

            self.seismogramFile = f"{self.seismogramFolder}AcousticSeismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

            # backward propagation
            print(f"info: Starting backward migration for shot {shot+1}")
            self.current.fill(0)
            self.future.fill(0)
            self.migrated_partial = np.zeros_like(self.migrated_image)

            # Top muting
            self.muted_seismogram = self.Mute(self.seismogram, shot)

            # Last time step with significant source amplitude
            self.last_t = self.LastTimeStepWithSignificantSourceAmplitude()    
                    
            # Begin backward propagation
            for t in range(self.nt - 1, self.last_t, -1):
                for r in range(len(rx)):
                    self.current[rz[r], rx[r]] += self.muted_seismogram[t, r]
                self.future = updateWaveEquation(self.future, self.current, self.vp_exp,self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)

                self.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                self.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)

                self.migrated_partial += self.snapshot[t, self.N_abc:self.nz_abc - self.N_abc, self.N_abc:self.nx_abc - self.N_abc] * self.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc] 

                self.current, self.future = self.future, self.current

            self.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")

        # Apply Laplacian filter 
        self.migrated_image = self.laplacian(self.migrated_image)
        
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_acoustic.bin"
        self.migrated_image.tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")
        
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
            self.snapshot.fill(0)
            self.Qc.fill(0)
            self.Qf.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc 

            for k in range(self.nt):
                self.current[sz,sx] += self.source[k]
                self.Qc[sz,sx] += self.source[k]

                self.future,self.Qf = updateWaveEquationVTI(self.future, self.current, self.Qc, self.Qf, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            
                # Apply absorbing boundary condition
                self.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                self.Qf = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.Qf, self.A)  

                self.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)
                self.Qc = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.Qc, self.A)
            
                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]
                
                self.snapshot[k, :, :] = self.current

                if (shot + 1) in self.shot_frame and k in self.frame:
                    snapshotFile = f"{self.snapshotFolder}VTI_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    self.snapshot[k,:,:].tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")
                #swap
                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc

            self.seismogramFile = f"{self.seismogramFolder}VTIseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

            # backward propagation
            print(f"info: Starting backward migration for shot {shot+1}")
            self.current.fill(0)
            self.future.fill(0)
            self.Qc.fill(0)
            self.Qf.fill(0)
            self.migrated_partial = np.zeros_like(self.migrated_image)

            # Top muting
            self.muted_seismogram = self.Mute(self.seismogram, shot)

            # Last time step with significant source amplitude
            self.last_t = self.LastTimeStepWithSignificantSourceAmplitude()

            # Begin backward propagation
            for t in range(self.nt - 1, self.last_t, -1):
                for r in range(len(rx)):
                    self.current[rz[r], rx[r]] += self.muted_seismogram[t, r]
                    self.Qc[rz[r], rx[r]] += self.muted_seismogram[t, r]

                self.future, self.Qf = updateWaveEquationVTI(self.future, self.current, self.Qc, self.Qf, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                
                self.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                self.Qf = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.Qf, self.A)  

                self.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)
                self.Qc = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.Qc, self.A)
            
                self.migrated_partial += self.snapshot[t, self.N_abc:self.nz_abc - self.N_abc, self.N_abc:self.nx_abc - self.N_abc] * self.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc]

                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc

            self.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")

        # Apply Laplacian filter 
        self.migrated_image = self.laplacian(self.migrated_image)

        self.migratedFile = f"{self.migratedimageFolder}migrated_image_VTI.bin"
        self.migrated_image.tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    
    def solveAcousticVTIWaveEquationCPML(self):
        start_time = time.time()
        print(f"info: Solving acoustic VTI CPML wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.ax, self.bx, self.az, self.bz =  self.dampening_profiles(self.vp_exp)
        self.d0, self.f_pico = self.dampening_const()

        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.snapshot.fill(0)
            self.Qc.fill(0)
            self.Qf.fill(0)
            self.PsixFR.fill(0)
            self.PsixFL.fill(0)
            self.PsizqFU.fill(0)
            self.PsizqFD.fill(0)
            self.ZetaxFR.fill(0)
            self.ZetaxFL.fill(0)
            self.ZetazqFU.fill(0)
            self.ZetazqFD.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc 

            for k in range(self.nt):
                self.current[sz,sx] += self.source[k]
                self.Qc[sz,sx] += self.source[k]
                self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD = updatePsi(self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.nx_abc, self.nz_abc, self.current, self.dx, self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD = updateZeta(self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.nx_abc, self.nz_abc, self.current, self.dx,self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.PsizqFU, self.PsizqFD = updatePsiVTI(self.PsizqFU, self.PsizqFD, self.nx_abc, self.nz_abc, self.az, self.bz, self.Qc, self.dz, self.N_abc) 
                self.ZetazqFU, self.ZetazqFD = updateZetaVTI(self.PsizqFU, self.PsizqFD, self.ZetazqFU, self.ZetazqFD, self.nx_abc, self.nz_abc, self.az, self.bz, self.Qc, self.dz, self.N_abc)
                self.future,self.Qf = updateWaveEquationVTICPML(self.future, self.current, self.Qc,self.Qf, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.nx_abc, self.nz_abc, self.PsixFR, self.PsixFL, self.PsizqFU, self.PsizqFD, self.ZetaxFR, self.ZetaxFL, self.ZetazqFU, self.ZetazqFD, self.N_abc)
                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]
                
                self.snapshot[k, :, :] = self.current

                if (shot + 1) in self.shot_frame and k in self.frame:
                    snapshotFile = f"{self.snapshotFolder}VTI_CPML_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    self.snapshot[k,:,:].tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")

                #swap
                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc


            self.seismogramFile = f"{self.seismogramFolder}VTICPMLseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")
        
    def solveAcousticTTIWaveEquation(self):
        start_time = time.time()
        print(f"info: Solving acoustic TTI wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.vs_exp = self.ExpandModel(self.vs)
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
            self.snapshot.fill(0)
            self.Qc.fill(0)
            self.Qf.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

            for k in range(self.nt):
                self.current[sz,sx] += self.source[k]
                self.Qc[sz,sx] += self.source[k]

                self.future,self.Qf = updateWaveEquationTTI(self.future, self.current, self.Qc, self.Qf, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.vs_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
            
                # Apply absorbing boundary condition
                self.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                self.Qf = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.Qf, self.A)  

                self.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)
                self.Qc = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.Qc, self.A)
            
                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                self.snapshot[k, :, :] = self.current

                if (shot + 1) in self.shot_frame and k in self.frame:
                    snapshotFile = f"{self.snapshotFolder}TTI_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    self.snapshot[k,:,:].tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")
                
                #swap
                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc

            self.seismogramFile = f"{self.seismogramFolder}TTIseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

            # backward propagation
            print(f"info: Starting backward migration for shot {shot+1}")
            self.current.fill(0)
            self.future.fill(0)
            self.Qc.fill(0)
            self.Qf.fill(0)
            self.migrated_partial = np.zeros_like(self.migrated_image)

            # Top muting
            self.muted_seismogram = self.Mute(self.seismogram, shot)

            # Last time step with significant source amplitude
            self.last_t = self.LastTimeStepWithSignificantSourceAmplitude()    
                    
            # Begin backward propagation
            for t in range(self.nt - 1, self.last_t, -1):
                for r in range(len(rx)):
                    self.current[rz[r], rx[r]] += self.muted_seismogram[t, r]
                    self.Qc[rz[r], rx[r]] += self.muted_seismogram[t, r]

                self.future, self.Qf = updateWaveEquationTTI(self.future, self.current, self.Qc, self.Qf, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.vs_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
            
                self.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                self.Qf = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.Qf, self.A)  

                self.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)
                self.Qc = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.Qc, self.A)
            
                self.migrated_partial += self.snapshot[t, self.N_abc:self.nz_abc - self.N_abc, self.N_abc:self.nx_abc - self.N_abc] * self.current[self.N_abc:self.nz_abc - self.N_abc,self.N_abc:self.nx_abc - self.N_abc]

                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc

            self.migrated_image += self.migrated_partial
            print(f"info: Shot {shot+1} backward done.")

        # Apply Laplacian filter 
        self.migrated_image = self.laplacian(self.migrated_image)
    
        self.migratedFile = f"{self.migratedimageFolder}migrated_image_TTI.bin"
        self.migrated_image.tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    def solveAcousticTTIWaveEquationCPML(self):
        start_time = time.time()
        print(f"info: Solving acoustic TTI CPML wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.vs_exp = self.ExpandModel(self.vs)
        self.theta_exp = self.ExpandModel(self.theta)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.ax, self.bx, self.az, self.bz =  self.dampening_profiles(self.vp_exp)
        self.d0, self.f_pico = self.dampening_const()

        rx = np.int32(self.rec_x/self.dx) + self.N_abc
        rz = np.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.snapshot.fill(0)
            self.Qc.fill(0)
            self.Qf.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

            for k in range(self.nt):
                self.current[sz,sx] += self.source[k]
                self.Qc[sz,sx] += self.source[k]
                self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD = updatePsi(self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.nx_abc, self.nz_abc, self.current, self.dx, self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD = updateZeta(self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.nx_abc, self.nz_abc, self.current, self.dx,self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz, self.f_pico, self.d0, self.dt, self.vp_exp)
                self.PsizqFU, self.PsizqFD = updatePsiVTI(self.PsizqFU, self.PsizqFD, self.nx_abc, self.nz_abc, self.az, self.bz, self.Qc, self.dz, self.N_abc) 
                self.ZetazqFU, self.ZetazqFD = updateZetaVTI(self.PsizqFU, self.PsizqFD, self.ZetazqFU, self.ZetazqFD, self.nx_abc, self.nz_abc, self.az, self.bz, self.Qc, self.dz, self.N_abc)
                self.PsixqFR, self.PsixqFL = updatePsiTTI(self.PsixqFR, self.PsixqFL, self.nx_abc, self.nz_abc, self.ax, self.bx, self.Qc, self.dx, self.N_abc)
                self.ZetaxqFL, self.ZetaxqFR,self.ZetaxzFUL,self.ZetaxzFUR, self.ZetaxzFDL,self.ZetaxzFDR, self.ZetaxzqFUL,self.ZetaxzqFUR, self.ZetaxzqFDL,self.ZetaxzqFDR = updateZetaTTI(self.PsixqFR, self.PsixqFL, self.PsizFU, self.PsizFD, self.PsizqFU, self.PsizqFD, self.ZetaxqFL, self.ZetaxqFR, self.ZetaxzFUL,self.ZetaxzFUR, self.ZetaxzFDL, self.ZetaxzFDR, self.ZetaxzqFUL, self.ZetaxzqFUR, self.ZetaxzqFDL, self.ZetaxzqFDR, self.nx_abc, self.nz_abc, self.ax, self.bx, self.Qc, self.current, self.dx, self.dz, self.N_abc)
                self.future,self.Qf = updateWaveEquationTTICPML(self.future, self.current, self.Qc, self.Qf, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.vs_exp, self.epsilon_exp, self.delta_exp, self.theta_exp, self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD,self.PsixqFR, self.PsixqFL,self.PsizqFU, self.PsizqFD,self.ZetaxFR, self.ZetaxFL,self.ZetazFU, self.ZetazFD,self.ZetaxzFUL,self.ZetaxzFUR,self.ZetaxzFDL,self.ZetaxzFDR,self.ZetaxqFL, self.ZetaxqFR,self.ZetazqFU, self.ZetazqFD,self.ZetaxzqFUL,self.ZetaxzqFUR,self.ZetaxzqFDL,self.ZetaxzqFDR, self.N_abc)
            
                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                self.snapshot[k, :, :] = self.current

                if (shot + 1) in self.shot_frame and k in self.frame:
                    snapshotFile = f"{self.snapshotFolder}TTI_CPML_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    self.snapshot[k,:,:].tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")
                
                #swap
                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc

            self.seismogramFile = f"{self.seismogramFolder}TTICPMLseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def SolveWaveEquation(self):
        if self.approximation == "acoustic":
            self.solveAcousticWaveEquation()
        elif self.approximation == "acousticCPML":
            self.solveAcousticWaveEquationCPML()
        elif self.approximation == "acousticVTI":
            self.solveAcousticVTIWaveEquation()
        elif self.approximation == "acousticVTICPML":
            self.solveAcousticVTIWaveEquationCPML()
        elif self.approximation == "acousticTTI":
            self.solveAcousticTTIWaveEquation()
        elif self.approximation == "acousticTTICPML":
            self.solveAcousticTTIWaveEquationCPML()
        else:
            raise ValueError("ERROR: Unknown approximation. Choose 'acoustic', 'acousticVTI' or 'acousticTTI'.")
        print(f"info: Wave equation solved")
