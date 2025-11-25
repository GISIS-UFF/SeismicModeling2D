import matplotlib.pyplot as plt
import pandas as pd
import json
import time
import cupy as cp

from utils import ricker
from utils import AnalyticalModel
from utils import updateWaveEquationGPU
from utils import updateWaveEquationCPMLGPU
from utils import updateWaveEquationVTIGPU
from utils import updateWaveEquationVTICPMLGPU
from utils import updateWaveEquationTTIGPU
from utils import AbsorbingBoundaryGPU
from utils import updatePsiGPU
from utils import updateZetaGPU
from utils import updatePsiVTIGPU
from utils import updateZetaVTIGPU

class wavefield_GPU: 

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
        self.dx   = cp.float32(self.parameters["dx"])
        self.dz   = cp.float32(self.parameters["dz"])
        self.dt   = cp.float32(self.parameters["dt"])
        
        # Model size
        self.L    = self.parameters["L"]
        self.D    = self.parameters["D"]
        self.T    = self.parameters["T"]

        # Number of point for absorbing boundary condition
        self.N_abc = cp.int32(self.parameters["N_abc"])

        # Number of points in each direction
        self.nx = int(self.L/self.dx)+1
        self.nz = int(self.D/self.dz)+1
        self.nt = int(self.T/self.dt)+1

        self.nx_abc = cp.int32(self.nx + 2*self.N_abc)
        self.nz_abc = cp.int32(self.nz + 2*self.N_abc)

        # Define arrays for space and time
        self.x = cp.linspace(0, self.L, self.nx)
        self.z = cp.linspace(0, self.D, self.nz)
        self.t = cp.linspace(0, self.T, self.nt)

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
        self.source = ricker(self.fcut, self.t, self.dt, self.dx, self.dz)
        self.source = cp.asarray(self.source, dtype=cp.float32)
        print(f"info: Ricker Source wavelet created: {self.nt} samples")
        
    def ImportModel(self, filename):
        data = cp.fromfile(filename, dtype=cp.float32).reshape(self.nx, self.nz)
        print(f"info: Imported: {filename}")
        return data.T

    def ExpandModel(self, model_data):
        N = self.N_abc
        nz_abc, nx_abc = self.nz_abc, self.nx_abc
        
        model_exp = cp.zeros((nz_abc, nx_abc),dtype=cp.float32)
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
        self.vp         = cp.zeros([self.nz,self.nx],dtype=cp.float32)
        self.current    = cp.zeros([self.nz_abc,self.nx_abc],dtype=cp.float32)
        self.future     = cp.zeros([self.nz_abc,self.nx_abc],dtype=cp.float32)
        self.seismogram = cp.zeros([self.nt,self.Nrec],dtype=cp.float32)
        self.snapshot    = cp.zeros([len(self.frame),self.nz_abc,self.nx_abc],dtype=cp.float32)
        self.migrated_image = cp.zeros((self.nz, self.nx), dtype=cp.float32)

        if self.approximation in ["acousticCPMLGPU","acousticVTICPMLGPU", "acousticTTICPMLGPU"]:
            # Initialize absorbing layers       
            self.PsixFR      = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
            self.PsixFL      = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)     
            self.PsizFU      = cp.zeros([self.N_abc+4, self.nx_abc], dtype=cp.float32) 
            self.PsizFD      = cp.zeros([self.N_abc+4, self.nx_abc], dtype=cp.float32)       
            self.ZetaxFR     = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
            self.ZetaxFL     = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
            self.ZetazFU     = cp.zeros([self.N_abc+4, self.nx_abc], dtype=cp.float32)
            self.ZetazFD     = cp.zeros([self.N_abc+4, self.nx_abc], dtype=cp.float32)
        

        print(f"info: Wavefields initialized: {self.nx}x{self.nz}x{self.nt}")

        #create or import velocity model
        if (self.vpFile==None):
            self.vpFile = "VpModel"
            self.createLayeredVpModel(self.vpLayer1,self.vpLayer2)
        else:
            self.vp = self.ImportModel(self.vpFile)
        
        if self.approximation in ["acousticVTIGPU","acousticVTIGPU","acousticTTIGPU","acousticVTICPMLGPU","acousticTTICPMLGPU"]:
            # Initialize velocity model and wavefields
            self.Qc = cp.zeros([self.nz_abc,self.nx_abc],dtype=cp.float32)
            self.Qf = cp.zeros([self.nz_abc,self.nx_abc],dtype=cp.float32)
            # Initialize epsilon and delta models
            self.epsilon = cp.zeros([self.nz,self.nx],dtype=cp.float32)
            self.delta = cp.zeros([self.nz,self.nx],dtype=cp.float32)

            if self.approximation in ["acousticVTICPMLGPU", "acousticTTIGPU", "acousticTTICPMLGPU"]:
                # Initialize absorbing layers
                self.PsizqFU     = cp.zeros([self.N_abc+4, self.nx_abc], dtype=cp.float32)
                self.PsizqFD     = cp.zeros([self.N_abc+4, self.nx_abc], dtype=cp.float32)
                self.ZetazqFU    = cp.zeros([self.N_abc+4, self.nx_abc], dtype=cp.float32)
                self.ZetazqFD    = cp.zeros([self.N_abc+4, self.nx_abc], dtype=cp.float32)

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
                
        if self.approximation in ["acousticTTIGPU", "acousticTTICPMLGPU"]:
            # Initialize vs and theta model
            self.vs = cp.zeros([self.nz,self.nx], dtype=cp.float32)
            self.theta = cp.zeros([self.nz,self.nx],dtype=cp.float32)

            if self.approximation == "acousticTTICPMLGPU":
                # Initialize absorbing layers
                self.PsixqFR     = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
                self.PsixqFL     = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
                self.PsiauxFL     = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
                self.PsiauxFR     = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
                self.PsiauxqFL     = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
                self.PsiauxqFR    = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
                

                self.ZetaxqFR   = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)  
                self.ZetaxqFL   = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)

                self.ZetaxzFL    = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
                self.ZetaxzFR    = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
                self.ZetaxzqFL   = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)
                self.ZetaxzqFR   = cp.zeros([self.nz_abc, self.N_abc+4], dtype=cp.float32)

                self.ZetaxzFLU    = cp.zeros([self.N_abc+4, self.N_abc+4], dtype=cp.float32)
                self.ZetaxzFLD    = cp.zeros([self.N_abc+4, self.N_abc+4], dtype=cp.float32)
                self.ZetaxzFRU    = cp.zeros([self.N_abc+4, self.N_abc+4], dtype=cp.float32)
                self.ZetaxzFRD    = cp.zeros([self.N_abc+4, self.N_abc+4], dtype=cp.float32)
                self.ZetaxzqFLU   = cp.zeros([self.N_abc+4, self.N_abc+4], dtype=cp.float32)
                self.ZetaxzqFLD   = cp.zeros([self.N_abc+4, self.N_abc+4], dtype=cp.float32)
                self.ZetaxzqFRU   = cp.zeros([self.N_abc+4, self.N_abc+4], dtype=cp.float32)
                self.ZetaxzqFRD   = cp.zeros([self.N_abc+4, self.N_abc+4], dtype=cp.float32)

            #import vs and theta models
            if (self.vsFile == None):
                self.vsFile = "VsModel"
                self.createLayeredVsModel()
            else: 
                self.vs = self.ImportModel(self.vsFile)

            if (self.thetaFile == None):
                self.thetaFile = "ThetaModel"
                self.createLayeredThetaModel(cp.radians(self.thetaLayer1), cp.radians(self.thetaLayer2))
            else:
                self.theta = self.ImportModel(self.thetaFile)
                self.theta = cp.radians(self.theta)
        
    def createLayeredVpModel(self,v1, v2):
        self.vp[0:self.nz//2, :] = v1
        self.vp[self.nz//2:self.nz, :] = v2

        self.modelFile = f"{self.modelFolder}layeredvp_Nz{self.nz}_Nx{self.nx}.bin"
        self.vp.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")

    def createLayeredVsModel(self):
        vs1 = cp.sqrt(self.vpLayer1*self.vpLayer1*(self.epsilonLayer1 - self.deltaLayer1)/0.8)
        vs2 = cp.sqrt(self.vpLayer2*self.vpLayer2*(self.epsilonLayer2 - self.deltaLayer2)/0.8)
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
        if not self.approximation in ["acousticVTIGPU","acousticTTIGPU", "acousticVTICPMLGPU", "acousticTTICPMLGPU"]:
            raise ValueError("ERROR: Change approximation parameter to 'acousticVTIGPU'or 'acousticTTIGPU'.")
        
        if self.vpFile == None:
            raise ValueError("ERROR: Import or create a velocity model first.")
            
        idx_water = cp.where(self.vp <= 1500)

        # create density model with Gardner's equation
        self.rho = cp.zeros([self.nz,self.nx],dtype=cp.float32)
        a, b = 0.23, 0.25
        self.rho = a * cp.power(self.vp/0.3048,b)*1000 # Gardner relation - Rosa (2010) apud Gardner et al. (1974) pag. 496 rho = a * v^b
        self.rho[idx_water] = 1000.0 # water density
        # self.viewModel(self.rho, "Density Model")

        # create epsilon model epsilon = 0.25 rho - 0.3 - Petrov et al. (2021) 
        self.epsilon = cp.zeros([self.nz,self.nx],dtype=cp.float32)
        self.epsilon = 0.25 * self.rho/1000 - 0.3 # rho in g/cm3
        self.epsilon[idx_water] = 0.0 # water epsilon
        # self.viewModel(self.epsilon, "Epsilon Model")
        self.epsilon.T.tofile(self.vpFile.replace(".bin","_epsilon.bin"))	
        print(f"info: Epsilon model saved to {self.vpFile.replace('.bin','_epsilon.bin')}")


        # create delta model delta = 0.125 rho - 0.1 - Petrov et al. (2021)
        self.delta = cp.zeros([self.nz,self.nx],dtype=cp.float32)
        self.delta = 0.125 * self.rho/1000 - 0.1 # rho in g/cm3
        self.delta[idx_water] = 0.0 # water delta
        # self.viewModel(self.delta, "Delta Model")
        self.delta.T.tofile(self.vpFile.replace(".bin","_delta.bin"))
        print(f"info: Delta model saved to {self.vpFile.replace('.bin','_delta.bin')}")

        #create vs model
        self.vs = cp.zeros([self.nz,self.nx], dtype=cp.float32)
        self.vs = cp.sqrt(self.vp*self.vp*(self.epsilon - self.delta)/0.8)
        self.vs[idx_water] = 0.0
        # self.viewModel(self.vs, "Vs Model")
        self.vs.T.tofile(self.vpFile.replace(".bin","_vs.bin"))	
        print(f"info: Vs model saved to {self.vpFile.replace('.bin','_vs.bin')}")


        # plt.show()

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
        d0 = - (M + 1)* cp.log(Rcoef) 

        return d0, f_pico
    #VER O FOR DESSA FUNÇÃO
    def dampening_profiles(self,vp):     
        deltas=(self.dz, self.dx)
        M = 2
        Rcoef = self.Reflectioncoefficient()
        f_pico = self.fcut/3
        for iN, N in enumerate(vp.shape):  
            dk = deltas[iN]    
            bordaCPML = self.N_abc * dk
            d0 = - (M + 1) * cp.log(Rcoef) / (2 * bordaCPML) 
            bx, ax, bz, az = cp.zeros([self.nz_abc, self.nx_abc], dtype=cp.float32), cp.zeros([self.nz_abc, self.nx_abc], dtype=cp.float32), cp.zeros([self.nz_abc, self.nx_abc], dtype=cp.float32), cp.zeros([self.nz_abc, self.nx_abc], dtype=cp.float32)

            for j in range(self.nz_abc):
                for i in range(self.nx_abc):
                    if i >= 0 and i < self.N_abc or i >= self.nx_abc - self.N_abc:
                        d = 0
                        alpha = 0
                        if i < self.N_abc:
                            points_CPML = (self.N_abc - i - 1)*self.dx
                            posicao_relativa = points_CPML / bordaCPML
                            d = d0 * (posicao_relativa**M) * vp[j,i]
                            alpha = cp.pi* f_pico * (1 - posicao_relativa**2)

                        elif i >= self.nx_abc - self.N_abc:
                            points_CPML = (i - self.nx_abc + self.N_abc)*self.dx
                            posicao_relativa = points_CPML / bordaCPML
                            d = d0 * (posicao_relativa**M) * vp[j,i]
                            alpha = cp.pi* f_pico * (1 - posicao_relativa**2)

                        ax[j,i] = cp.exp(-(d + alpha) * self.dt)
                        if (cp.abs((d + alpha)) > 1e-6):
                            bx[j,i] = (d / (d + alpha)) * (ax[j,i] - 1)
                    
                    if j >= 0 and j < self.N_abc or j >= self.nz_abc - self.N_abc:
                        d = 0
                        alpha = 0
                        if j < self.N_abc:
                            points_CPML = (self.N_abc - j - 1)*self.dz
                            posicao_relativa = points_CPML / bordaCPML
                            d = d0 * (posicao_relativa**M) * vp[j,i]
                            alpha = cp.pi* f_pico * (1 - posicao_relativa**2)

                        elif j >= self.nz_abc - self.N_abc:
                            points_CPML = (j - self.nz_abc + self.N_abc)*self.dz
                            posicao_relativa = points_CPML / bordaCPML
                            d = d0 * (posicao_relativa**M) * vp[j,i]
                            alpha = cp.pi* f_pico * (1 - posicao_relativa**2)

                        az[j,i] = cp.exp(-(d + alpha) * self.dt)
                        if (cp.abs((d + alpha)) > 1e-6):
                            bz[j,i] = (d / (d + alpha)) * (az[j,i] - 1)
       
        return ax, bx, az, bz

    def Mute(self, seismogram, shot): 
        muted = seismogram.copy() 
        v0 = self.vp[0, :]
        rec_idx = (self.rec_x / self.dx).astype(int)
        v0_rec = v0[rec_idx]
        distz = self.rec_z - self.shot_z[shot]   
        distx = self.rec_x - self.shot_x[shot]   
        dist = cp.sqrt(distx**2 + distz**2)
        t_lag = 2 * cp.sqrt(cp.pi) / self.fcut
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
    #     dist = cp.sqrt(distx**2 + distz**2)
    #     t_lag = 2 * cp.sqrt(cp.pi) / self.fcut
    #     traveltimes = dist / v0_rec + 3 * t_lag 
        
    #     for r in range(self.Nrec): 
    #         mute_samples = int(traveltimes[r] / self.dt)
    #         hann = cp.hanning(mute_samples)
    #         hann_invertido = 1 - hann
    #         muted[:mute_samples, r] = hann_invertido * muted[:mute_samples, r]

    #         plt.plot(hann_invertido)
    #         plt.plot(muted[:, r])
    #         plt.show()
                
    #     return muted
    
    def LastTimeStepWithSignificantSourceAmplitude(self):
        source_abs = cp.abs(self.source)
        source_max = source_abs.max()
        for k in range(self.nt):
            if abs(self.source[k]) > 1e-3 * source_max:
                last_t = k

        return last_t

    def checkDispersionAndStability(self):
        if self.approximation in ["acoustic", "acousticCPML", "acousticGPU", "acousticCPMLGPU" ]:
            vp_min = cp.min(self.vp)
            vp_max = cp.max(self.vp)
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
        
        elif self.approximation in ["acousticVTIGPU","acousticTTIGPU","acousticVTICPMLGPU", "acousticTTICPMLGPU"]:
            vp_min = cp.min(self.vp)
            vpx = self.vp*cp.sqrt(1+2*self.epsilon)
            vpx_max = cp.max(vpx)
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
        A = cp.ones(self.N_abc)
        for i in range(self.N_abc):
                fb = (self.N_abc - i) / (cp.sqrt(2) * sb)
                A[i] = cp.exp(-fb * fb)
                
        return A 
    
    def laplacian(self, f):
        dim1,dim2 = cp.shape(f)
        g = cp.zeros([dim1,dim2])
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
    
    def solveAcousticWaveEquationGPU(self):
        start_time = time.time()
        print(f"info: Solving acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.A = self.createCerjanVector()

        threadsperblock = (32, 16)
        blockspergrid_x = int((self.current.shape[1] + threadsperblock[0] - 1) / threadsperblock[0])
        blockspergrid_y = int((self.current.shape[0] + threadsperblock[1] - 1) / threadsperblock[1])  
        blockspergrid   = (blockspergrid_x, blockspergrid_y)
       
        rx = cp.int32(self.rec_x/self.dx) + self.N_abc
        rz = cp.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.snapshot.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = cp.int32(self.shot_x[shot]/self.dx) + self.N_abc
            sz = cp.int32(self.shot_z[shot]/self.dz) + self.N_abc           

            for k in range(self.nt):        
                self.current[sz,sx] += self.source[k]
                updateWaveEquationGPU[blockspergrid,threadsperblock](self.future, self.current, self.vp_exp, self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)

                # Apply absorbing boundary condition
                AbsorbingBoundaryGPU[blockspergrid,threadsperblock](self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                AbsorbingBoundaryGPU[blockspergrid,threadsperblock](self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                if (shot + 1) in self.shot_frame and k in self.frame:
                    frame_idx = self.frame.index(k)
                    self.snapshot[frame_idx, :, :] = self.current
                    snapshotFile = f"{self.snapshotFolder}Acoustic_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    snapshot_cpu = cp.asnumpy(self.snapshot[frame_idx, :, :])
                    snapshot_cpu.tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")
                
                #swap
                self.current, self.future = self.future, self.current
            
            self.seismogramFile = f"{self.seismogramFolder}Acousticseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram_cpu = cp.asnumpy(self.seismogram)
            self.seismogram_cpu.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def solveAcousticWaveEquationCPMLGPU(self):
        start_time = time.time()
        print(f"info: Solving acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.ax, self.bx, self.az, self.bz =  self.dampening_profiles(self.vp_exp)

        threadsperblock = (32, 16)
        blockspergrid_x = int((self.current.shape[1] + threadsperblock[0] - 1) / threadsperblock[0])
        blockspergrid_y = int((self.current.shape[0] + threadsperblock[1] - 1) / threadsperblock[1])  
        blockspergrid   = (blockspergrid_x, blockspergrid_y)
       
        rx = cp.int32(self.rec_x/self.dx) + self.N_abc
        rz = cp.int32(self.rec_z/self.dz) + self.N_abc

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
            sx = cp.int32(self.shot_x[shot]/self.dx) + self.N_abc
            sz = cp.int32(self.shot_z[shot]/self.dz) + self.N_abc           

            for k in range(self.nt):        
                self.current[sz,sx] += self.source[k]

                updatePsiGPU[blockspergrid,threadsperblock](self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.nx_abc, self.nz_abc, self.current, self.dx, self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz)
                updateZetaGPU[blockspergrid,threadsperblock](self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.nx_abc, self.nz_abc, self.current, self.dx,self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz)
                updateWaveEquationCPMLGPU[blockspergrid,threadsperblock](self.future, self.current, self.vp_exp, self.nx_abc, self.nz_abc, self.dz, self.dx, self.dt, self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD, self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD, self.N_abc)

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                if (shot + 1) in self.shot_frame and k in self.frame:
                    frame_idx = self.frame.index(k)
                    self.snapshot[frame_idx, :, :] = self.current
                    snapshotFile = f"{self.snapshotFolder}Acoustic_CPML_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    snapshot_cpu = cp.asnumpy(self.snapshot[frame_idx, :, :])
                    snapshot_cpu.tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")
               
                #swap
                self.current, self.future = self.future, self.current
            
            self.seismogramFile = f"{self.seismogramFolder}AcousticCPMLseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram_cpu = cp.asnumpy(self.seismogram)
            self.seismogram_cpu.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def solveAcousticVTIWaveEquationGPU(self):
        start_time = time.time()
        print(f"info: Solving acoustic VTI wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.A = self.createCerjanVector()

        threadsperblock = (32, 16)
        blockspergrid_x = int((self.current.shape[1] + threadsperblock[0] - 1) / threadsperblock[0])
        blockspergrid_y = int((self.current.shape[0] + threadsperblock[1] - 1) / threadsperblock[1])  
        blockspergrid   = (blockspergrid_x, blockspergrid_y)
       
        rx = cp.int32(self.rec_x/self.dx) + self.N_abc
        rz = cp.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.snapshot.fill(0)
            self.Qc.fill(0)
            self.Qf.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = cp.int32(self.shot_x[shot]/self.dx) + self.N_abc
            sz = cp.int32(self.shot_z[shot]/self.dz) + self.N_abc 

            for k in range(self.nt):
                self.current[sz,sx] += self.source[k]
                self.Qc[sz,sx] += self.source[k]

                updateWaveEquationVTIGPU[blockspergrid,threadsperblock](self.future, self.current, self.Qc, self.Qf, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            
                # Apply absorbing boundary condition
                AbsorbingBoundaryGPU[blockspergrid,threadsperblock](self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                AbsorbingBoundaryGPU[blockspergrid,threadsperblock](self.N_abc, self.nz_abc, self.nx_abc, self.Qf, self.A)  

                AbsorbingBoundaryGPU[blockspergrid,threadsperblock](self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)
                AbsorbingBoundaryGPU[blockspergrid,threadsperblock](self.N_abc, self.nz_abc, self.nx_abc, self.Qc, self.A)
            
                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]
                
                if (shot + 1) in self.shot_frame and k in self.frame:
                    frame_idx = self.frame.index(k)
                    self.snapshot[frame_idx, :, :] = self.current
                    snapshotFile = f"{self.snapshotFolder}VTI_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    snapshot_cpu = cp.asnumpy(self.snapshot[frame_idx, :, :])
                    snapshot_cpu.tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")
                    
                #swap
                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc

            self.seismogramFile = f"{self.seismogramFolder}VTIseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram_cpu = cp.asnumpy(self.seismogram)
            self.seismogram_cpu.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def solveAcousticVTIWaveEquationCPMLGPU(self):
        start_time = time.time()
        print(f"info: Solving acoustic VTI CPML wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.A = self.createCerjanVector()
        self.ax, self.bx, self.az, self.bz =  self.dampening_profiles(self.vp_exp)

        threadsperblock = (32, 16)
        blockspergrid_x = int((self.current.shape[1] + threadsperblock[0] - 1) / threadsperblock[0])
        blockspergrid_y = int((self.current.shape[0] + threadsperblock[1] - 1) / threadsperblock[1])  
        blockspergrid   = (blockspergrid_x, blockspergrid_y)
       
        rx = cp.int32(self.rec_x/self.dx) + self.N_abc
        rz = cp.int32(self.rec_z/self.dz) + self.N_abc

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
            sx = cp.int32(self.shot_x[shot]/self.dx) + self.N_abc
            sz = cp.int32(self.shot_z[shot]/self.dz) + self.N_abc 

            for k in range(self.nt):
                self.current[sz,sx] += self.source[k]
                self.Qc[sz,sx] += self.source[k]
                updatePsiGPU[blockspergrid,threadsperblock](self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.nx_abc, self.nz_abc, self.current, self.dx, self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz)
                updateZetaGPU[blockspergrid,threadsperblock](self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.nx_abc, self.nz_abc, self.current, self.dx,self.dz, self.N_abc,self.ax,self.bx,self.az,self.bz)
                updatePsiVTIGPU[blockspergrid,threadsperblock](self.PsizqFU, self.PsizqFD, self.nx_abc, self.nz_abc, self.az, self.bz, self.Qc, self.dz, self.N_abc) 
                updateZetaVTIGPU[blockspergrid,threadsperblock](self.PsizqFU, self.PsizqFD, self.ZetazqFU, self.ZetazqFD, self.nx_abc, self.nz_abc, self.az, self.bz, self.Qc, self.dz, self.N_abc)
                updateWaveEquationVTICPMLGPU[blockspergrid,threadsperblock](self.future, self.current, self.Qc,self.Qf, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.nx_abc, self.nz_abc, self.PsixFR, self.PsixFL, self.PsizqFU, self.PsizqFD, self.ZetaxFR, self.ZetaxFL, self.ZetazqFU, self.ZetazqFD, self.N_abc)
                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                if (shot + 1) in self.shot_frame and k in self.frame:
                    frame_idx = self.frame.index(k)
                    self.snapshot[frame_idx, :, :] = self.current
                    snapshotFile = f"{self.snapshotFolder}VTI_CPML_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    snapshot_cpu = cp.asnumpy(self.snapshot[frame_idx, :, :])
                    snapshot_cpu.tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")
                
                #swap
                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc


            self.seismogramFile = f"{self.seismogramFolder}VTICPMLseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram_cpu = cp.asnumpy(self.seismogram)
            self.seismogram_cpu.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def solveAcousticTTIWaveEquationGPU(self):
        start_time = time.time()
        print(f"info: Solving acoustic TTI wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.vs_exp = self.ExpandModel(self.vs)
        self.theta_exp = self.ExpandModel(self.theta)
        self.A = self.createCerjanVector()

        threadsperblock = (32, 16)
        blockspergrid_x = int((self.current.shape[1] + threadsperblock[0] - 1) / threadsperblock[0])
        blockspergrid_y = int((self.current.shape[0] + threadsperblock[1] - 1) / threadsperblock[1])  
        blockspergrid   = (blockspergrid_x, blockspergrid_y)
       
        rx = cp.int32(self.rec_x/self.dx) + self.N_abc
        rz = cp.int32(self.rec_z/self.dz) + self.N_abc

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.snapshot.fill(0)
            self.Qc.fill(0)
            self.Qf.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = cp.int32(self.shot_x[shot]/self.dx) + self.N_abc
            sz = cp.int32(self.shot_z[shot]/self.dz) + self.N_abc            

            for k in range(self.nt):
                self.current[sz,sx] += self.source[k]
                self.Qc[sz,sx] += self.source[k]

                updateWaveEquationTTIGPU[blockspergrid,threadsperblock](self.future, self.current, self.Qc, self.Qf, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.vs_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
            
                # Apply absorbing boundary condition
                AbsorbingBoundaryGPU[blockspergrid,threadsperblock](self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                AbsorbingBoundaryGPU[blockspergrid,threadsperblock](self.N_abc, self.nz_abc, self.nx_abc, self.Qf, self.A)  

                AbsorbingBoundaryGPU[blockspergrid,threadsperblock](self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)
                AbsorbingBoundaryGPU[blockspergrid,threadsperblock](self.N_abc, self.nz_abc, self.nx_abc, self.Qc, self.A)
            
                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                if (shot + 1) in self.shot_frame and k in self.frame:
                    frame_idx = self.frame.index(k)
                    self.snapshot[frame_idx, :, :] = self.current
                    snapshotFile = f"{self.snapshotFolder}TTI_shot_{shot+1}_Nx{self.nx}_Nz{self.nz}_Nt{self.nt}_frame_{k}.bin"
                    snapshot_cpu = cp.asnumpy(self.snapshot[frame_idx, :, :])
                    snapshot_cpu.tofile(snapshotFile)
                    print(f"info: Snapshot saved to {snapshotFile}")

                #swap
                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc

            self.seismogramFile = f"{self.seismogramFolder}TTIseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram_cpu = cp.asnumpy(self.seismogram)
            self.seismogram_cpu.tofile(self.seismogramFile)
            print(f"info: Seismogram saved to {self.seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")


    def SolveWaveEquation(self):
        if self.approximation == "acousticGPU":
            self.solveAcousticWaveEquationGPU()
        elif self.approximation == "acousticCPMLGPU":
            self.solveAcousticWaveEquationCPMLGPU()
        elif self.approximation == "acousticVTIGPU":
            self.solveAcousticVTIWaveEquationGPU()
        elif self.approximation == "acousticVTICPMLGPU":
            self.solveAcousticVTIWaveEquationCPMLGPU()
        elif self.approximation == "acousticTTIGPU":
            self.solveAcousticTTIWaveEquationGPU()

        else:
            raise ValueError("ERROR: Unknown approximation. Choose 'acoustic', 'acousticVTI' or 'acousticTTI'.")
        print(f"info: Wave equation solved")
