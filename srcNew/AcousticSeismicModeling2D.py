import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import ricker
from utils import updateWaveEquation
from utils import updateWaveEquationVTI

class wavefield:
    approximation = "acousticVTI" # "acoustic" # 

    def __init__(self):
        self.readParameters()
        self.readAcquisitionGeometry()

    def readParameters(self):
        self.dx   = 10.
        self.dz   = 10.
        self.dt   = 0.001
        
        # Model size
        self.L    = 3820
        self.D    = 1400
        self.T    = 4

        # Number of point for absorbing boundary condition
        self.N_abc = 50

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
        self.fcut = 60.

        self.seismogramFolder = "../outputs/seismograms/"

        # Source and receiver files
        self.rec_file = "../inputs/receivers.csv"
        self.src_file = "../inputs/sources.csv"

        # Velocity model file
        self.vpFile =  None#"../inputs/marmousi_vp_383x141.bin"

        # Snapshot flag
        self.snap       = False
        self.snapshot = []
        self.frame      = 500
        self.folderSnapshot = "../outputs/snapshots/"

        if self.approximation == "acousticVTI":
            # Anisotropy parameters files
            self.epsilonFile = None #"../inputs/epsilon_model.bin"  
            self.deltaFile = None #"../inputs/delta_model.bin"     

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

    def viewSourceWavelet(self):
        plt.figure()
        plt.plot(self.t, self.source)
        plt.title("Source Wavelet")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()
        plt.savefig("source_wavelet.png")
        
    def ImportModel(self, filename):
        data = np.fromfile(filename, dtype=np.float32).reshape(self.nx, self.nz)
        print(f"info: Imported: {filename}")
        return data.T

    def ExpandModel(self, model_data):
        nz, nx = self.nz, self.nx
        N = self.N_abc
        nz_abc, nx_abc = self.nz_abc, self.nx_abc
        
        model_exp = np.zeros((nz_abc, nx_abc))
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
        plt.figure()
        plt.imshow(model_exp, cmap='jet', aspect='auto')
        
        return model_exp
    
    def initializeWavefields(self):
        # Initialize velocity model and wavefields
        self.vp         = np.zeros([self.nz,self.nx])

        self.current    = np.zeros([self.nz_abc,self.nx_abc])
        self.past       = np.zeros([self.nz_abc,self.nx_abc])
        self.future     = np.zeros([self.nz_abc,self.nx_abc])

        self.seismogram = np.zeros([self.nt,self.Nrec])
        print(f"info: Wavefields initialized: {self.nx}x{self.nz}x{self.nt}")

        #create or import velocity model
        if (self.vpFile==None):
            self.createLayerdVelocityModel(2000,3000)
        else:
            self.vp = self.ImportModel(self.vpFile)
        
        if self.approximation == "acousticVTI":
            self.Qc = np.zeros([self.nz_abc,self.nx_abc])
            self.Qp = np.zeros([self.nz_abc,self.nx_abc])
            self.Qf = np.zeros([self.nz_abc,self.nx_abc])
            # Initialize epsilon and delta models
            self.epsilon = np.zeros([self.nz,self.nx])
            self.delta = np.zeros([self.nz,self.nx])

            #import epsilon and delta model
            if (self.epsilonFile == None):
                self.createLayerdEpsilonModel()
            else:
                self.epsilon = self.ImportModel(self.epsilonFile)

            if (self.deltaFile == None):
                self.createLayerdDeltaModel()
            else:
                self.delta = self.ImportModel(self.deltaFile)
        
    def createLayerdVelocityModel(self,v1=3000, v2=4000):
        self.vp[0:self.nz//2, :] = v1
        self.vp[self.nz//2:self.nz, :] = v2

    def createLayerdEpsilonModel(self,e1=0, e2=0.24):
        self.epsilon[0:self.nz//2, :] = e1
        self.epsilon[self.nz//2:self.nz, :] = e2

    def createLayerdDeltaModel(self, d1=0, d2=0.1):
        self.delta[0:self.nz//2, :] = d1
        self.delta[self.nz//2:self.nz, :] = d2
  
    def viewModel(self, model, title):
        plt.figure(figsize=(10, 5))
        plt.imshow(model, aspect='auto', cmap='jet', extent=[0, self.L, self.D, 0])
        plt.plot(self.rec_x, self.rec_z, 'bv', markersize=2, label='Receivers')
        plt.plot(self.shot_x, self.shot_z, 'r*', markersize=5, label='Sources')
        plt.legend()
        plt.colorbar(label='Velocity (m/s)')
        plt.title(title)
        plt.xlabel("Distance (m)")
        plt.ylabel("Depth (m)")
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{title}.png")

    def viewAllModels(self):

        self.viewModel(self.vp, "Velocity Model")

        if self.approximation == "acousticVTI":
            self.viewModel(self.epsilon, "Epsilon Model")
            self.viewModel(self.delta, "Delta Model")


    def viewSnapshot(self, k=0):
        plt.figure(figsize=(10, 5))
        plt.imshow(self.snapshot[k], aspect='auto', cmap='gray', extent=[0, self.L, self.D, 0])
        plt.plot(self.rec_x, self.rec_z, 'bv', markersize=2, label='Receivers')
        plt.plot(self.shot_x, self.shot_z, 'r*', markersize=5, label='Sources')
        plt.legend()
        plt.title(f"Snapshot at time step {k}")
        plt.colorbar(label='Amplitude')
        plt.xlabel("Distance (m)")
        plt.ylabel("Depth (m)")
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.savefig(f"snapshot_{k}.png")

    def viewSeismogram(self,perc=99):
        plt.figure(figsize=(5, 5))
        perc = np.percentile(self.seismogram, perc)
        plt.imshow(self.seismogram, aspect='auto', cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.Nrec, 0, self.T])
        plt.colorbar(label='Amplitude')
        plt.title("Seismogram")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.savefig("seismogram.png")


    def checkDispersionAndStability(self):
        if self.approximation == "acoustic":
            vp_min = np.min(self.vp)
            vp_max = np.max(self.vp)
            lambda_min = vp_min / self.fcut
            dh = np.min([self.dx, self.dz])
            dx_lim = lambda_min / 5
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
        
        elif self.approximation == "acousticVTI":
            vp_min = np.min(self.vp)
            vpx = self.vp*np.sqrt(1+2*self.epsilon)
            vpx_max = np.max(vpx)
            lambda_min = vp_min / self.fcut
            dx_lim = lambda_min / 5
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

    def createCerjanLayers(self):
        N = self.N_abc
        nx = self.nx_abc
        nz = self.nz_abc
        A = np.ones([self.nz_abc, self.nx_abc])
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
        plt.figure()
        plt.imshow(A, cmap='jet', aspect='auto')
        plt.title("Cerjan Absorbing Layer")
        plt.savefig("cerjan_layer.png")
        print(f"info: Cerjan absorbing layers")
        return A
    
    def solveAcousticWaveEquation(self):
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.A = self.createCerjanLayers()

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.past.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

            rx = np.int32(self.rec_x/self.dx) + self.N_abc
            rz = np.int32(self.rec_z/self.dz) + self.N_abc

            for k in range(self.nt):
                self.current[sz,sx] =+ self.source[k]
                self.future = updateWaveEquation(self.future, self.current, self.past, self.vp_exp, self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)
                
                # Apply absorbing boundary condition
                self.future *= self.A
                self.past = np.copy(self.current*self.A)
                self.current = np.copy(self.future)

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                if k == self.frame:
                    self.snapshot.append(self.current.copy())
                if k == self.frame+200:
                    self.snapshot.append(self.current.copy())

            seismogramFile = f"{self.seismogramFolder}seismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(seismogramFile)
            print(f"info: Seismogram saved to {seismogramFile}")

    def solveAcousticVTIWaveEquation(self):
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.A = self.createCerjanLayers()

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.past.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.Qc.fill(0)
            self.Qp.fill(0)
            self.Qf.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

            rx = np.int32(self.rec_x/self.dx) + self.N_abc
            rz = np.int32(self.rec_z/self.dz) + self.N_abc

            for k in range(self.nt):
                self.current[sz,sx] =+ self.source[k]
                self.Qc[sz,sx] =+ self.source[k]
                self.future,self.Qf = updateWaveEquationVTI(self.future, self.current, self.past, self.Qc, self.Qp, self.Qf, self.nx_abc, self.nz_abc, self.dt, self.dx, self.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            
                # Apply absorbing boundary condition
                self.future *= self.A
                self.Qf *= self.A
                self.past = np.copy(self.current*self.A) 
                self.Qp = np.copy(self.Qc*self.A) 
                self.current = np.copy(self.future)
                self.Qc = np.copy(self.Qf)
            
                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                if k == self.frame:
                    self.snapshot.append(self.current.copy())
                if k == self.frame+200:
                    self.snapshot.append(self.current.copy())

            seismogramFile = f"{self.seismogramFolder}seismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(seismogramFile)
            print(f"info: Seismogram saved to {seismogramFile}")

