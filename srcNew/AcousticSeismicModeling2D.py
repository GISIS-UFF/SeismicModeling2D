import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import ricker
from utils import updateWaveEquation

class wavefield:
    approximation = "acoustic" # or "acousticVTI"

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
        self.N_abc = 100

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
        self.fcut = 60 

        self.seismogramFolder = "../outputs/seismograms/"

        # Source and receiver files
        self.rec_file = "../inputs/receivers.csv"
        self.src_file = "../inputs/sources.csv"

        # Velocity model file
        self.vpFile =  "../inputs/marmousi_vp_383x141.bin"

        # Snapshot flag
        self.snap       = False
        self.snapshot = []
        self.frame      = 500
        self.folderSnapshot = "../outputs/snapshots/"



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
        # plt.savefig("source_wavelet.png")

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
            self.createLayerdVelocityModel()
        else:
            self.importVelocityModel(self.vpFile)

    def createLayerdVelocityModel(self,v1=3000, v2=4000):
        self.vp[0:self.nz//2, :] = v1
        self.vp[self.nz//2:self.nz, :] = v2
    
    def importVelocityModel(self, filename):
        self.vp = np.fromfile(filename, dtype=np.float32).reshape(self.nx,self.nz)
        self.vp = self.vp.T
        print(f"info: Imported: {filename}")

    def viewVelocityModel(self):
        plt.figure(figsize=(10, 5))
        plt.imshow(self.vp, aspect='auto', cmap='jet', extent=[0, self.L, self.D, 0])
        plt.plot(self.rec_x, self.rec_z, 'bv', markersize=2, label='Receivers')
        plt.plot(self.shot_x, self.shot_z, 'r*', markersize=5, label='Sources')
        plt.legend()
        plt.colorbar(label='Velocity (m/s)')
        plt.title("Velocity Model")
        plt.xlabel("Distance (m)")
        plt.ylabel("Depth (m)")
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.savefig("velocity_model.png")

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
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.savefig("seismogram.png")


    def checkDispersionAndStability(self):
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


    def expandVelocityModel(self):
        nz = self.nz
        nx = self.nx
        N = self.N_abc
        nz_abc = self.nz_abc
        nx_abc = self.nx_abc
        
        self.vp_exp = np.zeros((nz_abc, nx_abc))
        self.vp_exp[N:nz_abc-N, N:nx_abc-N] = self.vp
        self.vp_exp[0:N, N:nx_abc-N] = self.vp[0, :]
        self.vp_exp[nz_abc-N:nz_abc, N:nx_abc-N] = self.vp[-1, :]
        self.vp_exp[N:nz_abc-N, 0:N] = self.vp[:, 0:1]
        self.vp_exp[N:nz_abc-N, nx_abc-N:nx_abc] = self.vp[:, -1:]
        self.vp_exp[0:N, 0:N] = self.vp[0, 0]
        self.vp_exp[0:N, nx_abc-N:nx_abc] = self.vp[0, -1]
        self.vp_exp[nz_abc-N:nz_abc, 0:N] = self.vp[-1, 0]
        self.vp_exp[nz_abc-N:nz_abc, nx_abc-N:nx_abc] = self.vp[-1, -1]
        print(f"info: Velocity model expanded to {nz_abc}x{nx_abc}")
        plt.figure()
        plt.imshow(self.vp_exp, cmap='jet', aspect='auto')
        plt.savefig("expanded_velocity_model.png")
    
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

    def solveWaveEquation(self):
        
        self.expandVelocityModel()
        self.A = self.createCerjanLayers()

        for shot in range(self.Nshot):
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
                self.future = updateWaveEquation(self.future, self.current, self.past, self.vp_exp, self.nz_abc,self.nx_abc, self.dz, self.dx, self.dt)
                
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
