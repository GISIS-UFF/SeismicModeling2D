import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # nice colorbar
import pandas as pd
import json
import time

from utils import ricker
from utils import updateWaveEquation
from utils import updateWaveEquationVTI
from utils import updateWaveEquationTTI
from utils import AnalyticalModel
from utils import AbsorbingBoundary

class wavefield: 

    def __init__(self, parameters_path = "../inputs/parametersMarmousi.json"):
        self.parameters_path = parameters_path
        self.readParameters()
        self.readAcquisitionGeometry()

    def readParameters(self):
        with open(self.parameters_path) as f:
            parameters = json.load(f)

        # Approximation type
        self.approximation = parameters["approximation"]
        
        # Discretization parameters
        self.dx   = parameters["dx"]
        self.dz   = parameters["dz"]
        self.dt   = parameters["dt"]
        
        # Model size
        self.L    = parameters["L"]
        self.D    = parameters["D"]
        self.T    = parameters["T"]

        # Number of point for absorbing boundary condition
        self.N_abc = parameters["N_abc"]

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
        self.fcut = parameters["fcut"]

        self.seismogramFolder = parameters["seismogramFolder"]

        # Source and receiver files
        self.rec_file = parameters["rec_file"]
        self.src_file = parameters["src_file"]

        # Velocity model file
        self.vpFile = parameters["vpFile"]
        self.vsFile = parameters["vsFile"]
        self.thetaFile = parameters["thetaFile"]

        # Snapshot flag
        self.snap       = parameters["snap"]
        self.snapshot = []
        self.frame      = parameters["frame"] # time steps to save snapshots
        self.folderSnapshot = parameters["folderSnapshot"]

        # Anisotropy parameters files
        self.epsilonFile = parameters["epsilonFile"]  
        self.deltaFile   = parameters["deltaFile"]  

        #Anisotropy parameters for Layered model
        self.vpLayer1 = parameters["vpLayer1"]
        self.vpLayer2 = parameters["vpLayer2"]
        self.vsLayer1 = parameters["vsLayer1"]
        self.vsLayer2 = parameters["vsLayer2"]
        self.thetaLayer1 = parameters["thetaLayer1"]
        self.thetaLayer2 = parameters["thetaLayer2"]
        self.epsilonLayer1 = parameters["epsilonLayer1"]
        self.epsilonLayer2 = parameters["epsilonLayer2"]
        self.deltaLayer1   = parameters["deltaLayer1"]
        self.deltaLayer2  = parameters["deltaLayer2"]

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
        plt.savefig("source_wavelet.png")
        # plt.show()
        
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
        print(f"info: Wavefields initialized: {self.nx}x{self.nz}x{self.nt}")

        #create or import velocity model
        if (self.vpFile==None):
            self.createLayerdVpModel(self.vpLayer1,self.vpLayer2)
        else:
            self.vp = self.ImportModel(self.vpFile)
        
        if self.approximation == "acousticVTI":
            # Initialize velocity model and wavefields
            self.Qc = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
            self.Qf = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)

            # Initialize epsilon and delta models
            self.epsilon = np.zeros([self.nz,self.nx],dtype=np.float32)
            self.delta = np.zeros([self.nz,self.nx],dtype=np.float32)

            #import epsilon and delta model
            if (self.epsilonFile == None):
                self.createLayerdEpsilonModel(self.epsilonLayer1,self.epsilonLayer2)
            else:
                self.epsilon = self.ImportModel(self.epsilonFile)

            if (self.deltaFile == None):
                self.createLayerdDeltaModel(self.deltaLayer1,self.deltaLayer2)
            else:
                self.delta = self.ImportModel(self.deltaFile)
                
        if self.approximation == "acousticTTI":
            # Initialize velocity model and wavefields
            self.vs = np.zeros([self.nz,self.nx],dtype=np.float32)
            self.theta = np.zeros([self.nz,self.nx],dtype=np.float32)
            self.Qc = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
            self.Qf = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)

            # Initialize epsilon and delta models
            self.epsilon = np.zeros([self.nz,self.nx],dtype=np.float32)
            self.delta = np.zeros([self.nz,self.nx],dtype=np.float32)

            #import vs and theta model
            if (self.vsFile == None):
                self.createLayerdVsModel(self.vsLayer1,self.vsLayer2)
            else:
                self.vs = self.ImportModel(self.vsFile)

            if (self.thetaFile == None):
                self.createLayerdThetaModel(self.thetaLayer1,self.thetaLayer2)
            else:
                self.theta = self.ImportModel(self.thetaFile)

            #import epsilon and delta model
            if (self.epsilonFile == None):
                self.createLayerdEpsilonModel(self.epsilonLayer1,self.epsilonLayer2)
            else:
                self.epsilon = self.ImportModel(self.epsilonFile)

            if (self.deltaFile == None):
                self.createLayerdDeltaModel(self.deltaLayer1,self.deltaLayer2)
            else:
                self.delta = self.ImportModel(self.deltaFile)
        
    def createLayerdVpModel(self,v1=3000, v2=4000):
        self.vp[0:self.nz//2, :] = v1
        self.vp[self.nz//2:self.nz, :] = v2
    
    def createLayerdVsModel(self,v1=0, v2=0):
        self.vs[0:self.nz//2, :] = v1
        self.vs[self.nz//2:self.nz, :] = v2
    
    def createLayerdThetaModel(self, t1=0, t2=0):
        self.theta[0:self.nz//2, :] = t1
        self.theta[self.nz//2:self.nz, :] = t2

    def createLayerdEpsilonModel(self,e1=0.2, e2=0.2):
        self.epsilon[0:self.nz//7, :] = e1
        self.epsilon[self.nz//7:self.nz, :] = e2

    def createLayerdDeltaModel(self, d1=0.2, d2=0.2):
        self.delta[0:self.nz//7, :] = d1
        self.delta[self.nz//7:self.nz, :] = d2

    def createVTIModelFromVp(self):
        if not (self.approximation == "acousticVTI"):
            raise ValueError("ERROR: Change approximation parameter to 'acousticVTI' .")
        
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
        self.viewModel(self.rho, "Density Model")

        # create epsilon model epsilon = 0.25 rho - 0.3 - Petrov et al. (2021) 
        self.epsilon = np.zeros([self.nz,self.nx],dtype=np.float32)
        self.epsilon = 0.25 * self.rho/1000 - 0.3 # rho in g/cm3
        self.epsilon[idx_water] = 0.0 # water epsilon
        self.viewModel(self.epsilon, "Epsilon Model")
        self.epsilon.T.tofile(self.vpFile.replace(".bin","_epsilon.bin"))	
        print(f"info: Epsilon model saved to {self.vpFile.replace('.bin','_epsilon.bin')}")


        # create delta model delta = 0.125 rho - 0.1 - Petrov et al. (2021)
        self.delta = np.zeros([self.nz,self.nx],dtype=np.float32)
        self.delta = 0.125 * self.rho/1000 - 0.1 # rho in g/cm3
        self.delta[idx_water] = 0.0 # water delta
        self.viewModel(self.delta, "Delta Model")
        self.delta.T.tofile(self.vpFile.replace(".bin","_delta.bin"))
        print(f"info: Delta model saved to {self.vpFile.replace('.bin','_delta.bin')}")

        # plt.show()


    def adjustColorBar(self,fig,ax,im):
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append an axes to the right of the current axes, with the same height
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im,cax=cax)
        return cbar
    
    def viewModel(self, model, title):
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(model, aspect='equal', cmap='jet', extent=[0, self.L, self.D, 0])
        ax.plot(self.rec_x, self.rec_z, 'bv', markersize=2, label='Receivers')
        ax.plot(self.shot_x, self.shot_z, 'r*', markersize=5, label='Sources')
        ax.set_title(title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True)
        
        # nice colorbar
        cbar = self.adjustColorBar(fig,ax,im)
        if title == "Velocity Model": units = " (m/s)"
        else: units = ""
        cbar.set_label(title+units)
        
        # ax.legend()
        plt.tight_layout()
        plt.savefig(f"{title}.png")
        # plt.show()

    def viewAllModels(self):

        self.viewModel(self.vp, "Vp Model")

        if self.approximation == "acousticVTI":
            self.viewModel(self.epsilon, "Epsilon Model")
            self.viewModel(self.delta, "Delta Model")
        
        if self.approximation == "acousticTTI":
            self.viewModel(self.vs, "Vs Model")
            self.viewModel(self.theta, "Theta Model")
            self.viewModel(self.epsilon, "Epsilon Model")
            self.viewModel(self.delta, "Delta Model")

        # plt.show()


    def viewSnapshotAtTime(self, k):
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(self.snapshot[k], aspect='equal', cmap='gray', extent=[0, self.L, self.D, 0])
        ax.plot(self.rec_x, self.rec_z, 'bv', markersize=2, label='Receivers')
        ax.plot(self.shot_x, self.shot_z, 'r*', markersize=5, label='Sources')
        ax.legend()
        ax.set_title(f"Snapshot at time step {self.frame[k]}")
        
        # nice colorbar
        cbar = self.adjustColorBar(fig,ax,im)
        cbar.set_label("Amplitude")
        
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True)
        plt.tight_layout()
        if self.approximation == "acousticVTI":
            plt.savefig(f"snapshotVTI_{k}.png")
        if self.approximation == "acoustic":
            plt.savefig(f"snapshotAcoustic_{k}.png")
       
    def viewSnapshot(self):
        for k in range(len(self.snapshot)):
            self.viewSnapshotAtTime(k)
        print(f"info: {len(self.snapshot)} snapshots saved to {self.folderSnapshot}")
        
        plt.show(block=False)

    def viewSnapshotAnalyticalComparison(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot snapshot
        im = ax.imshow(self.snapshot[0], aspect='equal', cmap='gray', extent=[0, self.L, self.D, 0])
        ax.plot(self.rec_x, self.rec_z, 'bv', markersize=2, label='Receivers')
        ax.plot(self.shot_x, self.shot_z, 'r*', markersize=5, label='Sources')
        
        # Compute the analytical wavefront
        if self.approximation == "acoustic":
            vel = self.vp[ int(self.shot_z[0]/self.dz), int(self.shot_x[0]/self.dx) ]
            Rp = AnalyticalModel(vel, 0, 0, self.dt, self.fcut, self.frame[0])
        elif self.approximation == "acousticVTI":
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

        ax.plot(x_rp, z_rp, 'r', label='Analytical wavefront')
        ax.legend()
        ax.set_title(f"Snapshot at time step {self.frame[0]} (shot {1})")

        # nice colorbar
        cbar = self.adjustColorBar(fig, ax, im)
        cbar.set_label("Amplitude")

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True)
        plt.tight_layout()
        

        if self.approximation == "acoustic":
            plt.savefig(f"{self.folderSnapshot}SnapshotAnalyticalComparison_acoustic_{0}_shot{1}.png")
        if self.approximation == "acousticVTI":
            plt.savefig(f"{self.folderSnapshot}SnapshotAnalyticalComparison_acousticVTI_{0}_shot{1}.png")

        plt.show()

    def viewSeismogramComparison(self, path1, path2, title="Seismogram Difference"):
        seismo1 = np.fromfile(path1, dtype=np.float32).reshape(self.nt, self.Nrec)
        seismo2 = np.fromfile(path2, dtype=np.float32).reshape(self.nt, self.Nrec)

        diff = seismo1 - seismo2

        perc = np.percentile(diff, 99)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(diff, aspect='auto', cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.Nrec, self.T, 0])

        ax.set_title(f"Seismogram Comparison ({title})")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (s)")
        ax.grid(True)

        cbar = self.adjustColorBar(fig, ax, im)
        cbar.set_label(f"Amplitude ({title})")

        plt.tight_layout()
        plt.savefig(f"Comparison_{title}.png")
        plt.show()

    def viewSeismogram(self,perc=99):
        plt.figure(figsize=(5, 5))
        perc = np.percentile(self.seismogram, perc)
        plt.imshow(self.seismogram, aspect='auto', cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.Nrec, self.T, 0])
        plt.colorbar(label='Amplitude')
        plt.title("Seismogram")
        plt.ylabel("Time (s)")
        # plt.legend()
        plt.grid()
        plt.tight_layout()
        if self.approximation == "acoustic":
            plt.savefig("seismogramAcoustic.png")
        if self.approximation == "acousticVTI":
            plt.savefig("seismogramVTI.png")
        # plt.show()

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
    
    def createCerjanVector(self):
        sb = 3 * self.N_abc
        A = np.ones(self.N_abc)
        for i in range(self.N_abc):
                fb = (self.N_abc - i) / (np.sqrt(2) * sb)
                A[i] = np.exp(-fb * fb)
                
        return A
    
    def solveAcousticWaveEquation(self):
        start_time = time.time()
        print(f"info: Solving acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.A = self.createCerjanVector()

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

            rx = np.int32(self.rec_x/self.dx) + self.N_abc
            rz = np.int32(self.rec_z/self.dz) + self.N_abc

            for k in range(self.nt):
                self.current[sz,sx] += self.source[k]
                self.future = updateWaveEquation(self.future, self.current, self.vp_exp, self.nz_abc, self.nx_abc, self.dz, self.dx, self.dt)
                
                # Apply absorbing boundary condition
                self.future = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.future, self.A)
                self.current = AbsorbingBoundary(self.N_abc, self.nz_abc, self.nx_abc, self.current, self.A)

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]

                if k in self.frame:
                    self.snapshot.append(self.current[self.N_abc : self.nz_abc - self.N_abc, self.N_abc : self.nx_abc - self.N_abc].copy())

                #swap
                self.current, self.future = self.future, self.current

            seismogramFile = f"{self.seismogramFolder}AcousticSeismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(seismogramFile)
            print(f"info: Seismogram saved to {seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def solveAcousticVTIWaveEquation(self):
        start_time = time.time()
        print(f"info: Solving acoustic VTI wave equation")
        # Expand models and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.epsilon_exp = self.ExpandModel(self.epsilon)
        self.delta_exp = self.ExpandModel(self.delta)
        self.A = self.createCerjanVector()

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.Qc.fill(0)
            self.Qf.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

            rx = np.int32(self.rec_x/self.dx) + self.N_abc
            rz = np.int32(self.rec_z/self.dz) + self.N_abc

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

                if k in self.frame:
                    self.snapshot.append(self.current[self.N_abc : self.nz_abc - self.N_abc, self.N_abc : self.nx_abc - self.N_abc].copy())

                #swap
                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc

            seismogramFile = f"{self.seismogramFolder}VTIseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(seismogramFile)
            print(f"info: Seismogram saved to {seismogramFile}")
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

        for shot in range(self.Nshot):
            print(f"info: Shot {shot+1} of {self.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)
            self.Qc.fill(0)
            self.Qf.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.shot_x[shot]/self.dx) + self.N_abc
            sz = int(self.shot_z[shot]/self.dz) + self.N_abc            

            rx = np.int32(self.rec_x/self.dx) + self.N_abc
            rz = np.int32(self.rec_z/self.dz) + self.N_abc

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

                if k in self.frame:
                    self.snapshot.append(self.current[self.N_abc : self.nz_abc - self.N_abc, self.N_abc : self.nx_abc - self.N_abc].copy())

                #swap
                self.current, self.future, self.Qc, self.Qf = self.future, self.current, self.Qf, self.Qc

            seismogramFile = f"{self.seismogramFolder}TTIseismogram_shot_{shot+1}_Nt{self.nt}_Nrec{self.Nrec}.bin"
            self.seismogram.tofile(seismogramFile)
            print(f"info: Seismogram saved to {seismogramFile}")
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def SolveWaveEquation(self):
        if self.approximation == "acoustic":
            self.solveAcousticWaveEquation()
        elif self.approximation == "acousticVTI":
            self.solveAcousticVTIWaveEquation()
        elif self.approximation == "acousticTTI":
            self.solveAcousticTTIWaveEquation()
        else:
            raise ValueError("ERROR: Unknown approximation. Choose 'acoustic', 'acousticVTI' or 'acousticTTI'.")
        print(f"info: Wave equation solved")