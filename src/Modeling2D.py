import numpy as np
import time

from utils import ricker
from utils import updateWaveEquation
from utils import updateWaveEquationCPML
from utils import updateWaveEquationVTI
from utils import updateWaveEquationVTICPML
from utils import updateWaveEquationTTI
from utils import AbsorbingBoundary
from utils import updatePsi
from utils import updateZeta

class wavefield: 
    def __init__(self, parameters):
        self.pmt = parameters

    def createSourceWavelet(self):
        # Create Ricker wavelet
        self.source = ricker(self.pmt.fcut, self.pmt.t)
        self.source = self.source * (self.pmt.dt * self.pmt.dt)/(self.pmt.dx*self.pmt.dz)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.pmt.t, self.source)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

        fft_result = np.fft.rfft(self.source)
        amplitude = np.abs(fft_result)
        freq = np.fft.rfftfreq(len(self.source), d=self.pmt.dt)

        plt.figure()
        plt.plot(freq, amplitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.show()
        print(f"info: Ricker Source wavelet created: {self.pmt.nt} samples")
        
    def ImportModel(self, filename):
        data = np.fromfile(filename, dtype=np.float32).reshape(self.pmt.nx, self.pmt.nz)
        print(f"info: Imported: {filename}")
        return data.T
        
    def ExpandModel(self, model_data):
        N = self.pmt.N_abc
        nz_abc, nx_abc = self.pmt.nz_abc, self.pmt.nx_abc
        
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

        return model_exp
    
    def initializeWavefields(self):
        # Initialize velocity model and wavefields
        self.vp         = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)
        self.current    = np.zeros([self.pmt.nz_abc,self.pmt.nx_abc],dtype=np.float32)
        self.future     = np.zeros([self.pmt.nz_abc,self.pmt.nx_abc],dtype=np.float32)
        self.seismogram = np.zeros([self.pmt.nt,self.pmt.Nrec],dtype=np.float32)
        if self.pmt.approximation in ["VTI", "TTI"]:
            # Initialize epsilon and delta models
            self.epsilon = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)
            self.delta = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)
        if self.pmt.approximation == "TTI":
            # Initialize vs and theta model
            self.vs = np.zeros([self.pmt.nz,self.pmt.nx], dtype=np.float32)
            self.theta = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)
        if self.pmt.ABC == "CPML":
            # Initialize absorbing layers       
            self.PsixFR      = np.zeros([self.pmt.nz_abc, self.pmt.N_abc+4], dtype=np.float32)
            self.PsixFL      = np.zeros([self.pmt.nz_abc, self.pmt.N_abc+4], dtype=np.float32)     
            self.PsizFU      = np.zeros([self.pmt.N_abc+4, self.pmt.nx_abc], dtype=np.float32) 
            self.PsizFD      = np.zeros([self.pmt.N_abc+4, self.pmt.nx_abc], dtype=np.float32)       
            self.ZetaxFR     = np.zeros([self.pmt.nz_abc, self.pmt.N_abc+4], dtype=np.float32)
            self.ZetaxFL     = np.zeros([self.pmt.nz_abc, self.pmt.N_abc+4], dtype=np.float32)
            self.ZetazFU     = np.zeros([self.pmt.N_abc+4, self.pmt.nx_abc], dtype=np.float32)
            self.ZetazFD     = np.zeros([self.pmt.N_abc+4, self.pmt.nx_abc], dtype=np.float32)
        if self.pmt.migration in ["checkpoint", "SB", "RBC", "onthefly"]:
            self.migrated_image = np.zeros((self.pmt.nz, self.pmt.nx), dtype=np.float32)
            self.currentbck  = np.zeros([self.pmt.nz_abc,self.pmt.nx_abc],dtype=np.float32)
            self.futurebck   = np.zeros([self.pmt.nz_abc,self.pmt.nx_abc],dtype=np.float32)
            if self.pmt.migration == "SB":
                self.top   = np.zeros((self.pmt.nt, 4, self.pmt.nx), dtype=np.float32)
                self.bot   = np.zeros((self.pmt.nt, 4, self.pmt.nx), dtype=np.float32)
                self.left  = np.zeros((self.pmt.nt, self.pmt.nz, 4), dtype=np.float32)
                self.right = np.zeros((self.pmt.nt, self.pmt.nz, 4), dtype=np.float32)

        print(f"info: Wavefields initialized: {self.pmt.nx}x{self.pmt.nz}x{self.pmt.nt}")
    
    def loadModels(self):
        self.vp = self.ImportModel(self.pmt.vpFile)
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon = self.ImportModel(self.pmt.epsilonFile)
            self.delta = self.ImportModel(self.pmt.deltaFile)
        if self.pmt.approximation == "TTI":
            self.vs = self.ImportModel(self.pmt.vsFile)
            self.theta = self.ImportModel(self.pmt.thetaFile)
            self.theta = np.radians(self.theta)
        
        print(f"info: Models loaded: {self.pmt.nx}x{self.pmt.nz}")
        
    def Reflectioncoefficient(self):
        borda_ref = 10
        R_ref = 1e-3
        R = R_ref ** (self.pmt.N_abc/borda_ref)

        if self.pmt.N_abc >= 200:
            R = R_ref ** (150/borda_ref)

        return R

    def dampening_const(self):     
        M = 2
        Rcoef = self.Reflectioncoefficient()
        f_pico = self.pmt.fcut/3
        d0 = - (M + 1)* np.log(Rcoef) 

        return d0, f_pico

    def checkDispersionAndStability(self):
        if self.pmt.approximation == "acoustic":
            vp_min = np.min(self.vp)
            vp_max = np.max(self.vp)
            lambda_min = vp_min / self.pmt.fcut
            dx_lim = lambda_min / 4.28
            dt_lim = dx_lim / (4 * vp_max)
            print(f"info: Dispersion and stability check")
            print(f"info: Minimum velocity: {vp_min:.2f} m/s")
            print(f"info: Maximum velocity: {vp_max:.2f} m/s")
            print(f"info: Maximum frequency: {self.pmt.fcut:.2f} Hz")
            print(f"info: Current dx: {self.pmt.dx:.2f} m")
            print(f"info: Current dt: {self.pmt.dt:.5f} s")
            print(f"info: Critical dx: {dx_lim:.2f} m")
            print(f"info: Critical dt: {dt_lim:.5f} s")
            if self.pmt.dx <= dx_lim and self.pmt.dt <= dt_lim:
                print("info: Dispersion and stability conditions satisfied.")
            else:
                print("WARNING: Dispersion or stability conditions not satisfied.")
        
        elif self.pmt.approximation in ["VTI", "TTI"]:
            vp_min = np.min(self.vp)
            vpx = self.vp*np.sqrt(1+2*self.epsilon)
            vpx_max = np.max(vpx)
            lambda_min = vp_min / self.pmt.fcut
            dx_lim = lambda_min / 4.28
            dt_lim = dx_lim / (4 * vpx_max)
            print(f"info: Dispersion and stability check")
            print(f"info: Minimum velocity: {vp_min:.2f} m/s")
            print(f"info: Maximum velocity: {vpx_max:.2f} m/s")
            print(f"info: Maximum frequency: {self.pmt.fcut:.2f} Hz")
            print(f"info: Current dx: {self.pmt.dx:.2f} m")
            print(f"info: Current dt: {self.pmt.dt:.5f} s")
            print(f"info: Critical dx: {dx_lim:.2f} m")
            print(f"info: Critical dt: {dt_lim:.5f} s")
            if self.pmt.dx <= dx_lim and self.pmt.dt <= dt_lim:
                print("info: Dispersion and stability conditions satisfied.")
            else:
                print("WARNING: Dispersion or stability conditions not satisfied.")
    
    def createCerjanVector(self):
        sb = 4. * self.pmt.N_abc
        A = np.ones(self.pmt.N_abc)
        for i in range(self.pmt.N_abc):
                fb = (self.pmt.N_abc - i) / (np.sqrt(2.) * sb)
                A[i] = np.exp(-fb * fb)
                
        return A 
    
    def save_snapshot(self,shot, k):        
        if not self.pmt.snap:
            return
        if k > self.pmt.last_save:
            return
        if k % self.pmt.step != 0:
            return

        snapshot = self.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]

        snapshotFile = (f"{self.pmt.snapshotFolder}{self.pmt.approximation}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")
        snapshot.tofile(snapshotFile)
        print(f"info: Snapshot saved to {snapshotFile}")
    
    def save_seismogram(self,shot):        
        self.seismogramFile = f"{self.pmt.seismogramFolder}seismogram_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        self.seismogram.tofile(self.seismogramFile)
        print(f"info: Seismogram saved to {self.seismogramFile}")

    def solveAcousticWaveEquation(self):
        start_time = time.time()
        print(f"info: Solving acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp_exp = self.ExpandModel(self.vp)
        self.A = self.createCerjanVector()
        
        rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc

        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc           

            for k in range(self.pmt.nt):        
                self.future = updateWaveEquation(self.future, self.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
                self.future[sz,sx] += self.source[k]

                # Apply absorbing boundary condition
                self.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.future, self.A)
                self.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.current, self.A)

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

        rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc

        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
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
            sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc

            for k in range(self.pmt.nt):
                self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD = updatePsi(self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.current, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD = updateZeta(self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.current, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.future = updateWaveEquationCPML(self.future, self.current, self.vp_exp, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt, self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD, self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD, self.pmt.N_abc)
                
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

        rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc

        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc 

            for k in range(self.pmt.nt):
                self.future= updateWaveEquationVTI(self.future, self.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                self.future[sz,sx] += self.source[k]
                # Apply absorbing boundary condition
                self.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.future, self.A)
                self.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.current, self.A)

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

        rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc

        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
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
            sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc 

            for k in range(self.pmt.nt):
                self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD = updatePsi(self.PsixFR, self.PsixFL,self.PsizFU, self.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.current, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD = updateZeta(self.PsixFR, self.PsixFL, self.ZetaxFR, self.ZetaxFL,self.PsizFU, self.PsizFD, self.ZetazFU, self.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.current, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.future = updateWaveEquationVTICPML(self.future, self.current, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.pmt.nx_abc, self.pmt.nz_abc, self.PsixFR, self.PsixFL, self.PsizFU, self.PsizFD, self.ZetaxFR, self.ZetaxFL, self.ZetazFU, self.ZetazFD, self.pmt.N_abc)
                
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

        rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc

        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.current.fill(0)
            self.future.fill(0)
            self.seismogram.fill(0)

            # convert acquisition geometry coordinates to grid points
            sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc            

            for k in range(self.pmt.nt):
                self.future= updateWaveEquationTTI(self.future, self.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
                self.future[sz,sx] += self.source[k]
                # Apply absorbing boundary condition
                self.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.future, self.A)
                self.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.current, self.A)

                # Register seismogram
                self.seismogram[k, :] = self.current[rz, rx]
                self.save_snapshot(shot, k)
                
                #swap
                self.current, self.future = self.future, self.current


            self.save_seismogram(shot)
            print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")

    def SolveWaveEquation(self):
        if self.pmt.approximation == "acoustic" and self.pmt.ABC == "cerjan":
            self.solveAcousticWaveEquation()
        elif self.pmt.approximation == "acoustic" and self.pmt.ABC == "CPML":
            self.solveAcousticWaveEquationCPML()
        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "cerjan":
            self.solveAcousticVTIWaveEquation()
        elif self.pmt.approximation == "VTI" and self.pmt.ABC == "CPML":
            self.solveAcousticVTIWaveEquationCPML()
        elif self.pmt.approximation == "TTI" and self.pmt.ABC == "cerjan":
            self.solveAcousticTTIWaveEquation()
        else:
            raise ValueError("ERROR: Unknown approximation. Choose 'acoustic', 'VTI' or 'TTI'. Otherwise, unknown Absorbing Boundary Condition. Choose 'cerjan' or 'CPML'.")
        print(f"info: Wave equation solved")
