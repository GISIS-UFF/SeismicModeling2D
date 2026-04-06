import numpy as np
import time
import cupy as cp
from utils import smooth_model

class fwi:
    def __init__(self,parameters,wavefield,migration):
        self.pmt = parameters
        self.wf = wavefield
        self.mig = migration

    def objective_function(self, m):
        X = 0.0
        self.wf.vp = m 
        self.wf.vp_exp = self.ExpandModel(self.wf.vp)
        if self.pmt.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.delta)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.theta)
        
        rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        for shot in range(self.pmt.Nshot):
            dobs = self.loadObsSeismogram(shot)
            self.wf.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.wf.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.wf.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc           
            for k in range(self.pmt.nt): 
                self.wf.forward_step(k)
                # Register seismogram and snapshot
                self.wf.store_seismogram(k,rz,rx)      
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current

                residual = self.wf.seismogram - dobs
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow(residual, aspect='auto',label = "residual" )
                # plt.figure()
                # plt.imshow(dcal,aspect='auto',label = "dcal")
                # plt.figure()
                # plt.imshow(dobs,aspect='auto',label = "dobs")
                # plt.show()
                X += 0.5 * (residual)*(residual)
        return X

    def calculate_gradient(self, m):
        grad = np.zeros_like(self.wf.vp)
        rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        self.wf.vp = m
        self.wf.solveWaveEquation() 
        self.mig.SolveBackwardWaveEquation()
        grad = self.loadGradient()
        grad = grad / np.max(np.abs(grad))
        return grad

    def quadratic_interpolation(self):


        return 

    def loadGradient(self):
        gradientFile = f"{self.pmt.migratedimageFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad
    
    def loadObsSeismogram(self,shot):
        seismogramFile = f"{self.pmt.seismogramFolder}seismogram_obs_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        seismogram = np.fromfile(seismogramFile, dtype=np.float32).reshape(self.pmt.nt,self.pmt.Nrec) 
        return seismogram

    def solveFullWaveformInversion(self):
        start_time = time.time()
        print("info: Solving Full Waveform Inversion")

        # Modelo inicial
        m = smooth_model(self.wf.vp, self.pmt.sigma).copy()
        final_model_file = (f"{self.pmt.modelFolder}fwi_vp_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        m.astype(np.float32).tofile(final_model_file)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(m)
        plt.show()

        history = []

        for itr in range(self.pmt.niter):
            print(f"info: FWI iteration {itr + 1}/{self.pmt.niter}")

            # Gradiente e função objetivo no modelo atual
            fx = self.objective_function(m)
            g = self.calculate_gradient(m)
            
            # Salvar gradiente da iteração atual
            gradient_file = (f"{self.pmt.migratedimageFolder}gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
            g.astype(np.float32).tofile(gradient_file)
            print(f"info: Gradient saved to {gradient_file}")

            # Direção de busca: steepest descent
            p = -g

            # Line search
            alpha = self.linesearch(m, p, g, fx)

            # Atualização do modelo
            m = m + alpha * p

            # Salvar histórico
            history.append((itr + 1, fx, g, alpha))

        # Atualiza modelo final
        self.wf.vp = m.copy()

        # Salvar modelo final
        final_model_file = (f"{self.pmt.modelFolder}fwi_vp_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.wf.vp.astype(np.float32).tofile(final_model_file)
        print(f"info: Final model saved to {final_model_file}")

        end_time = time.time()
        print(f"\ninfo: FWI finished in {end_time - start_time:.2f} s")



