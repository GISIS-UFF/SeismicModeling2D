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
        self.wf.vp_exp = self.wf.ExpandModel(self.wf.vp)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.wf.theta)
        
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

            self.residual = self.wf.seismogram - dobs
            self.save_residual(shot,self.residual)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(residual, aspect='auto',label = "residual" )
            # plt.figure()
            # plt.imshow(dcal,aspect='auto',label = "dcal")
            # plt.figure()
            # plt.imshow(dobs,aspect='auto',label = "dobs")
            # plt.show()
            X += 0.5 * np.sum(self.residual * self.residual)
        return X

    def calculate_gradient(self, m):
        grad = np.zeros_like(self.wf.vp)
        self.wf.vp = m
        self.mig.SolveBackwardWaveEquation()
        grad = self.loadGradient()
        grad = grad / np.max(np.abs(grad))
        return grad

    def cubic_interpolation(self,m,p, max):
        alpha_initial = 0.0
        X_initial = self.objective_function(m + alpha_initial * p)
        g_initial = self.calculate_gradient(m + alpha_initial * p)
        dX_initial = np.sum(g_initial * p)

        alpha_current = 1.0
        X_current = self.objective_function(m + alpha_current * p)
        g_current = self.calculate_gradient(m + alpha_current * p)
        dX_current = np.sum(g_current * p)

        c1 = 1e-4
        c2 = 0.9
        for _ in range(max):
            d1 = dX_initial + dX_current - 3*(X_initial - X_current)/(alpha_initial - alpha_current)
            d2 = np.sign(alpha_current - alpha_initial) * np.sqrt(d1*d1 - dX_initial*dX_current)
            alpha_new = alpha_current - (alpha_current - alpha_initial)*(dX_current + d2 + d1)/(dX_current - dX_initial + 2.0 *d2)

            X_new = self.objective_function(m + alpha_new * p)
            g_new = self.calculate_gradient(m + alpha_new * p)
            dX_new = np.sum(g_new * p)

            armijo = X_new <= X_initial + c1 * alpha_new * dX_initial
            curvature = abs(dX_new) <= c2 * abs(dX_initial)

            if armijo and curvature:
                return alpha_new

            alpha_initial = alpha_current
            X_initial = X_current
            dX_initial = dX_current

            alpha_current = alpha_new
            X_current = X_new
            dX_current = dX_new

        return alpha_current

    def loadGradient(self):
        gradientFile = f"{self.pmt.migratedimageFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad
    
    def loadObsSeismogram(self,shot):
        seismogramFile = f"{self.pmt.seismogramFolder}seismogram_obs_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        seismogram = np.fromfile(seismogramFile, dtype=np.float32).reshape(self.pmt.nt,self.pmt.Nrec) 
        return seismogram

    def save_residual(self,shot,residual):        
        self.seismogramFile = f"{self.pmt.seismogramFolder}residual_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        self.residual.tofile(self.seismogramFile)
        residual.tofile(self.seismogramFile)
        print(f"info: Residuo saved to {self.seismogramFile}")

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
            alpha = self.cubic_interpolation(m,p, 1)

            # Atualização do modelo
            m = m + alpha * p

        # Atualiza modelo final
        self.wf.vp = m.copy()

        # Salvar modelo final
        final_model_file = (f"{self.pmt.modelFolder}fwi_vp_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.wf.vp.astype(np.float32).tofile(final_model_file)
        print(f"info: Final model saved to {final_model_file}")

        end_time = time.time()
        print(f"\ninfo: FWI finished in {end_time - start_time:.2f} s")



