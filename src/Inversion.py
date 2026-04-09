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
            # plt.imshow(self.residual, aspect='auto',label = "residual" )
            # plt.figure()
            # plt.imshow(self.wf.seismogram,aspect='auto',label = "dcal")
            # plt.figure()
            # plt.imshow(dobs,aspect='auto',label = "dobs")
            # plt.show()
            X += 0.5 * np.sum(self.residual * self.residual)*self.pmt.dt
        return X

    def objective_functionGPU(self, m):
        X = 0.0
        self.wf.vp = m      
        self.wf.source = cp.asarray(self.wf.source, dtype=cp.float32)
        self.wf.vp_exp = self.wf.ExpandModel(self.wf.vp)
        self.wf.vp_exp = cp.asarray(self.wf.vp_exp, dtype=cp.float32)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
            self.wf.A = cp.asarray(self.wf.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.wf.delta)
            self.wf.epsilon_exp  = cp.asarray(self.wf.epsilon_exp, dtype=cp.float32)
            self.wf.delta_exp  = cp.asarray(self.wf.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                self.wf.theta_exp = self.wf.ExpandModel(self.wf.theta)
                self.wf.theta_exp  = cp.asarray(self.wf.theta_exp, dtype=cp.float32)
        
        rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        rx = cp.asarray(rx)
        rz = cp.asarray(rz)
        for shot in range(self.pmt.Nshot):
            dobs = self.loadObsSeismogram(shot)
            self.wf.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.wf.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.wf.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc           
            for k in range(self.pmt.nt): 
                self.wf.forward_stepGPU(k)
                # Register seismogram and snapshot
                self.wf.store_seismogram(k,rz,rx)      
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
           
            self.seismogram = cp.asnumpy(self.wf.seismogram_gpu)
            self.residual = self.seismogram - dobs
            self.save_residual(shot,self.residual)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(self.residual, aspect='auto',label = "residual" )
            # plt.figure()
            # plt.imshow(self.seismogram,aspect='auto',label = "dcal")
            # plt.figure()
            # plt.imshow(dobs,aspect='auto',label = "dobs")
            # plt.show()
            X += 0.5 * np.sum(self.residual * self.residual)*self.pmt.dt
        return X

    def calculate_gradient(self, m):
        grad = np.zeros_like(self.wf.vp)
        self.wf.vp = m
        self.mig.SolveBackwardWaveEquation()
        grad = self.loadGradient()
        return grad
    
    def stepsearch(self, m, p):
        vmax = np.max(np.abs(m))

        alpha0 = 0.0
        alpha1 = 0.001 * vmax * vmax
        alpha2 = 0.005 * vmax * vmax

        m0 = m + alpha0 * p
        m1 = m + alpha1 * p
        m2 = m + alpha2 * p

        if self.pmt.unit == "CPU":
            X0 = self.objective_function(m0)
            X1 = self.objective_function(m1)
            X2 = self.objective_function(m2)
        else:
            X0 = self.objective_functionGPU(m0)
            X1 = self.objective_functionGPU(m1)
            X2 = self.objective_functionGPU(m2)

        print("X0 =", X0)
        print("X1 =", X1)
        print("X2 =", X2)

        if (not np.isfinite(X0)) or (not np.isfinite(X1)) or (not np.isfinite(X2)):
            print("warning: funcao objetivo invalida")
            return alpha1

        if not (X1 <= X0 and X1 <= X2):
            print("warning: pontos nao sao validos para interpolacao parabolica")
            return alpha1

        num = (alpha1*alpha1 - alpha2*alpha2)*X0 + (alpha2*alpha2 - alpha0*alpha0)*X1 + (alpha0*alpha0 - alpha1*alpha1)*X2
        den = (alpha1 - alpha2)*X0 + (alpha2 - alpha0)*X1 + (alpha0 - alpha1)*X2

        print("num =", num)
        print("den =", den)

        if den == 0.0 or not np.isfinite(den):
            print("warning: denominador invalido")
            return alpha1

        alpha_new = 0.5 * (num / den)

        if (not np.isfinite(alpha_new)) or (alpha_new <= alpha0) or (alpha_new >= alpha2):
            print("warning: alpha_new fora da faixa, usando alpha1")
            return alpha1

        return alpha_new
    
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
        # self.m0 = smooth_model(self.wf.vp, self.pmt.sigma).copy()
        m = self.wf.vp
        final_model_file = (f"{self.pmt.modelFolder}fwi_vp_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        m.astype(np.float32).tofile(final_model_file)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(m)
        # plt.show()

        for itr in range(self.pmt.niter):
            print(f"\033[31minfo: FWI iteration {itr + 1}/{self.pmt.niter}\033[0m")

            # Gradiente e função objetivo no modelo atual
            if self.pmt.unit == "CPU":
                self.objective_function(m)
            else:
                self.objective_functionGPU(m)
            
            g = self.calculate_gradient(m)
            g = g/np.max(np.abs(g))
            
            # Salvar gradiente da iteração atual
            gradient_file = (f"{self.pmt.migratedimageFolder}gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
            g.astype(np.float32).tofile(gradient_file)
            print(f"info: Gradient saved to {gradient_file}")

            # Direção de busca: steepest descent
            p = -g

            # Line search
            alpha = self.stepsearch(m,p)
            print("alpha=",alpha)

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



