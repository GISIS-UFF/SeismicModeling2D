import numpy as np
import time
import cupy as cp
from utils import smooth_model

class fwi:
    def __init__(self,parameters,wavefield,migration):
        self.pmt = parameters
        self.wf = wavefield
        self.mig = migration

    def objective_function(self, m, save_residual):
        X = 0.0
        self.wf.vp = 1.0 / np.sqrt(m)      
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
            residual = dobs - self.seismogram
            if save_residual==True:
                self.save_residual(shot,residual) 
            X += 0.5 * np.sum(residual * residual)
        return X

    def calculate_gradient(self, m):
        grad = np.zeros_like(m)
        self.wf.vp = 1.0 / np.sqrt(m)
        self.mig.SolveBackwardWaveEquation()
        grad = self.loadGradient()
        return grad
    
    def two_loop_recursion(self,g,s_store,y_store):
        q = g.copy()
        alpha = np.zeros(len(s_store))
        beta = np.zeros(len(s_store))

        for i in reversed(range(len(s_store))):
            s = s_store[i]
            y = y_store[i]
            sy = np.sum(s * y)
            rho = 1.0 / sy
            alpha[i] = rho * np.sum(s * q)
            q = q - alpha[i] * y

        if len(s_store) > 0:
            s_last = s_store[-1]
            y_last = y_store[-1]

            sy = np.sum(s_last * y_last)
            yy = np.sum(y_last * y_last)

            gamma = sy / yy

        else:
            gamma = 1.0
 
        r = gamma * q

        for i in range(len(s_store)):
            s = s_store[i]
            y = y_store[i]
            sy = np.sum(s * y)
            rho = 1.0 / sy
            beta[i] = rho * np.sum(y * r)
            r = r + s * (alpha[i] - beta[i])

        return r
    
    def step_length(self, m, p, g, X0):
        c1 = 1e-4
        c2 = 0.9
        gTp0 = np.sum(g * p)
        vmin = np.min(self.m0)
        alpha = 0.01 * (1.0 / (vmin*vmin))
        for _ in range(10):
            m_new = m + alpha * p
            X_new = self.objective_function(m_new, save_residual=False)
            armijo = X_new <= X0 + c1 * alpha * gTp0
            print("X0 + c1 * alpha * gTp0 = " , X0 + c1 * alpha * gTp0)
            print("alpha =", alpha)
            print("X0 =", X0)
            print("X_new =", X_new)
            print("gTp0 =", gTp0)
            print("Armijo =", armijo)
            print()
        
            if armijo:
                return alpha

            alpha *= 0.5 

        return alpha
    
    def loadGradient(self):
        gradientFile = f"{self.pmt.migratedimageFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad
    
    def loadObsSeismogram(self,shot):
        seismogramFile = f"{self.pmt.seismogramFolder}seismogram_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        seismogram = np.fromfile(seismogramFile, dtype=np.float32).reshape(self.pmt.nt,self.pmt.Nrec) 
        return seismogram

    def save_residual(self,shot,residual):        
        self.seismogramFile = f"{self.pmt.seismogramFolder}residual_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        residual.tofile(self.seismogramFile)
        print(f"info: Residuo saved to {self.seismogramFile}")

    def solveFullWaveformInversion(self):
        start_time = time.time()
        print("info: Solving Full Waveform Inversion")
        
        # Modelo inicial
        water_mask = np.abs(self.wf.vp - 1500.0) < 1e-3
        self.m0 = smooth_model(self.wf.vp, self.pmt.sigma, water_mask).copy()
        m = 1.0 / (self.m0 * self.m0)
        smooth_model_file = (f"{self.pmt.modelFolder}fwi_vp_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.m0.astype(np.float32).tofile(smooth_model_file)

        s_store = []
        y_store = []

        # Gradiente e função objetivo no modelo atual
        X = self.objective_function(m, save_residual = True)
        g = self.calculate_gradient(m)
        for itr in range(self.pmt.niter):
            print(f"\033[31minfo: FWI iteration {itr + 1}/{self.pmt.niter}\033[0m")
            
            # Salvar gradiente da iteração atual
            gradient_file = (f"{self.pmt.migratedimageFolder}gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
            (g).astype(np.float32).tofile(gradient_file)
            print(f"info: Gradient saved to {gradient_file}")

            # Direção de busca: LBFGS
            p = -self.two_loop_recursion(g,s_store,y_store)
            p = p/np.max(np.abs(p))

            # Line search
            alpha = self.step_length(m, p, g, X)

            # Atualização do modelo
            m_new = m + alpha * p
    
            X_new = self.objective_function(m_new, save_residual = True)
            g_new = self.calculate_gradient(m_new)

            s = (m_new - m)
            y = (g_new - g)
            s_store.append(s)
            y_store.append(y)

            m = m_new.copy()
            X = X_new
            g = g_new.copy()

            m_it = 1.0 / np.sqrt(m)
            model_file = (f"{self.pmt.modelFolder}fwi_vp_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}.bin")
            m_it.astype(np.float32).tofile(model_file)
            print(f"info: Model of {itr+1} iteration saved to {model_file}")

        end_time = time.time()
        print(f"\ninfo: FWI finished in {end_time - start_time:.2f} s")

