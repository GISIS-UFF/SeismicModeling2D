import numpy as np
import time
import cupy as cp
from utils import smooth_model
from utils import smooth_parameter
# from utils import low_pass_filter
from utils import ricker
from utils import AGC

class fwi:
    def __init__(self,parameters,wavefield,migration):
        self.pmt = parameters
        self.wf = wavefield
        self.mig = migration

    def objective_function(self, m, itr, save_residual):
        X = 0.0
        self.vp = 1.0 / np.sqrt(m)   
        self.wf.source = cp.asarray(self.wf.source, dtype=cp.float32)
        self.wf.vp_exp = self.wf.ExpandModel(self.vp)
        self.wf.vp_exp = cp.asarray(self.wf.vp_exp, dtype=cp.float32)
        if self.pmt.ABC == "cerjan":
            self.wf.A = self.wf.createCerjanVector()
            self.wf.A = cp.asarray(self.wf.A, dtype=cp.float32)
        elif self.pmt.ABC == "CPML":
            self.wf.d0, self.wf.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            if self.pmt.multiparameter == False:
                self.epsilon = self.wf.epsilon#smooth_parameter(self.wf.epsilon,20)
                self.delta = self.wf.delta#smooth_parameter(self.wf.delta, 20)
            self.wf.epsilon_exp = self.wf.ExpandModel(self.epsilon)
            self.wf.delta_exp = self.wf.ExpandModel(self.delta)
            self.wf.epsilon_exp  = cp.asarray(self.wf.epsilon_exp, dtype=cp.float32)
            self.wf.delta_exp  = cp.asarray(self.wf.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                if self.pmt.multiparameter == False:
                    self.theta = self.wf.theta#smooth_parameter(self.wf.theta, 20)
                self.wf.theta_exp = self.wf.ExpandModel(self.theta)
                self.wf.theta_exp  = cp.asarray(self.wf.theta_exp, dtype=cp.float32)
        
        self.pmt.rx = cp.asarray(self.pmt.rx)
        self.pmt.rz = cp.asarray(self.pmt.rz)
        for shot in range(self.pmt.Nshot):
            dobs = self.loadObsSeismogram(shot)
            self.wf.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.wf.isx = self.pmt.sx[shot]
            self.wf.isz = self.pmt.sz[shot]            
            for k in range(self.pmt.nt): 
                self.wf.forward_stepGPU(k)
                # Register seismogram and snapshot
                self.wf.store_seismogram(k,self.pmt.rz,self.pmt.rx)      
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            self.seismogram = cp.asnumpy(self.wf.seismogram_gpu)
            if 10 <= itr < 15:
                dobs = AGC(dobs, self.pmt.dt)
                self.seismogram = AGC(self.seismogram, self.pmt.dt)
            residual = dobs - self.seismogram
            if save_residual==True:
                self.save_residual(shot,residual) 
            X += 0.5 * np.sum(residual * residual)
        return X

    def calculate_gradient(self, m):
        self.mig.vp = 1.0 / np.sqrt(m)
        grad = np.zeros_like(m)
        self.mig.ilum.fill(0)
        self.mig.migrated_image.fill(0)
        if self.pmt.fwi == True and self.pmt.multiparameter == True:
            if self.pmt.approximation in ["VTI", "TTI"]:
                self.mig.epsilon_grad.fill(0)
                self.mig.delta_grad.fill(0)
            if self.pmt.approximation == "TTI":
                self.mig.theta_grad.fill(0)
        self.mig.SolveBackwardWaveEquation()
        grad = self.loadGradient()
        water_mask = np.abs(self.wf.vp - 1500) < 1e-3
        grad[water_mask] = 0.0
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
    
    def step_length(self, m, p, g, X, itr):
        c1 = 1e-4
        gTp0 = np.sum(g * p)
        vmin = np.min(self.m0)
        alpha = 0.01 * (1.0 / (vmin*vmin))
        m_min = 1.0/(self.pmt.vmax*self.pmt.vmax)
        m_max = 1.0/(self.pmt.vmin*self.pmt.vmin)
        for _ in range(10):
            m_new = m + alpha * p
            m_new = np.clip(m_new, m_min, m_max)

            X_new = self.objective_function(m_new, itr, save_residual=False)

            armijo = X_new <= X + c1 * alpha * gTp0

            print("alpha =", alpha)
            print("X =", X)
            print("X_new =", X_new)
            print("gTp0 =", gTp0)
            print("Armijo =", armijo)

            if armijo:
                return alpha

            alpha *= 0.5

        return 0.0

    def loadGradient(self):
        gradientFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad
    
    def loadObsSeismogram(self,shot):
        seismogramFile = f"{self.pmt.seismogramFolder}seismogram_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}_fcut{self.pmt.fcut}.bin"
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
        mask = np.abs(self.wf.vp - np.min(self.wf.vp)) < 1e-3 
        self.m0 = smooth_model(self.wf.vp,self.pmt.sigma,mask).copy()
        smooth_model_file = (f"{self.pmt.modelFolder}fwi_vp_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.m0.astype(np.float32).tofile(smooth_model_file)

        # Vagarosidade ao quadrado
        m = 1.0 / (self.m0 * self.m0)

        self.history = []

        for fmax in self.pmt.freqs:
            print(f"\033[31minfo: FWI frequency {fmax} of {self.pmt.freqs}\033[0m")

            self.pmt.fcut = fmax
            self.wf.createSourceWavelet()

            s_store = []
            y_store = []

            # Gradiente e função objetivo no modelo atual
            X = self.objective_function(m, 0, save_residual = True)
            g = self.calculate_gradient(m)
            
            X0 = X

            self.history.append([X/X0, fmax])

            for itr in range(self.pmt.niter):
                print(f"\033[31minfo: FWI iteration {itr + 1}/{self.pmt.niter} for frequency {fmax}\033[0m")

                if itr in [10, 15]:
                    if itr == 10:
                        print(f"\033[31minfo: Applying AGC to residuals for iteration {itr + 1}\033[0m")
                    elif itr == 15:
                        print(f"\033[31minfo: Removing AGC from residuals for iteration {itr + 1}\033[0m")
                    s_store.clear()
                    y_store.clear()

                    X = self.objective_function(m, itr, save_residual=True)
                    g = self.calculate_gradient(m)

                # Salvar gradiente da iteração atual
                gradient_file = (f"{self.pmt.gradientsFolder}gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                (g).astype(np.float32).tofile(gradient_file)
                print(f"info: Gradient saved to {gradient_file}")

                # Direção de busca: LBFGS
                p = -self.two_loop_recursion(g,s_store,y_store)
                p = p/np.max(np.abs(p))

                # Line search
                alpha = self.step_length(m, p, g, X, itr)

                # Atualização do modelo
                m_min = 1.0/(self.pmt.vmax*self.pmt.vmax)
                m_max = 1.0/(self.pmt.vmin*self.pmt.vmin)

                m_new = m + alpha * p
                m_new = np.clip(m_new, m_min, m_max)

                X_new = self.objective_function(m_new, itr, save_residual = True)
                g_new = self.calculate_gradient(m_new)

                self.history.append([X_new/X0, fmax])

                s = (m_new - m)
                y = (g_new - g)
                sy = np.sum(s * y)
                if sy > 0:
                    s_store.append(s)
                    y_store.append(y)

                if len(s_store) > 8:
                    s_store.pop(0)
                    y_store.pop(0)

                m = m_new.copy()
                X = X_new
                g = g_new.copy()

                m_it = 1.0 / np.sqrt(m)
                model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_vp_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                m_it.astype(np.float32).tofile(model_file)
                print(f"info: Model of {itr+1} iteration saved to {model_file}")
            
        history = np.array(self.history, dtype=np.float32)
        history_file = (f"../outputs/history.txt")
        np.savetxt(history_file,history)

        print(f"info: FWI history saved to {history_file}")

        end_time = time.time()
        print(f"\ninfo: FWI finished in {end_time - start_time:.2f} s")