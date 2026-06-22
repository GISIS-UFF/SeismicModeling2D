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

    def objective_function(self, vp, epsilon, delta, theta, itr, save_residual):
        X = 0.0
        self.vp = 1.0 / np.sqrt(vp)   
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
                self.epsilon = smooth_parameter(self.wf.epsilon,self.pmt.sigma)
                self.delta = smooth_parameter(self.wf.delta, self.pmt.sigma)
                self.wf.epsilon_exp = self.wf.ExpandModel(self.epsilon)
                self.wf.delta_exp = self.wf.ExpandModel(self.delta)
            else:
                self.wf.epsilon_exp = self.wf.ExpandModel(epsilon)
                self.wf.delta_exp = self.wf.ExpandModel(delta)
            self.wf.epsilon_exp  = cp.asarray(self.wf.epsilon_exp, dtype=cp.float32)
            self.wf.delta_exp  = cp.asarray(self.wf.delta_exp, dtype=cp.float32)
            if self.pmt.approximation == "TTI":
                if self.pmt.multiparameter == False:
                    self.theta = smooth_parameter(self.wf.theta, 20)
                    self.wf.theta_exp = self.wf.ExpandModel(self.theta)
                else:
                    self.wf.theta_exp = self.wf.ExpandModel(theta)
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
            if self.pmt.AGC == True and self.pmt.agc1itr <= itr < self.pmt.agc2itr:
                dobs = AGC(dobs, self.pmt.dt)
                self.seismogram = AGC(self.seismogram, self.pmt.dt)
            residual = dobs - self.seismogram
            if save_residual==True:
                self.save_residual(shot,residual) 
            X += 0.5 * np.sum(residual * residual)
        return X

    def calculate_gradient(self, vp, epsilon, delta, theta):
        self.mig.vp = 1.0 / np.sqrt(vp)
        self.mig.ilum.fill(0)
        self.mig.migrated_image.fill(0)
        if self.pmt.fwi == True and self.pmt.multiparameter == True:
            if self.pmt.approximation in ["VTI", "TTI"]:
                self.mig.epsilon = epsilon
                self.mig.delta = delta
                self.mig.epsilon_grad.fill(0)
                self.mig.delta_grad.fill(0)
            if self.pmt.approximation == "TTI":
                self.mig.theta = theta
                self.mig.theta_grad.fill(0)
        self.mig.SolveBackwardWaveEquation()
        water_mask = np.abs(self.wf.vp - 1500) < 1e-3
        g_vp = self.loadGradientVp()
        g_vp[water_mask] = 0.0
        g_eps = None
        g_delta = None
        g_theta = None
        if self.pmt.fwi == True and self.pmt.multiparameter == True:
            if self.pmt.approximation in ["VTI", "TTI"]:
                g_eps = self.loadGradientEps()
                g_delta = self.loadGradientDelta()
                g_eps[water_mask] = 0.0
                g_delta[water_mask] = 0.0
            if self.pmt.approximation == "TTI":
                g_theta = self.loadGradientTheta()
                g_theta[water_mask] = 0.0

        return g_vp, g_eps, g_delta, g_theta
    
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
    
    def step_length(self, parametro, vp, epsilon, delta, theta, p, g, X, itr):
        c1 = 1e-4
        gTp0 = np.sum(g * p)

        if parametro == "vp":
            alpha = 0.01 * np.max(np.abs(vp))
            m_min = 1.0 / (self.pmt.vmax * self.pmt.vmax)
            m_max = 1.0 / (self.pmt.vmin * self.pmt.vmin)

        elif parametro == "epsilon":
            alpha = 0.01 * np.max(np.abs(epsilon))
            m_min = self.pmt.epsmin
            m_max = self.pmt.epsmax

        elif parametro == "delta":
            alpha = 0.01 * np.max(np.abs(delta))
            m_min = self.pmt.deltamin
            m_max = self.pmt.deltamax

        elif parametro == "theta":
            alpha = 0.01 * np.max(np.abs(theta))
            m_min = self.pmt.thetamin
            m_max = self.pmt.thetamax

        for _ in range(10):

            if parametro == "vp":
                vp_new = vp + alpha * p
                vp_new = np.clip(vp_new, m_min, m_max)

                epsilon_new = epsilon
                delta_new = delta
                theta_new = theta

            elif parametro == "epsilon":
                vp_new = vp

                epsilon_new = epsilon + alpha * p
                epsilon_new = np.clip(epsilon_new, m_min, m_max)

                delta_new = delta
                theta_new = theta

            elif parametro == "delta":
                vp_new = vp
                epsilon_new = epsilon

                delta_new = delta + alpha * p
                delta_new = np.clip(delta_new, m_min, m_max)

                theta_new = theta

            elif parametro == "theta":
                vp_new = vp
                epsilon_new = epsilon
                delta_new = delta

                theta_new = theta + alpha * p
                theta_new = np.clip(theta_new, m_min, m_max)

            X_new = self.objective_function(vp_new, epsilon_new, delta_new, theta_new, itr, save_residual=False)

            armijo = X_new <= X + c1 * alpha * gTp0

            print("parametro =", parametro)
            print("alpha =", alpha)
            print("X =", X)
            print("X_new =", X_new)
            print("gTp0 =", gTp0)
            print("Armijo =", armijo)

            if armijo:
                return alpha

            alpha *= 0.5

        return alpha

    def loadGradientVp(self):
        gradientFile = f"{self.pmt.gradientsFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad

    def loadGradientEps(self):
        gradientFile = f"{self.pmt.gradientsFolder}epsilon_gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad

    def loadGradientDelta(self):
        gradientFile = f"{self.pmt.gradientsFolder}delta_gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad

    def loadGradientTheta(self):
        gradientFile = f"{self.pmt.gradientsFolder}theta_gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
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
        
        # Modelo inicial de vp
        mask_vp = np.abs(self.wf.vp - np.min(self.wf.vp)) < 1e-3 
        self.vp0 = smooth_model(self.wf.vp,self.pmt.sigma,mask_vp)
        smooth_model_file = (f"{self.pmt.modelFolder}fwi_vp_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
        self.vp0.astype(np.float32).tofile(smooth_model_file)

        # Vagarosidade ao quadrado
        vp = 1.0 / (self.vp0 * self.vp0)
        epsilon = None
        delta = None
        theta = None

        if self.pmt.multiparameter == True:
            if self.pmt.approximation in ["VTI","TTI"]:
                # Modelo inicial de epsilon
                mask_eps = np.abs(self.wf.epsilon - np.min(self.wf.epsilon)) < 1e-3 
                self.eps0 = smooth_parameter(self.wf.epsilon,self.pmt.sigma,mask_eps)
                smooth_model_file = (f"{self.pmt.modelFolder}fwi_epsilon_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
                self.eps0.astype(np.float32).tofile(smooth_model_file)
                
                # Modelo inicial de delta
                mask_delta = np.abs(self.wf.delta - np.min(self.wf.delta)) < 1e-3 
                self.delta0 = smooth_parameter(self.wf.delta,self.pmt.sigma,mask_delta)
                smooth_model_file = (f"{self.pmt.modelFolder}fwi_delta_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
                self.delta0.astype(np.float32).tofile(smooth_model_file)

                epsilon = self.eps0
                delta = self.delta0

            if self.pmt.approximation == "TTI":
                # Modelo inicial de theta
                mask_theta = np.abs(self.wf.theta - np.min(self.wf.theta)) < 1e-3 
                self.theta0 = smooth_parameter(self.wf.theta,self.pmt.sigma,mask_theta)
                smooth_model_file = (f"{self.pmt.modelFolder}fwi_theta_smooth_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin")
                self.theta0.astype(np.float32).tofile(smooth_model_file)

                theta = self.theta0

        self.history = []

        for fmax in self.pmt.freqs:
            print(f"\033[31minfo: FWI frequency {fmax} of {self.pmt.freqs}\033[0m")

            self.pmt.fcut = fmax
            self.wf.createSourceWavelet()

            s_vp_store = []
            y_vp_store = []
            if self.pmt.multiparameter == True:
                if self.pmt.approximation in ["VTI","TTI"]:
                    s_eps_store = []
                    y_eps_store = []
                    s_delta_store = []
                    y_delta_store = []
                if self.pmt.approximation == "TTI":
                    s_theta_store = []
                    y_theta_store = []

            # Gradiente e função objetivo no modelo atual
            X = self.objective_function(vp, epsilon, delta, theta, 0, save_residual = True)
            g_vp, g_eps, g_delta, g_theta = self.calculate_gradient(vp, epsilon, delta, theta)
            
            X0 = X

            self.history.append([X/X0, fmax])

            for itr in range(self.pmt.niter):
                print(f"\033[31minfo: FWI iteration {itr + 1}/{self.pmt.niter} for frequency {fmax}\033[0m")

                if itr in [10, 15]:
                    if itr == 10:
                        print(f"\033[31minfo: Applying AGC to residuals for iteration {itr + 1}\033[0m")
                    elif itr == 15:
                        print(f"\033[31minfo: Removing AGC from residuals for iteration {itr + 1}\033[0m")
                    s_vp_store.clear()
                    y_vp_store.clear()
                    if self.pmt.multiparameter == True:
                        if self.pmt.approximation in ["VTI","TTI"]:
                            s_eps_store.clear()
                            y_eps_store.clear()
                            s_delta_store.clear()
                            y_delta_store.clear()
                        if self.pmt.approximation == "TTI":
                            s_theta_store.clear()
                            y_theta_store.clear()

                    X = self.objective_function(vp, epsilon, delta, theta, itr, save_residual=True)
                    g_vp, g_eps, g_delta, g_theta = self.calculate_gradient(vp, epsilon, delta, theta)


                # Salvar gradiente da iteração atual
                gradient_file = (f"{self.pmt.gradientsFolder}vp_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                (g_vp).astype(np.float32).tofile(gradient_file)
                print(f"info: Vp gradient saved to {gradient_file}")
                if self.pmt.multiparameter == True:
                    if self.pmt.approximation in ["VTI","TTI"]:
                        gradient_file = (f"{self.pmt.gradientsFolder}epsilon_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                        (g_eps).astype(np.float32).tofile(gradient_file)
                        print(f"info: Epsilon gradient saved to {gradient_file}")
                        gradient_file = (f"{self.pmt.gradientsFolder}delta_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                        (g_delta).astype(np.float32).tofile(gradient_file)
                        print(f"info: Delta gradient saved to {gradient_file}")
                    if self.pmt.approximation == "TTI":
                        gradient_file = (f"{self.pmt.gradientsFolder}theta_gradient_fwi_iter_{itr+1}_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_freq{fmax}.bin")
                        (g_theta).astype(np.float32).tofile(gradient_file)
                        print(f"info: Theta gradient saved to {gradient_file}")

                # Direção de busca: LBFGS
                p_vp = -self.two_loop_recursion(g_vp,s_vp_store,y_vp_store)
                p_vp = p_vp/np.max(np.abs(p_vp))

                p_eps = None
                p_delta = None
                p_theta = None
                if self.pmt.multiparameter == True:
                    if self.pmt.approximation in ["VTI","TTI"]:
                        p_eps = -self.two_loop_recursion(g_eps,s_eps_store,y_eps_store)
                        p_eps = p_eps/np.max(np.abs(p_eps))

                        p_delta = -self.two_loop_recursion(g_delta,s_delta_store,y_delta_store)
                        p_delta = p_delta/np.max(np.abs(p_delta))

                    if self.pmt.approximation == "TTI":
                        p_theta = -self.two_loop_recursion(g_theta,s_theta_store,y_theta_store)
                        p_theta = p_theta/np.max(np.abs(p_theta))

                # Line search
                alpha_vp = self.step_length("vp", vp, epsilon, delta, theta, p_vp, g_vp, X, itr)
                if self.pmt.multiparameter == True:
                    if self.pmt.approximation in ["VTI","TTI"]:
                        alpha_eps = self.step_length("epsilon", vp, epsilon, delta, theta, p_eps, g_eps, X, itr)
                        alpha_delta = self.step_length("delta", vp, epsilon, delta, theta, p_delta, g_delta, X, itr)
                    if self.pmt.approximation == "TTI":
                        alpha_theta = self.step_length("theta", vp, epsilon, delta, theta, p_theta, g_theta, X, itr)

                # Atualização do modelo
                vp_min = 1.0/(self.pmt.vmax*self.pmt.vmax)
                vp_max = 1.0/(self.pmt.vmin*self.pmt.vmin)

                vp_new = vp + alpha_vp * p_vp
                vp_new = np.clip(vp_new, vp_min, vp_max)

                epsilon_new = epsilon
                delta_new = delta
                theta_new = theta
                if self.pmt.multiparameter == True:
                    if self.pmt.approximation in ["VTI","TTI"]:
                        epsilon_new = epsilon + alpha_eps * p_eps
                        delta_new = delta + alpha_delta * p_delta
                        epsilon_new = np.clip(epsilon_new, self.pmt.epsmin, self.pmt.epsmax)
                        delta_new = np.clip(delta_new, self.pmt.deltamin, self.pmt.deltamax)
                    if self.pmt.approximation == "TTI":
                        theta_new = theta + alpha_theta * p_theta
                        theta_new = np.clip(theta_new, self.pmt.thetamin, self.pmt.thetamax)

                X_new = self.objective_function(vp_new, epsilon_new, delta_new, theta_new, itr, save_residual = True)
                g_vp_new, g_eps_new, g_delta_new, g_theta_new = self.calculate_gradient(vp_new, epsilon_new, delta_new, theta_new)

                self.history.append([X_new/X0, fmax])

                s_vp = (vp_new - vp)
                y_vp = (g_vp_new - g_vp)
                sy_vp = np.sum(s_vp * y_vp)
                if sy_vp > 0:
                    s_vp_store.append(s_vp)
                    y_vp_store.append(y_vp)

                if self.pmt.multiparameter == True:
                    if self.pmt.approximation in ["VTI","TTI"]:
                        s_eps = (epsilon_new - epsilon)
                        y_eps = (g_eps_new - g_eps)
                        sy_eps = np.sum(s_eps * y_eps)
                        if sy_eps > 0:
                            s_eps_store.append(s_eps)
                            y_eps_store.append(y_eps)

                        s_delta = (delta_new - delta)
                        y_delta = (g_delta_new - g_delta)
                        sy_delta = np.sum(s_delta * y_delta)
                        if sy_delta > 0:
                            s_delta_store.append(s_delta)
                            y_delta_store.append(y_delta)

                    if self.pmt.approximation == "TTI":
                        s_theta = (theta_new - theta)
                        y_theta = (g_theta_new - g_theta)
                        sy_theta = np.sum(s_theta * y_theta)
                        if sy_theta > 0:
                            s_theta_store.append(s_theta)
                            y_theta_store.append(y_theta)

                if len(s_vp_store) > 8:
                    s_vp_store.pop(0)
                    y_vp_store.pop(0)
                if self.pmt.multiparameter == True:
                    if self.pmt.approximation in ["VTI","TTI"]:
                        if len(s_eps_store) > 8:
                            s_eps_store.pop(0)
                            y_eps_store.pop(0)
                        if len(s_delta_store) > 8:
                            s_delta_store.pop(0)
                            y_delta_store.pop(0)
                    if self.pmt.approximation == "TTI":
                        if len(s_theta_store) > 8:
                            s_theta_store.pop(0)
                            y_theta_store.pop(0)

                vp = vp_new.copy()
                g_vp = g_vp_new.copy()
                X = X_new

                if self.pmt.multiparameter == True:
                    if self.pmt.approximation in ["VTI","TTI"]:
                        epsilon = epsilon_new.copy()
                        delta = delta_new.copy()
                        g_eps = g_eps_new.copy()
                        g_delta = g_delta_new.copy()
                    if self.pmt.approximation == "TTI":
                        theta = theta_new.copy()
                        g_theta = g_theta_new.copy()

                m_it = 1.0 / np.sqrt(vp)
                model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_vp_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                m_it.astype(np.float32).tofile(model_file)
                print(f"info: Model of {itr+1} iteration saved to {model_file}")
                if self.pmt.multiparameter == True:
                    if self.pmt.approximation in ["VTI","TTI"]:
                        model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_epsilon_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                        epsilon.astype(np.float32).tofile(model_file)
                        print(f"info: Epsilon model of {itr+1} iteration saved to {model_file}")
                        model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_delta_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                        delta.astype(np.float32).tofile(model_file)
                        print(f"info: Delta model of {itr+1} iteration saved to {model_file}")
                    if self.pmt.approximation == "TTI":
                        model_file = (f"{self.pmt.estimatedmodelsFolder}fwi_theta_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_itr{itr+1}_freq{fmax}.bin")
                        theta.astype(np.float32).tofile(model_file)
                        print(f"info: Theta model of {itr+1} iteration saved to {model_file}")
            
            
        history = np.array(self.history, dtype=np.float32)
        history_file = (f"../outputs/history.txt")
        np.savetxt(history_file,history)

        print(f"info: FWI history saved to {history_file}")

        end_time = time.time()
        print(f"\ninfo: FWI finished in {end_time - start_time:.2f} s")