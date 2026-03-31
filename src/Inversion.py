import numpy as np
import time
from utils import smooth_model

class fwi:
    def __init__(self,parameters,wavefield,migration):
        self.pmt = parameters
        self.wf = wavefield
        self.mig = migration

    def reg_tv(self):
        dm = self.vp0 - self.vpk
        tv = 0.0
        for j in range(1, self.pmt.nz):
            for i in range(1, self.pmt.nx):
                dmx = (dm[j, i] - dm[j, i-1]) / self.pmt.dx
                dmz = (dm[j, i] - dm[j-1, i]) / self.pmt.dz
                tv += np.sqrt(dmx*dmx + dmz*dmz) * self.pmt.dx * self.pmt.dz
        return tv

    def objective_function(self):
        self.X = 0.0
        for r in range(self.pmt.Nrec):
            for t in range(self.pmt.nt):
                self.X += 0.5 * (self.residual[t, r])*(self.residual[t, r])
        return self.X
    
    def updateHessian(self,H, m_new,m,g_new,g):   
        N = self.pmt.nz * self.pmt.nx    
        s = (m_new - m).reshape(N, 1)
        y = (g_new - g).reshape(N, 1)
        ys = (y.T @ s)[0,0]
        if ys <= 1e-12:
            return H
        rho = 1.0 / ys
        I = np.eye(N, dtype=np.float32)
        A = I - rho * (s @ y.T)
        B = I - rho * (y @ s.T)
        H_new = A @ H @ B + rho * (s @ s.T)
        return H_new
    
    def line_search(self,m, p, g, X,rx,rz,max_iter=10):
    alpha = 1.0
    c1 = 1e-4
    c2 = 0.9

    for _ in range(max_iter):
        m_new = m + alpha * p
        self.wf.vp = m_new.reshape(self.pmt.nz, self.pmt.nx).copy()
        for shot in range(self.pmt.Nshot):
            for k in range(self.pmt.nt):
                self.wf.forward_step(k)
                self.wf.store_seismogram(k,rz,rx) 
                self.dcal = self.wf.seismogram 

            self.dobs = self.mig.loadSeismogram(shot)
            self.residual = self.dcal - self.dobs
            self.save_residual(shot)
            X_new += self.objective_function()

        self.mig.SolveBackwardWaveEquation()
        g_new += self.loadGradient()
        g_new = g_new / np.max(np.abs(g_new))
                
        armijo = X_new <= X + c1 * alpha * g.T @ p
        curvature = abs(g_new.T @ p) <= c2 * abs(g.T @ p)

        if armijo and curvature:
            return alpha

        alpha *= 0.5

    return alpha

    def loadGradient(self):
        gradientFile = f"{self.pmt.migratedimageFolder}gradient_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        grad = np.fromfile(gradientFile, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        return grad
    
    def save_residual(self,shot):        
        self.seismogramFile = f"{self.pmt.seismogramFolder}residual_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        self.residual.tofile(self.seismogramFile)
        print(f"info: Residuo saved to {self.seismogramFile}")

    def solveFullWaveformInversion(self):
        start_time = time.time()
        print(f"info: Solving Full Waveform Inversion")
        # Expand velocity model and Create absorbing layers
        self.m = smooth_model(self.wf.vp, self.pmt.sigma)
        self.H = np.eye(self.pmt.nz * self.pmt.nx, dtype=np.float32)
        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        for it in range(self.pmt.iter):
            for shot in range(self.pmt.Nshot):

                for k in range(self.pmt.nt):
                    self.wf.forward_step(k)
                    self.wf.store_seismogram(k,rz,rx) 
                    self.dcal = self.wf.seismogram 

                self.dobs = self.mig.loadSeismogram(shot)
                self.residual = self.dcal - self.dobs
                self.save_residual(shot)
                self.X = self.objective_function()
                self.mig.SolveBackwardWaveEquation()
                self.grad = self.loadGradient()
                self.grad = self.grad / np.max(np.abs(self.grad))
                g_vec = self.grad.reshape(-1)
                self.p = -self.H @ g_vec
                self.alpha = self.line_search(self.m,self.p,self.grad,self.X,rx,rz)
                m_vec = self.m.reshape(-1)
                m_new_vec = m_vec + self.alpha * self.p
                self.m_new = m_new_vec.reshape(self.pmt.nz, self.pmt.nx)
                self.mig.SolveBackwardWaveEquation()
                self.grad_new = self.loadGradient()
                self.grad_new = self.grad_new / np.max(np.abs(self.grad_new))
                self.H = self.updateHessian(self.H,self.m_new,self.m,self.grad_new,self.grad)
                self.grad = self.grad_new
                self.m = self.m_new
                self.wf.vp = self.m_new.copy()


 


