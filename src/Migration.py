import numpy as np
from utils import updateWaveEquation
from utils import updateWaveEquationCPML
from utils import updateWaveEquationVTI
from utils import updateWaveEquationVTICPML
from utils import updateWaveEquationTTI
from utils import AbsorbingBoundary
from utils import updatePsi
from utils import updateZeta
from utils import Mute

class migration: 

    def __init__(self,wavefield,parameters):
        self.pmt = parameters
        self.wf = wavefield
    
    def loadSeismogram(self, shot):
        seismogramFile = f"{self.pmt.seismogramFolder}seismogram_shot_{shot+1}_Nt{self.pmt.nt}_Nrec{self.pmt.Nrec}.bin"
        seismogram = np.fromfile(seismogramFile, dtype=np.float32).reshape(self.pmt.nt,self.pmt.Nrec) 
        return seismogram
        
    def laplacian_filter(self, f):
        dim1,dim2 = np.shape(f)
        g = np.zeros([dim1,dim2])
        lap_z = 0
        lap_x = 0
        for ix in range(1, dim2 - 1):
            for iz in range(1, dim1 - 1):
                lap_z = f[iz+1, ix] + f[iz-1, ix] - 2 * f[iz, ix]
                lap_x = f[iz, ix+1] + f[iz, ix-1] - 2 * f[iz, ix]
                g[iz, ix] = lap_z/(self.pmt.dz*self.pmt.dz) + lap_x/(self.pmt.dx*self.pmt.dx)

        for ix in range(dim2):
            g[0, ix] = g[1, ix]
            g[-1, ix] = g[-2, ix]
        for iz in range(dim1):
            g[iz, 0] = g[iz, 1]
            g[iz, -1] = g[iz, -2]

        return(-g)
    
    def gaussian_kernel(self, x, z, sigma):
        fator = 1. / (2.*np.pi*sigma*sigma)
        expoente = -(x * x + z * z)/(2.*sigma*sigma)
        return fator * np.exp(expoente)

    def gaussian_filter2D(self,sigma):
        kernel_size = np.ceil(2 * sigma + 1).astype(int)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel2d = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        total = 0.0

        for lin in range(kernel_size):
            for col in range(kernel_size):
                x = lin - kernel_size // 2
                y = col - kernel_size // 2
                val = self.gaussian_kernel(x, y, sigma)
                kernel2d[lin, col] = val
                total += val

        kernel2d /= total

        return kernel2d

    def smooth_model(self,f,sigma):
        s = 1.0 / f
        kernel = self.gaussian_filter2D(sigma)
        ksize = kernel.shape[0]
        half = ksize // 2

        nz, nx = np.shape(s)

        for z in range(half, nz - half):
            for x in range(half, nx - half):
                new_value = 0.0
                for i in range(ksize):
                    for j in range(ksize):
                        new_value += (kernel[i, j] * s[z + i - half, x + j - half])
                s[z, x] = new_value

        for z in range(half):
            s[z, :] = s[half, :]
            s[nz - 1 - z, :] = s[nz - 1 - half, :]
        for x in range(half):
            s[:, x] = s[:, half]
            s[:, nx - 1 - x] = s[:, nx - 1 - half]

        return (1.0 / s)
                   
    def load_checkpoint(self, shot, k):
        checkpointFile = (f"{self.pmt.checkpointFolder}{self.pmt.approximation}{self.pmt.ABC}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")
        with open(checkpointFile, "rb") as file:
            count = self.pmt.nx_abc * self.pmt.nz_abc
            self.wf.current = np.fromfile(file, np.float32, count).reshape(self.pmt.nz_abc, self.pmt.nx_abc)
            self.wf.future  = np.fromfile(file, np.float32, count).reshape(self.pmt.nz_abc, self.pmt.nx_abc)
    
    def save_checkpoint(self, shot, k):
        if self.pmt.migration != "checkpoint":
            return
        if k > self.pmt.last_save:
            return
        if k % self.pmt.step != 0:
            return
        
        checkpointFile = (f"{self.pmt.checkpointFolder}{self.pmt.approximation}{self.pmt.ABC}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}.bin")

        save = [self.wf.current, self.wf.future]

        with open(checkpointFile, "wb") as file:
            for field in save:
                field.astype(np.float32).tofile(file)

        print(f"info: Checkpoint saved to {checkpointFile}")
    
    def forward_step(self,k):
        if self.pmt.migration in ["onthefly","checkpoint","RBC"]:
            if self.pmt.approximation == "acoustic":
                self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
            if self.pmt.approximation == "VTI":
                self.wf.future= updateWaveEquationVTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
            elif self.pmt.approximation == "TTI":
                self.wf.future= updateWaveEquationTTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
        
        if self.pmt.migration == "SB":
            if self.pmt.approximation == "acoustic" and self.pmt.ABC == "cerjan":
                self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.current, self.A)

            elif self.pmt.approximation == "acoustic" and self.pmt.ABC == "CPML":
                self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.future = updateWaveEquationCPML(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)             
                self.wf.future[self.sz,self.sx] += self.wf.source[k]

            elif self.pmt.approximation == "VTI" and self.pmt.ABC == "cerjan":
                self.wf.future= updateWaveEquationVTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.current, self.A)

            elif self.pmt.approximation == "VTI" and self.pmt.ABC == "CPML":
                self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD = updatePsi(self.wf.PsixFR, self.wf.PsixFL,self.wf.PsizFU, self.wf.PsizFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx, self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD = updateZeta(self.wf.PsixFR, self.wf.PsixFL, self.wf.ZetaxFR, self.wf.ZetaxFL,self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.nx_abc, self.pmt.nz_abc, self.wf.current, self.pmt.dx,self.pmt.dz, self.pmt.N_abc, self.f_pico, self.d0, self.pmt.dt, self.vp_exp)
                self.wf.future = updateWaveEquationVTICPML(self.wf.future, self.wf.current, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp,self.pmt.nx_abc, self.pmt.nz_abc, self.wf.PsixFR, self.wf.PsixFL, self.wf.PsizFU, self.wf.PsizFD, self.wf.ZetaxFR, self.wf.ZetaxFL, self.wf.ZetazFU, self.wf.ZetazFD, self.pmt.N_abc)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]

            elif self.pmt.approximation == "TTI" and self.pmt.ABC == "cerjan":
                self.wf.future= updateWaveEquationTTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
                self.wf.future[self.sz,self.sx] += self.wf.source[k]
                # Apply absorbing boundary condition
                self.wf.future = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.future, self.A)
                self.wf.current = AbsorbingBoundary(self.pmt.N_abc, self.pmt.nz_abc, self.pmt.nx_abc, self.wf.current, self.A)

    def reconstructed_step(self,t):
        if self.pmt.approximation == "acoustic":
            self.wf.future = updateWaveEquation(self.wf.future, self.wf.current, self.vp_exp, self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            self.wf.future[self.sz, self.sx] -= self.wf.source[t]
        elif self.pmt.approximation == "VTI":
            self.wf.future= updateWaveEquationVTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            self.wf.future[self.sz, self.sx] -= self.wf.source[t]
        elif self.pmt.approximation == "TTI":
            self.wf.future= updateWaveEquationTTI(self.wf.future, self.wf.current, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
            self.wf.future[self.sz, self.sx] -= self.wf.source[t]
        
    def save_boundaries(self,k): 
        if self.pmt.migration == "SB":
            self.wf.top[k,:,:]   = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + 4, self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]
            self.wf.bot[k,:,:]   = self.wf.future[self.pmt.N_abc + self.pmt.nz - 4: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]
            self.wf.left[k,:,:]  = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc+4]
            self.wf.right[k,:,:] = self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz,self.pmt.N_abc + self.pmt.nx - 4: self.pmt.N_abc + self.pmt.nx]

    def apply_boundaries(self,k):
        if self.pmt.migration == "SB":
            self.wf.future[self.pmt.N_abc: self.pmt.N_abc + 4, self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx] = self.wf.top[k,:,:]
            self.wf.future[self.pmt.N_abc + self.pmt.nz - 4: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc + self.pmt.nx]  = self.wf.bot[k,:,:]
            self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz , self.pmt.N_abc: self.pmt.N_abc+4] = self.wf.left[k,:,:] 
            self.wf.future[self.pmt.N_abc: self.pmt.N_abc + self.pmt.nz,self.pmt.N_abc + self.pmt.nx - 4: self.pmt.N_abc + self.pmt.nx] = self.wf.right[k,:,:]
       
    def backward_step(self,k):
        if self.pmt.approximation == "acoustic":
            self.wf.futurebck = updateWaveEquation(self.wf.futurebck, self.wf.currentbck, self.vp_exp,self.pmt.nz_abc, self.pmt.nx_abc, self.pmt.dz, self.pmt.dx, self.pmt.dt)
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]
            
        elif self.pmt.approximation == "VTI":
            self.wf.futurebck= updateWaveEquationVTI(self.wf.futurebck, self.wf.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp)
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]
            
        elif self.pmt.approximation == "TTI":
            self.wf.futurebck= updateWaveEquationTTI(self.wf.futurebck, self.wf.currentbck, self.pmt.nx_abc, self.pmt.nz_abc, self.pmt.dt, self.pmt.dx, self.pmt.dz, self.vp_exp, self.epsilon_exp, self.delta_exp, self.theta_exp)
            self.wf.futurebck[self.rz, self.rx] += self.muted_seismogram[k, :]

    def build_ckpts_steps(self):
        self.ckpts_steps = []
        for t0 in range (250,self.pmt.nt-1,self.pmt.step):
            t1 = min(t0 + self.pmt.step,self.pmt.nt-1)
            self.ckpts_steps.append((t0,t1))
    
    def reset_field(self):
        self.wf.current.fill(0)
        self.wf.future.fill(0)
        self.wf.currentbck.fill(0)
        self.wf.futurebck.fill(0)
        if self.pmt.ABC == "CPML":
            self.wf.PsixFR.fill(0)
            self.wf.PsixFL.fill(0)
            self.wf.PsizFU.fill(0)  
            self.wf.PsizFD.fill(0) 
            self.wf.ZetaxFR.fill(0)
            self.wf.ZetaxFL.fill(0)
            self.wf.ZetazFU.fill(0)
            self.wf.ZetazFD.fill(0)

    def create_random_boundary(self):
        cmax = 0.5
        v_limite = (cmax*self.pmt.dx)/self.pmt.dt
        A = self.vp.min() * 1.3
        for i in range(self.pmt.nx_abc):
            for j in range(self.pmt.nz_abc):
                if i < self.pmt.N_abc:
                    dx = self.pmt.N_abc - i
                elif i >= self.pmt.nx_abc - self.pmt.N_abc:
                    dx = i - (self.pmt.nx_abc - self.pmt.N_abc)
                else:
                    dx = 0

                if j < self.pmt.N_abc:
                    dz = self.pmt.N_abc - j
                elif j >= self.pmt.nz_abc - self.pmt.N_abc:
                    dz = j - (self.pmt.nz_abc - self.pmt.N_abc)
                else:
                    dz = 0

                d = np.sqrt(dx*dx + dz*dz)
                
                d = d / self.pmt.N_abc
                
                found = False
                while found == False:
                    r = A * (2.0*np.random.rand()-1.0)
                    vtest = self.vp_exp[j,i] + r * d
                    if vtest <= v_limite:
                        self.vp_exp[j,i] = vtest
                        found = True
        # plt.figure()
        # plt.imshow(self.vp_exp, cmap = 'jet')
        # plt.show()

    #On the fly
    def solveBackwardWaveEquationOntheFly(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)
        
        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        save_field = np.zeros([self.pmt.nt,self.pmt.nz,self.pmt.nx],dtype=np.float32)
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")

            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,window = 0.3,v0=2000)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5,5))
            plt.plot(self.muted_seismogram[:, 300],label = "muted")
            plt.plot(seismogram[:, 300],label = "seism")
            plt.legend()
            plt.show()
            plt.figure()
            plt.imshow(self.muted_seismogram)
            plt.show()
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.forward_step(k)
                save_field[k,:,:] = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for t in range(self.pmt.nt - 1, -1, -1):
                self.backward_step(t)
                self.migrated_partial += (save_field[t,:,:] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.ilum += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                #swap
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum + 1e-12)
            print(f"info: Shot {shot+1} backward done.")
     
        # Apply laplacian_filter filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    # REGULAR CHECKPOINTING
    def solveBackwardWaveEquationCheckpointing(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)
        
        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,window = 0.3,v0=2000)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            self.build_ckpts_steps()
            for k in range(self.pmt.nt):
                self.forward_step(k)
                self.save_checkpoint(shot, k)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            for (t0,t1) in reversed(self.ckpts_steps):
                self.load_checkpoint(shot,t1)
                for t in range(t1, t0, -1):
                    self.reconstructed_step(t)
                    self.backward_step(t)
                    self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                    self.ilum += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                    #swap
                    self.wf.current, self.wf.future = self.wf.future, self.wf.current
                    self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck
            self.wf.migrated_image += self.migrated_partial / (self.ilum + 1e-12)
            print(f"info: Shot {shot+1} backward done.")
     
        # Apply laplacian_filter filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    #Saving Boundaries
    def solveBackwardWaveEquationSavingBoundaries(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.ABC == "cerjan":
            self.A = self.wf.createCerjanVector()
        elif self.pmt.ABC == "CPML":
            self.d0, self.f_pico = self.wf.dampening_const()
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)

        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,window = 0.3,v0=1500)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.save_boundaries(k)
                self.forward_step(k)            
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current 
            self.wf.current, self.wf.future = self.wf.future, self.wf.current 
            for t in range(self.pmt.nt - 1, 200, -1):
                self.reconstructed_step(t)
                self.apply_boundaries(t)           
                self.backward_step(t)
                self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])  
                self.ilum += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck

            self.wf.migrated_image += self.migrated_partial / (self.ilum + 1e-12)
            print(f"info: Shot {shot+1} backward done.")
     
        # Apply laplacian_filter filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    #Random Boundary Condintion
    def solveBackwardWaveEquationRBC(self):
        print(f"info: Solving backward acoustic wave equation")
        # Expand velocity model and Create absorbing layers
        self.vp = self.smooth_model(self.wf.vp, 9)
        self.vp_exp = self.wf.ExpandModel(self.vp)
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.epsilon_exp = self.wf.ExpandModel(self.wf.epsilon)
            self.delta_exp = self.wf.ExpandModel(self.wf.delta)
            if self.pmt.approximation == "TTI":
                self.theta_exp = self.wf.ExpandModel(self.wf.theta)

        self.rx = np.int32(self.pmt.rec_x/self.pmt.dx) + self.pmt.N_abc
        self.rz = np.int32(self.pmt.rec_z/self.pmt.dz) + self.pmt.N_abc
        for shot in range(self.pmt.Nshot):
            print(f"info: Shot {shot+1} of {self.pmt.Nshot}")
            self.reset_field()
            self.create_random_boundary()

            # convert acquisition geometry coordinates to grid points
            self.sx = int(self.pmt.shot_x[shot]/self.pmt.dx) + self.pmt.N_abc
            self.sz = int(self.pmt.shot_z[shot]/self.pmt.dz) + self.pmt.N_abc  

            # Top muting
            seismogram = self.loadSeismogram(shot)
            self.muted_seismogram = Mute(seismogram, shot, self.pmt.rec_x, self.pmt.rec_z, self.pmt.shot_x, self.pmt.shot_z, self.pmt.dt,window = 0.3,v0=2000)
            self.migrated_partial = np.zeros_like(self.wf.migrated_image)
            self.ilum = np.zeros_like(self.wf.migrated_image)
            for k in range(self.pmt.nt):
                self.forward_step(k)
                import matplotlib.pyplot as plt
                if k == 1200:
                    snapshot = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                    snapshotFile = (f"{self.pmt.snapshotFolder}{self.pmt.approximation}{self.pmt.ABC}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{k}forward.bin")
                    snapshot.tofile(snapshotFile)
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
            self.wf.current, self.wf.future = self.wf.future, self.wf.current    
            for t in range(self.pmt.nt - 1, 200, -1): 
                self.reconstructed_step(t)
                self.backward_step(t) 
                if t == 1200:
                    snapshot = self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                    snapshotFile = (f"{self.pmt.snapshotFolder}{self.pmt.approximation}{self.pmt.ABC}_shot_{shot+1}_Nx{self.pmt.nx}_Nz{self.pmt.nz}_Nt{self.pmt.nt}_frame_{t}RBC.bin")
                    snapshot.tofile(snapshotFile)     
                self.migrated_partial += (self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.currentbck[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc])
                self.ilum += self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc] * self.wf.current[self.pmt.N_abc:self.pmt.nz_abc - self.pmt.N_abc,self.pmt.N_abc:self.pmt.nx_abc - self.pmt.N_abc]
                #swap
                self.wf.current, self.wf.future = self.wf.future, self.wf.current
                self.wf.currentbck, self.wf.futurebck = self.wf.futurebck, self.wf.currentbck

            self.wf.migrated_image += self.migrated_partial / (self.ilum + 1e-12)
            print(f"info: Shot {shot+1} backward done.")
     
        # Apply Laplacian filter 
        self.wf.migrated_image = self.laplacian_filter(self.wf.migrated_image)
        
        self.migratedFile = f"{self.pmt.migratedimageFolder}migrated_image_{self.pmt.approximation}_Nx{self.pmt.nx}_Nz{self.pmt.nz}.bin"
        self.wf.migrated_image.astype(np.float32).tofile(self.migratedFile)
        print(f"info: Final migrated image saved to {self.migratedFile}")

    def SolveBackwardWaveEquation(self):
        if self.pmt.migration == "onthefly":
            self.solveBackwardWaveEquationOntheFly()
        elif self.pmt.migration == "checkpoint":
            self.solveBackwardWaveEquationCheckpointing()
        elif self.pmt.migration == "SB":
            self.solveBackwardWaveEquationSavingBoundaries()
        elif self.pmt.migration == "RBC":
            self.solveBackwardWaveEquationRBC()
        else:
            raise ValueError("Unknown migration method. Choose 'onthefly','checkpoint' or 'SB'.")
        print(f"info: Migration solved")


