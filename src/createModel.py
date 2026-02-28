import numpy as np

class model:
    def __init__(self, parameters, wavefield):
        self.pmt = parameters
        self.wf = wavefield

        self.vp1 = 1500.0 
        self.vp2 = 2000.0
        self.vp3 = 00.0

        self.epsilon1 = 0.0
        self.epsilon2 = 0.0
        self.epsilon3 = 0.0

        self.delta1 = 0.0
        self.delta2 = 0.0 
        self.delta3 = 0.0

        self.theta1 = 0.0
        self.theta2 = 0.0
        self.theta3 = 0.0

    def create2LayerModel(self,v1,v2,e1,e2,d1,d2,t1,t2):
        self.wf.vp[:self.pmt.nz//2, :] = v1
        self.wf.vp[self.pmt.nz//2:self.pmt.nz, :] = v2
        self.modelFile = f"{self.pmt.modelFolder}layer2vp_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
        self.wf.vp.T.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")

        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon[:self.pmt.nz//2, :] = e1
            self.wf.epsilon[self.pmt.nz//2:self.pmt.nz, :] = e2
            self.modelFile = f"{self.pmt.modelFolder}layer2epsilon_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.epsilon.T.tofile(self.modelFile)
            print(f"info: Epsilon saved to {self.modelFile}")

            self.wf.delta[:self.pmt.nz//2, :] = d1
            self.wf.delta[self.pmt.nz//2:self.pmt.nz, :] = d2
            self.modelFile = f"{self.pmt.modelFolder}layer2delta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.delta.T.tofile(self.modelFile)
            print(f"info: Delta saved to {self.modelFile}")
        
        if self.pmt.approximation == "TTI":
            vs1 = np.sqrt(v1*v1*(e1 - d1)/0.8)
            vs2 = np.sqrt(v2*v2*(e2 - d2)/0.8)
            self.wf.vs[:self.pmt.nz//2, :] = vs1
            self.wf.vs[self.pmt.nz//2:self.pmt.nz, :] = vs2
            self.modelFile = f"{self.pmt.modelFolder}layer2vs_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.vs.T.tofile(self.modelFile)
            print(f"info: Vs saved to {self.modelFile}")

            self.wf.theta[:self.pmt.nz//2, :] = t1
            self.wf.theta[self.pmt.nz//2:self.pmt.nz, :] = t2
            self.modelFile = f"{self.pmt.modelFolder}layer2theta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.theta.T.tofile(self.modelFile)
            print(f"info: Theta saved to {self.modelFile}")
        
    
    def create3LayerModel(self,v1, v2, v3,e1,e2,e3,d1,d2,d3,t1,t2,t3):
        self.wf.vp[:self.pmt.nz//3, :] = v1
        self.wf.vp[self.pmt.nz//3:2*self.pmt.nz//3, :] = v2
        self.wf.vp[2*self.pmt.nz//3:, :] = v3
        self.modelFile = f"{self.pmt.modelFolder}layer3vp_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
        self.wf.vp.T.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")
        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon[:self.pmt.nz//3, :] = e1
            self.wf.epsilon[self.pmt.nz//3:2*self.pmt.nz//3, :] = e2
            self.wf.epsilon[2*self.pmt.nz//3:, :] = e3
            self.modelFile = f"{self.pmt.modelFolder}layer3epsilon_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.epsilon.T.tofile(self.modelFile)
            print(f"info: Epsilon saved to {self.modelFile}")

            self.wf.delta[:self.pmt.nz//3, :] = d1
            self.wf.delta[self.pmt.nz//3:2*self.pmt.nz//3, :] = d2
            self.wf.delta[2*self.pmt.nz//3:, :] = d3
            self.modelFile = f"{self.pmt.modelFolder}layer3delta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.delta.T.tofile(self.modelFile)
            print(f"info: Delta saved to {self.modelFile}")
        
        if self.pmt.approximation == "TTI":
            vs1 = np.sqrt(v1*v1*(e1 - d1)/0.8)
            vs2 = np.sqrt(v2*v2*(e2 - d2)/0.8)
            vs3 = np.sqrt(v3*v3*(e3 - d3)/0.8)
            self.wf.vs[:self.pmt.nz//3, :] = vs1
            self.wf.vs[self.pmt.nz//3:2*self.pmt.nz//3, :] = vs2
            self.wf.vs[2*self.pmt.nz//3:, :] = vs3
            self.modelFile = f"{self.pmt.modelFolder}layer3vs_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.vs.T.tofile(self.modelFile)
            print(f"info: Vs saved to {self.modelFile}")

            self.wf.theta[:self.pmt.nz//3, :] = t1
            self.wf.theta[self.pmt.nz//3:2*self.pmt.nz//3, :] = t2
            self.wf.theta[2*self.pmt.nz//3:, :] = t3
            self.modelFile = f"{self.pmt.modelFolder}layer3theta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.theta.T.tofile(self.modelFile)
            print(f"info: Theta saved to {self.modelFile}")
    
    def createDiffractorModel(self,v1,v2,e1,e2,d1,d2,t1,t2):
        self.wf.vp[:, :] = v1
        self.wf.vp[self.pmt.nz // 2, self.pmt.nx // 2] = v2
        self.modelFile = f"{self.pmt.modelFolder}diffractorvp_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
        self.wf.vp.T.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")

        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon[:, :] = e1
            self.wf.epsilon[self.pmt.nz // 2, self.pmt.nx // 2] = e2
            self.modelFile = f"{self.pmt.modelFolder}diffractorepsilon_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.epsilon.T.tofile(self.modelFile)
            print(f"info: Epsilon saved to {self.modelFile}")

            self.wf.delta[:, :] = d1
            self.wf.delta[self.pmt.nz // 2, self.pmt.nx // 2] = d2
            self.modelFile = f"{self.pmt.modelFolder}diffractordelta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.delta.T.tofile(self.modelFile)
            print(f"info: Delta saved to {self.modelFile}")
            
        if self.pmt.approximation == "TTI":
            vs1 = np.sqrt(v1*v1*(e1 - d1)/0.8)
            vs2 = np.sqrt(v2*v2*(e2 - d2)/0.8)
            self.wf.vs[:, :] = vs1
            self.wf.vs[self.pmt.nz // 2, self.pmt.nx // 2] = vs2
            self.modelFile = f"{self.pmt.modelFolder}diffractorvs_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.vs.T.tofile(self.modelFile)
            print(f"info: Vs saved to {self.modelFile}")

            self.wf.theta[:, :] = t1
            self.wf.theta[self.pmt.nz // 2, self.pmt.nx // 2] = t2
            self.modelFile = f"{self.pmt.modelFolder}diffractortheta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.theta.T.tofile(self.modelFile)
            print(f"info: Theta saved to {self.modelFile}")
    
    def createGradientModel(self,v1,e1,d1,t1):
        self.wf.vp[:, :] = v1
        alpha = 0.7
        for iz in range(self.pmt.nz):
            self.wf.vp[iz,:] = v1 + alpha*self.pmt.z[iz]

        self.modelFile = f"{self.pmt.modelFolder}gradientvp_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
        self.wf.vp.T.tofile(self.modelFile)
        print(f"info: Vp saved to {self.modelFile}")

        if self.pmt.approximation in ["VTI", "TTI"]:
            self.wf.epsilon[:, :] = e1
            for iz in range(self.pmt.nz):
                self.wf.epsilon[iz,:] = e1 + alpha*self.pmt.z[iz]
            self.modelFile = f"{self.pmt.modelFolder}gradientepsilon_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.epsilon.T.tofile(self.modelFile)
            print(f"info: Epsilon saved to {self.modelFile}")

            self.wf.delta[:, :] = d1
            for iz in range(self.pmt.nz):
                self.wf.delta[iz,:] = d1 + alpha*self.pmt.z[iz]
            self.modelFile = f"{self.pmt.modelFolder}gradientdelta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.delta.T.tofile(self.modelFile)
            print(f"info: Delta saved to {self.modelFile}")
            
        if self.pmt.approximation == "TTI":
            vs1 = np.sqrt(v1*v1*(e1 - d1)/0.8)
            self.wf.vs[:, :] = vs1
            for iz in range(self.pmt.nz):
                self.wf.vs[iz,:] = vs1 + alpha*self.pmt.z[iz]
            self.modelFile = f"{self.pmt.modelFolder}gradientvs_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.vs.T.tofile(self.modelFile)
            print(f"info: Vs saved to {self.modelFile}")

            self.wf.theta[:, :] = t1
            for iz in range(self.pmt.nz):
                self.wf.theta[iz,:] = t1 + alpha*self.pmt.z[iz]
            self.modelFile = f"{self.pmt.modelFolder}gradienttheta_Nz{self.pmt.nz}_Nx{self.pmt.nx}.bin"
            self.wf.theta.T.tofile(self.modelFile)
            print(f"info: Theta saved to {self.modelFile}")

    def createModelFromVp(self):
        if not self.pmt.approximation in ["VTI", "TTI"]:
            raise ValueError("ERROR: Change approximation parameter to 'VTI'or 'TTI'.")
        
        if self.pmt.vpFile == None:
            raise ValueError("ERROR: Import or create a velocity model first.")
            
        idx_water = np.where(self.wf.vp <= 1500)

        # create density model with Gardner's equation
        self.rho = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)
        a, b = 0.23, 0.25
        self.rho = a * np.power(self.wf.vp/0.3048,b)*1000 # Gardner relation - Rosa (2010) apud Gardner et al. (1974) pag. 496 rho = a * v^b
        self.rho[idx_water] = 1000.0 # water density

        # create epsilon model epsilon = 0.25 rho - 0.3 - Petrov et al. (2021) 
        self.wf.epsilon = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)
        self.wf.epsilon = 0.25 * self.rho/1000 - 0.3 # rho in g/cm3
        self.wf.epsilon[idx_water] = 0.0 # water epsilon
        self.wf.epsilon.T.tofile(self.pmt.vpFile.replace(".bin","_epsilon.bin"))	
        print(f"info: Epsilon model saved to {self.pmt.vpFile.replace('.bin','_epsilon.bin')}")

        # create delta model delta = 0.125 rho - 0.1 - Petrov et al. (2021)
        self.wf.delta = np.zeros([self.pmt.nz,self.pmt.nx],dtype=np.float32)
        self.wf.delta = 0.125 * self.rho/1000 - 0.1 # rho in g/cm3
        self.wf.delta[idx_water] = 0.0 # water delta
        self.wf.delta.T.tofile(self.pmt.vpFile.replace(".bin","_delta.bin"))
        print(f"info: Delta model saved to {self.pmt.vpFile.replace('.bin','_delta.bin')}")

        #create vs model
        self.wf.vs = np.zeros([self.pmt.nz,self.pmt.nx], dtype=np.float32)
        self.wf.vs = np.sqrt(self.wf.vp*self.wf.vp*(self.wf.epsilon - self.wf.delta)/0.8)
        self.wf.vs[idx_water] = 0.0
        self.wf.vs.T.tofile(self.pmt.vpFile.replace(".bin","_vs.bin"))	
        print(f"info: Vs model saved to {self.pmt.vpFile.replace('.bin','_vs.bin')}")
    
    def buildModel(self):
        if self.pmt.layer2 == True:
            self.create2LayerModel(self.vp1,self.vp2,self.epsilon1,self.epsilon2,self.delta1,self.delta2,self.theta1,self.theta2)
        elif self.pmt.layer3 == True:
            self.create3LayerModel(self.vp1,self.vp2,self.vp3,self.epsilon1,self.epsilon2,self.epsilon3,self.delta1,self.delta2,self.delta3,self.theta1,self.theta2,self.theta3)
        elif self.pmt.diffractor == True:
            self.createDiffractorModel(self.vp1,self.vp2,self.epsilon1,self.epsilon2,self.delta1,self.delta2,self.theta1,self.theta2)
        elif self.pmt.gradient == True:
            self.createGradientModel(self.vp1,self.epsilon1,self.delta1,self.theta1)
        elif self.pmt.modelfromvp == True:
            self.createModelFromVp()
        else:
            raise ValueError(f"ERROR: Unknwon synthetic model.")
        
if __name__ == "__main__":
    from survey import parameters
    from Modeling2D import wavefield

    pmt = parameters("../inputs/Parameters.json")

    wf = wavefield(pmt)
    wf.initializeWavefields()

    model = model(pmt,wf)
    model.buildModel()      
 