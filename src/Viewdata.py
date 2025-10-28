import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import ricker
import pandas as pd
import json
import os

class Plotting:
    def __init__(self, parameters_path):
        self.parameters_path = parameters_path
        self.readParameters()
        self.readAcquisitionGeometry()

    def readParameters(self):
        with open(self.parameters_path) as f:
            self.parameters = json.load(f)
        
         # Approximation type
        self.approximation = self.parameters["approximation"]
        
        # Discretization self.parameters
        self.dx   = self.parameters["dx"]
        self.dz   = self.parameters["dz"]
        self.dt   = self.parameters["dt"]
        
        # Model size
        self.L    = self.parameters["L"]
        self.D    = self.parameters["D"]
        self.T    = self.parameters["T"]

        # Number of point for absorbing boundary condition
        self.N_abc = self.parameters["N_abc"]
        
        # Number of points in each direction
        self.nx = int(self.L/self.dx)+1
        self.nz = int(self.D/self.dz)+1
        self.nt = int(self.T/self.dt)+1

        self.nx_abc = self.nx + 2*self.N_abc
        self.nz_abc = self.nz + 2*self.N_abc

        # Define arrays for space and time
        self.x = np.linspace(0, self.L, self.nx)
        self.z = np.linspace(0, self.D, self.nz)
        self.t = np.linspace(0, self.T, self.nt)

        # Max frequency
        self.fcut = self.parameters["fcut"]

        # Output folders
        self.seismogramFolder = self.parameters["seismogramFolder"]
        self.migratedimageFolder = self.parameters["migratedimageFolder"]
        self.snapshotFolder = self.parameters["snapshotFolder"]
        self.modelFolder = self.parameters["modelFolder"]

        # Source and receiver files
        self.rec_file = self.parameters["rec_file"]
        self.src_file = self.parameters["src_file"]

        # Snapshot flag 
        self.shot_frame = self.parameters["shot_frame"] 

    def readAcquisitionGeometry(self):        
        # Read receiver and source coordinates from CSV files
        receiverTable = pd.read_csv(self.rec_file)
        print(f"info: Imported: {self.rec_file}")     
        sourceTable = pd.read_csv(self.src_file)
        print(f"info: Imported: {self.src_file}")

        # Read receiver and source coordinates
        self.rec_x = receiverTable['coordx'].to_numpy()
        self.rec_z = receiverTable['coordz'].to_numpy()
        self.shot_x = sourceTable['coordx'].to_numpy()
        self.shot_z = sourceTable['coordz'].to_numpy()

        self.Nrec = len(self.rec_x)
        self.Nshot = len(self.shot_x)   

    def adjustColorBar(self,fig,ax,im):
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append an axes to the right of the current axes, with the same height
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im,cax=cax)
        return cbar

    def viewSeismogram(self,filename, perc=99):
        sism = np.fromfile(filename, dtype=np.float32).reshape(self.nt,self.Nrec) 
        plt.figure(figsize=(5, 5))
        perc = np.percentile(sism, perc)
        plt.imshow(sism, aspect='auto', cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.Nrec, self.T, 0])
        plt.colorbar(label='Amplitude')
        plt.title("Seismogram")
        plt.ylabel("Time (s)")
        # plt.legend()
        plt.show()

    def viewModel(self,keyword):
        for filename in sorted(os.listdir(self.modelFolder)):
            if keyword in filename and filename.endswith(".bin"):
                path = os.path.join(self.modelFolder, filename)
                model = np.fromfile(path, dtype=np.float32).reshape(self.nx,self.nz).T
                fig, ax = plt.subplots(figsize=(10, 5))
                im = ax.imshow(model, aspect='equal', cmap='jet', extent=[0, self.L, self.D, 0])
                ax.plot(self.rec_x, self.rec_z, 'bv', markersize=2, label='Receivers')
                ax.plot(self.shot_x, self.shot_z, 'r*', markersize=5, label='Sources')
                ax.set_xlabel("Distance (m)")
                ax.set_ylabel("Depth (m)")
                
                # nice colorbar
                name = filename.lower()
                if "epsilon" in name:
                    label = "Epsilon"
                elif "delta" in name:
                    label = "Delta"
                elif "vs" in name:
                    label = "Shear velocity (m/s)"
                elif "theta" in name:
                    label = "Tilt angle (°)"
                elif "vp" in name:
                    label = "Velocity (m/s)"
                else:
                    label = "Amplitude"

                cbar = self.adjustColorBar(fig, ax, im)
                cbar.set_label(label)

                plt.tight_layout()
                plt.show()

    def viewSnapshot(self, keyword):
        for filename in sorted(os.listdir(self.snapshotFolder)):
            if keyword in filename and filename.endswith(".bin"):
                path = os.path.join(self.snapshotFolder, filename)
                snapshot = np.fromfile(path, dtype=np.float32).reshape(self.nz_abc, self.nx_abc)
                fig, ax = plt.subplots(figsize=(10, 5))
                im = ax.imshow(snapshot, aspect='equal', cmap='gray', extent=[0, self.L, self.D, 0])       
                # nice colorbar
                cbar = self.adjustColorBar(fig,ax,im)
                cbar.set_label("Amplitude")
                ax.set_xlabel("Distance (m)")
                ax.set_ylabel("Depth (m)")
                plt.show()

    def viewMigratedImage(self,filename,perc=99):
        migrated_image = np.fromfile(filename, dtype=np.float32).reshape(self.nz, self.nx)
        perc = np.percentile(migrated_image, perc)
        plt.imshow(migrated_image, cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.nx*self.dx, self.nz*self.dz, 0])  
        plt.colorbar(label='Amplitude')
        plt.title("Migrated Image")
        plt.xlabel("Distance (m)")
        plt.ylabel("Depth (m)")
        plt.show()

    def viewSourceWavelet(self):
        self.source = ricker(self.fcut, self.t)
        plt.figure()
        plt.plot(self.t, self.source)
        plt.title("Source Wavelet")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def viewSeismogramComparison(self,filename1, filename2, title="Seismogram Difference"):
        seismo1 = np.fromfile(filename1, dtype=np.float32).reshape(self.nt, self.Nrec)
        seismo2 = np.fromfile(filename2, dtype=np.float32).reshape(self.nt, self.Nrec)

        plt.figure(figsize=(5, 5))
        plt.plot(seismo1[:,self.Nrec//2], label = "sismograma 1")
        plt.plot(seismo2[:,self.Nrec//2], label = "sismograma 2")
        plt.legend()
        plt.ylabel("Time (s)")

        diff = seismo1 - seismo2

        perc = np.percentile(diff, 99)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(diff, aspect='auto', cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.Nrec, self.T, 0])

        ax.set_title(f"{title}")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (s)")
        ax.grid(True)

        cbar = self.adjustColorBar(fig, ax, im)
        cbar.set_label("Amplitude")

        plt.tight_layout()
        plt.show()
