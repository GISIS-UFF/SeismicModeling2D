import keyword
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import ricker
import pandas as pd
import json
import os
from matplotlib.animation import FuncAnimation
from utils import AnalyticalModel

class plotting:
    def __init__(self, parameters_path):
        self.parameters_path = parameters_path
        self.readParameters()
        self.readAcquisitionGeometry()

    def readParameters(self):
        with open(self.parameters_path) as f:
            self.parameters = json.load(f)
        
        # Approximation type
        self.approximation = self.parameters["approximation"]
        self.migration = self.parameters["migration"]
        self.ABC = self.parameters["ABC"]
        
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
        self.checkpointFolder = self.parameters["checkpointFolder"]

        # Source and receiver files
        self.rec_file = self.parameters["rec_file"]
        self.src_file = self.parameters["src_file"]

        # Velocity model file
        self.vpFile = self.parameters["vpFile"]
        self.vsFile = self.parameters["vsFile"]
        self.thetaFile = self.parameters["thetaFile"]

        # Snapshot flag
        self.snap = self.parameters["snap"]
        self.step = self.parameters["step"]
        self.last_save = self.parameters["last_save"]

        # Anisotropy parameters files
        self.epsilonFile = self.parameters["epsilonFile"]  
        self.deltaFile   = self.parameters["deltaFile"]  

        #Anisotropy parameters for Layered model and diffractor model
        self.vp1 = self.parameters["vp1"]
        self.vp2 = self.parameters["vp2"]
        self.theta1 = self.parameters["theta1"]
        self.theta2 = self.parameters["theta2"]
        self.epsilon1 = self.parameters["epsilon1"]
        self.epsilon2 = self.parameters["epsilon2"]
        self.delta1   = self.parameters["delta1"]
        self.delta2  = self.parameters["delta2"]
        self.diffractor = self.parameters["diffractor"]

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

    def get_shot_frame(self,filename):
        name = filename.replace(".bin", "")
        parts = name.split("_")

        shot = None
        frame = None

        for i, p in enumerate(parts):
            if p == "shot":
                shot = int(parts[i + 1])
            elif p == "frame":
                frame = int(parts[i + 1])

        return shot, frame

    def viewSnapshot(self, keyword_snap, path_model):
        perc = 1e-8

        model = np.fromfile(path_model, dtype=np.float32).reshape(self.nx, self.nz).T

        files = []
        for file in os.listdir(self.snapshotFolder):
            if file.endswith(".bin") and keyword_snap in file:
                shot, frame = self.get_shot_frame(file)
                if shot is not None and frame is not None:
                    files.append((shot, frame, file))

        # ordena por shot e depois por frame
        files.sort(key=lambda x: (x[0], x[1]))

        for shot, frame, filename in files:
            path_snap = os.path.join(self.snapshotFolder, filename)

            snapshot = np.fromfile(path_snap, dtype=np.float32).reshape(self.nz, self.nx)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(model,cmap="jet",aspect="equal",extent=[0, self.L, self.D, 0])

            im = ax.imshow(snapshot,cmap="gray",aspect="equal",extent=[0, self.L, self.D, 0],vmin=-perc,vmax=perc,alpha=0.4)
            ax.plot(self.rec_x, self.rec_z, 'bv', markersize=2, label='Receivers')
            ax.plot(self.shot_x, self.shot_z, 'r*', markersize=5, label='Sources')
            ax.set_title(f"Shot {shot} | Frame {frame}")
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Depth (m)")

            cbar = self.adjustColorBar(fig, ax, im)
            cbar.set_label("Amplitude")

            plt.tight_layout()
            plt.show()

    def movieSnapshot(self, keyword_snap, path_model, interval=200, savegif = False):
        perc = 1e-8

        snap_files = []
        for filename in os.listdir(self.snapshotFolder):
            if filename.endswith(".bin") and keyword_snap in filename:
                shot, frame = self.get_shot_frame(filename)
                snap_files.append((shot, frame, filename))

        snap_files.sort(key=lambda x: (x[0], x[1]))

        model = np.fromfile(path_model, dtype=np.float32).reshape(self.nx, self.nz).T

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(model, cmap="jet", aspect="equal", extent=[0, self.L, self.D, 0])

        first_file = snap_files[0][2]
        snap0 = np.fromfile(os.path.join(self.snapshotFolder, first_file),dtype=np.float32).reshape(self.nz, self.nx)

        im = ax.imshow(snap0,cmap="gray",aspect="equal",extent=[0, self.L, self.D, 0],vmin=-perc,vmax=perc,alpha=0.4)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")

        def update(i):
            filename = snap_files[i][2]
            snapshot = np.fromfile(os.path.join(self.snapshotFolder, filename),dtype=np.float32).reshape(self.nz, self.nx)
            im.set_data(snapshot)
            return [im]

        ani = FuncAnimation(fig,update,frames=len(snap_files),interval=interval,blit=True)

        if savegif == True:
            gif_path = os.path.join(self.snapshotFolder, "snapshots.gif")
            ani.save(gif_path, writer="pillow", fps=1000/interval)

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

    def viewSeismogramComparison(self,perc, offset, filename1, filename2, title="Seismogram Difference"):
        seismo1 = np.fromfile(filename1, dtype=np.float32).reshape(self.nt, self.Nrec)
        seismo2 = np.fromfile(filename2, dtype=np.float32).reshape(self.nt, self.Nrec)

        # Calcula offsets para este tiro
        offsets = np.abs(self.rec_x - self.shot_x[0])
        
        # Encontra os receptores mais próximos das distâncias desejadas
        rec_idx = np.argmin(np.abs(offsets - offset))

        plt.figure(figsize=(5, 5))
        plt.plot(self.t,seismo1[:,rec_idx], label = "VTI")
        plt.plot(self.t,seismo2[:,rec_idx], label = "VTI NEW")
        plt.legend()
        plt.ylabel("Amplitude")
        plt.title(f"Offset: {offsets[rec_idx]}m")

        diff = seismo1 - seismo2

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(diff, aspect='auto', cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.Nrec, self.T, 0])
        ax.set_title(title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (s)")
        ax.grid(True)
        cbar = self.adjustColorBar(fig, ax, im)
        cbar.set_label("Amplitude")

        plt.show()

    def viewSnapshotAnalyticalComparison(self,frame,filename):
        snapshot = np.fromfile(filename, dtype=np.float32).reshape(self.nz_abc, self.nx_abc)
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot snapshot
        im = ax.imshow(snapshot[self.N_abc:self.N_abc + self.nz, self.N_abc:self.N_abc + self.nx], aspect='equal', cmap='gray', extent=[0, self.L, self.D, 0])
        ax.plot(self.rec_x, self.rec_z, 'bv', markersize=2, label='Receivers')
        ax.plot(self.shot_x, self.shot_z, 'r*', markersize=5, label='Sources')
        
        # Compute the analytical wavefront
        self.vp = np.zeros([self.nz_abc,self.nx_abc],dtype=np.float32)
        self.vp[0:self.nz//2, :] = self.vpLayer1
        self.vp[self.nz//2:self.nz, :] = self.vpLayer2
        vel = self.vp[ int(self.shot_z[0]/self.dz), int(self.shot_x[0]/self.dx) ]
        Rp = AnalyticalModel(vel, self.epsilonLayer1, self.deltaLayer1, self.dt, self.fcut, self.frame[frame])

        # Source coordinates
        x0 = self.shot_x[0]
        z0 = self.shot_z[0]
        
        # coordenates of the analytical wavefront
        theta = np.linspace(0, 2*np.pi, 500)
        x_rp = x0 + Rp * np.sin(theta)
        z_rp = z0 + Rp * np.cos(theta)

        if self.approximation in ["acousticTTI", "acousticTTICPML"]:
            angle = -60  
            angle_rad = np.radians(angle)

            x_shifted = x_rp - x0
            z_shifted = z_rp - z0

            x_rot = x_shifted * np.cos(angle_rad) - z_shifted * np.sin(angle_rad)
            z_rot = x_shifted * np.sin(angle_rad) + z_shifted * np.cos(angle_rad)

            x = x0 + x_rot
            z = z0 + z_rot

        else:
            x = x_rp
            z = z_rp

        # Plot the analytical wavefront    
        ax.plot(x, z, 'r', label='Analytical wavefront')
        ax.legend()
        ax.set_title(f"Snapshot at time step {self.frame[frame]} (shot {1})")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True)
        plt.tight_layout()

        if self.approximation in ["acoustic", "acousticCPML"]:
            plt.savefig(f"{self.snapshotFolder}SnapshotAnalyticalComparison_acoustic_{frame}_shot{1}.png")
        if self.approximation in ["acousticVTI", "acousticVTICPML"]:
            plt.savefig(f"{self.snapshotFolder}SnapshotAnalyticalComparison_acousticVTI_{frame}_shot{1}.png")
        if self.approximation in ["acousticTTI", "acousticTTICPML"]:
            plt.savefig(f"{self.snapshotFolder}SnapshotAnalyticalComparison_acousticTTI_{frame}_shot{1}.png")

        plt.show()