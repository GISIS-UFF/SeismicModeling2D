import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import AnalyticalModel
from AcousticSeismicModeling2D import wavefield 

class Plotting:
    def __init__(self, wf: "wavefield"):
        self.wf = wf

    def adjustColorBar(self,fig,ax,im):
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append an axes to the right of the current axes, with the same height
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im,cax=cax)
        return cbar

    def viewSeismogram(self,perc=99):
        sism = np.fromfile(self.wf.viewSeismogramFile, dtype=np.float32).reshape(6001,235) 
        plt.figure(figsize=(5, 5))
        perc = np.percentile(sism, perc)
        plt.imshow(sism, aspect='auto', cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.wf.Nrec, self.wf.T, 0])
        plt.colorbar(label='Amplitude')
        plt.title("Seismogram")
        plt.ylabel("Time (s)")
        # plt.legend()
        plt.show()

    def viewModel(self):
        model = np.fromfile(self.wf.viewModelFile, dtype=np.float32).reshape(self.wf.nx,self.wf.nz).T
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(model, aspect='equal', cmap='jet', extent=[0, self.wf.L, self.wf.D, 0])
        ax.plot(self.wf.rec_x, self.wf.rec_z, 'bv', markersize=2, label='Receivers')
        ax.plot(self.wf.shot_x, self.wf.shot_z, 'r*', markersize=5, label='Sources')
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True)
        
        # nice colorbar
        cbar = Plotting.adjustColorBar(fig,ax,im)
        cbar.set_label()    
        plt.show()

    def viewSnapshot(self):
        snapshot = np.fromfile(self.wf.viewSnapshotFile, dtype=np.float32).reshape(self.wf.nz, self.wf.nx)
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(snapshot, aspect='equal', cmap='gray', extent=[0, self.wf.L, self.wf.D, 0])       
        # nice colorbar
        cbar = Plotting.adjustColorBar(fig,ax,im)
        cbar.set_label("Amplitude")
        
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        plt.show()

    def viewMigratedImage(self,perc=99):
        migrated_image = np.fromfile(self.wf.viewMigratedImageFile, dtype=np.float32).reshape(self.wf.nz, self.wf.nx)
        perc = np.percentile(migrated_image, perc)
        plt.imshow(migrated_image, cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.wf.nx*self.wf.dx, self.wf.nz*self.wf.dz, 0])  
        plt.colorbar(label='Amplitude')
        plt.title("Migrated Image")
        plt.xlabel("Distance (m)")
        plt.ylabel("Depth (m)")
        plt.show()

    def viewSourceWavelet(self):
        plt.figure()
        plt.plot(self.wf.t, self.wf.source)
        plt.title("Source Wavelet")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def viewSnapshotAnalyticalComparison(self,k):
        snapshot = np.fromfile(self.wf.viewSnapshotFile, dtype=np.float32).reshape(self.wf.nz, self.wf.nx)
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot snapshot
        im = ax.imshow(snapshot[k, self.wf.N_abc:-self.wf.N_abc, self.wf.N_abc:-self.wf.N_abc], aspect='equal', cmap='gray', extent=[0, self.wf.L, self.wf.D, 0])
        ax.plot(self.wf.rec_x, self.wf.rec_z, 'bv', markersize=2, label='Receivers')
        ax.plot(self.wf.shot_x, self.wf.shot_z, 'r*', markersize=5, label='Sources')
        
        # Compute the analytical wavefront
        if self.wf.approximation == "acoustic":
            vel = self.wf.vp[ int(self.wf.shot_z[0]/self.wf.dz), int(self.wf.shot_x[0]/self.wf.dx) ]
            Rp = AnalyticalModel(vel, 0, 0, self.wf.dt, self.wf.fcut, self.wf.frame[0])
        elif self.wf.approximation in ["acousticVTI", "acousticTTI", "acousticVTICPML", "acousticTTICPML"]:
            vel = self.wf.vp[ int(self.wf.shot_z[0]/self.wf.dz), int(self.wf.shot_x[0]/self.wf.dx) ]
            Rp = AnalyticalModel(vel, 0.2, 0.2, self.wf.dt, self.wf.fcut, self.wf.frame[0])
        else:
            raise ValueError("Info: Unknown approximation.")

        # Source coordinates
        x0 = self.wf.shot_x[0]
        z0 = self.wf.shot_z[0]
        
        # coordenates of the analytical wavefront
        theta = np.linspace(0, 2*np.pi, 500)
        x_rp = x0 + Rp * np.sin(theta)
        z_rp = z0 + Rp * np.cos(theta)

        if self.wf.approximation == "acousticTTI":
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
        ax.set_title(f"Snapshot at time step {self.wf.frame[0]} (shot {1})")

        # nice colorbar
        cbar = Plotting.adjustColorBar(fig, ax, im)
        cbar.set_label("Amplitude")

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.grid(True)
        plt.tight_layout()

        if self.wf.approximation == "acoustic":
            plt.savefig(f"{self.wf.snapshotFolder}SnapshotAnalyticalComparison_acoustic_{0}_shot{1}.png")
        if self.wf.approximation == "acousticVTI":
            plt.savefig(f"{self.wf.snapshotFolder}SnapshotAnalyticalComparison_acousticVTI_{0}_shot{1}.png")
        if self.wf.approximation == "acousticTTI":
            plt.savefig(f"{self.wf.snapshotFolder}SnapshotAnalyticalComparison_acousticTTI_{0}_shot{1}.png")

        plt.show()

    def viewSeismogramComparison(self,filename1, filename2, title="Seismogram Difference"):
        seismo1 = np.fromfile(filename1, dtype=np.float32).reshape(self.wf.nt, self.wf.Nrec)
        seismo2 = np.fromfile(filename2, dtype=np.float32).reshape(self.wf.nt, self.wf.Nrec)

        diff = seismo1 - seismo2

        perc = np.percentile(diff, 99)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(diff, aspect='auto', cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.wf.Nrec, self.wf.T, 0])

        ax.set_title(f"{title}")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (s)")
        ax.grid(True)

        cbar = Plotting.adjustColorBar(fig, ax, im)
        cbar.set_label("Amplitude")

        plt.tight_layout()
        plt.savefig(f"{title}.png")
        plt.show()
