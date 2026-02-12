import keyword
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import ricker
import pandas as pd
import json
import os
from matplotlib.animation import FuncAnimation

class plotting:
    def __init__(self, parameters):
        self.pmt = parameters

    def adjustColorBar(self,fig,ax,im):
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append an axes to the right of the current axes, with the same height
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im,cax=cax)
        return cbar

    def viewSeismogram(self,filename, perc=99):
        sism = np.fromfile(filename, dtype=np.float32).reshape(self.pmt.nt,self.pmt.Nrec) 
        plt.figure(figsize=(5, 5))
        perc = np.percentile(sism, perc)
        plt.imshow(sism, aspect='auto', cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.pmt.Nrec, self.pmt.T, 0])
        plt.colorbar(label='Amplitude')
        plt.title("Seismogram")
        plt.ylabel("Time (s)")
        plt.show()

    def viewModel(self,keyword):
        for filename in sorted(os.listdir(self.pmt.modelFolder)):
            if keyword in filename and filename.endswith(".bin"):
                path = os.path.join(self.pmt.modelFolder, filename)
                model = np.fromfile(path, dtype=np.float32).reshape(self.pmt.nx,self.pmt.nz).T
                fig, ax = plt.subplots(figsize=(10, 5))
                im = ax.imshow(model, aspect='equal', cmap='jet', extent=[0, self.pmt.L, self.pmt.D, 0])
                ax.plot(self.pmt.rec_x, self.pmt.rec_z, 'bv', markersize=2, label='Receivers')
                ax.plot(self.pmt.shot_x, self.pmt.shot_z, 'r*', markersize=5, label='Sources')
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

        model = np.fromfile(path_model, dtype=np.float32).reshape(self.pmt.nx, self.pmt.nz).T

        files = []
        for file in os.listdir(self.pmt.snapshotFolder):
            if file.endswith(".bin") and keyword_snap in file:
                shot, frame = self.get_shot_frame(file)
                if shot is not None and frame is not None:
                    files.append((shot, frame, file))

        # ordena por shot e depois por frame
        files.sort(key=lambda x: (x[0], x[1]))

        for shot, frame, filename in files:
            path_snap = os.path.join(self.pmt.snapshotFolder, filename)

            snapshot = np.fromfile(path_snap, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(model,cmap="jet",aspect="equal",extent=[0, self.pmt.L, self.pmt.D, 0])

            im = ax.imshow(snapshot,cmap="gray",aspect="equal",extent=[0, self.pmt.L, self.pmt.D, 0],vmin=-perc,vmax=perc,alpha=0.4)
            ax.plot(self.pmt.rec_x, self.pmt.rec_z, 'bv', markersize=2, label='Receivers')
            ax.plot(self.pmt.shot_x, self.pmt.shot_z, 'r*', markersize=5, label='Sources')
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
        for filename in os.listdir(self.pmt.snapshotFolder):
            if filename.endswith(".bin") and keyword_snap in filename:
                shot, frame = self.get_shot_frame(filename)
                snap_files.append((shot, frame, filename))

        snap_files.sort(key=lambda x: (x[0], x[1]))

        model = np.fromfile(path_model, dtype=np.float32).reshape(self.pmt.nx, self.pmt.nz).T

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(model, cmap="jet", aspect="equal", extent=[0, self.pmt.L, self.pmt.D, 0])

        first_file = snap_files[0][2]
        snap0 = np.fromfile(os.path.join(self.pmt.snapshotFolder, first_file),dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)

        im = ax.imshow(snap0,cmap="gray",aspect="equal",extent=[0, self.pmt.L, self.pmt.D, 0],vmin=-perc,vmax=perc,alpha=0.4)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")

        def update(i):
            filename = snap_files[i][2]
            snapshot = np.fromfile(os.path.join(self.pmt.snapshotFolder, filename),dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
            im.set_data(snapshot)
            return [im]

        ani = FuncAnimation(fig,update,frames=len(snap_files),interval=interval,blit=True)

        if savegif == True:
            gif_path = os.path.join(self.pmt.snapshotFolder, "snapshots.gif")
            ani.save(gif_path, writer="pillow", fps=1000/interval)

        plt.show()

    def viewMigratedImage(self,filename,perc=99):
        migrated_image = np.fromfile(filename, dtype=np.float32).reshape(self.pmt.nz, self.pmt.nx)
        perc = np.percentile(migrated_image, perc)
        plt.imshow(migrated_image, cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.pmt.nx*self.pmt.dx, self.pmt.nz*self.pmt.dz, 0])  
        plt.colorbar(label='Amplitude')
        plt.title("Migrated Image")
        plt.xlabel("Distance (m)")
        plt.ylabel("Depth (m)")
        plt.show()

    def viewSourceWavelet(self):
        self.source = ricker(self.pmt.fcut, self.pmt.T)
        plt.figure()
        plt.plot(self.pmt.T, self.source)
        plt.title("Source Wavelet")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def viewSeismogramComparison(self,perc, offset, filename1, filename2, title="Seismogram Difference"):
        seismo1 = np.fromfile(filename1, dtype=np.float32).reshape(self.pmt.nt, self.pmt.Nrec)
        seismo2 = np.fromfile(filename2, dtype=np.float32).reshape(self.pmt.nt, self.pmt.Nrec)

        # Calcula offsets para este tiro
        offsets = np.abs(self.pmt.rec_x - self.pmt.shot_x[0])
        
        # Encontra os receptores mais próximos das distâncias desejadas
        rec_idx = np.argmin(np.abs(offsets - offset))

        plt.figure(figsize=(5, 5))
        plt.plot(self.pmt.T,seismo1[:,rec_idx], label = "VTI")
        plt.plot(self.pmt.T,seismo2[:,rec_idx], label = "VTI NEW")
        plt.legend()
        plt.ylabel("Amplitude")
        plt.title(f"Offset: {offsets[rec_idx]}m")

        diff = seismo1 - seismo2

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(diff, aspect='auto', cmap='gray', vmin=-perc, vmax=perc, extent=[0, self.pmt.Nrec, self.pmt.T, 0])
        ax.set_title(title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (s)")
        ax.grid(True)
        cbar = self.adjustColorBar(fig, ax, im)
        cbar.set_label("Amplitude")

        plt.show()
