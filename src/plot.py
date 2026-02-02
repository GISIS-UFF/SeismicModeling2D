from Viewdata import plotting 

plt = plotting("../inputs/Parameters.json")

# plt.viewModel("layeredvp_Nz301_Nx301.bin")
# plt.viewSnapshot("acousticcerjan_shot_1_Nx301_Nz101_Nt4001_frame_1200.bin", "../inputs/layeredvp_Nz101_Nx301.bin")
# plt.movieSnapshot("acousticcerjan_shot_1_Nx301_Nz101_Nt4001_frame_", "../inputs/layeredvp_Nz101_Nx301.bin", savegif = True)
# plt.viewSeismogram("../outputs/seismograms/acousticCPML_seismogram_shot_1_Nt4001_Nrec501.bin", perc=95)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewMigratedImage("../outputs/migrated_image/migrated_image_acoustic_Nx301_Nz301.bin",perc=99)
# plt.viewSnapshotAnalyticalComparison(1,"../outputs/snapshots/TTI_CPML_shot_1_Nx501_Nz201_Nt6000_frame_1000.bin")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def adjustColorBar(fig,ax,im):
    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)
    # Append an axes to the right of the current axes, with the same height
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im,cax=cax)
    return cbar

nx = 301
nz = 301

# snaponthefly = np.fromfile("../outputs/snapshots/acousticcerjan_shot_1_Nx301_Nz101_Nt4001_frame_1200ONTHEFLY.bin", dtype=np.float32).reshape(nz, nx)
snap = np.fromfile("../outputs/snapshots/acousticCPML_shot_1_Nx301_Nz301_Nt4001_frame_1200FORWARD.bin", dtype=np.float32).reshape(nz, nx)
# snapcheck = np.fromfile("../outputs/snapshots/acousticCPML_shot_1_Nx301_Nz101_Nt4001_frame_1200CHECKPOINT.bin", dtype=np.float32).reshape(nz, nx)
snapSB = np.fromfile("../outputs/snapshots/acousticCPML_shot_1_Nx301_Nz301_Nt4001_frame_1200SB.bin", dtype=np.float32).reshape(nz, nx)
# snapRBC = np.fromfile("../outputs/snapshots/acousticCPML_shot_1_Nx301_Nz101_Nt4001_frame_1200RBC.bin", dtype=np.float32).reshape(nz, nx)

diff = snap - snapSB

plt.figure()
plt.plot(snap[:, nx//2], label="forward")
plt.plot(snapSB[:, nx//2], label="reconstruído")
plt.plot(diff[:, nx//2], label="diff")
plt.legend()

fig, ax = plt.subplots()
im = ax.imshow(snap)
cbar = adjustColorBar(fig, ax, im)
cbar.set_label("Amplitude")
ax.set_title("Forward")

fig, ax = plt.subplots()
im = ax.imshow(snapSB)
cbar = adjustColorBar(fig, ax, im)
cbar.set_label("Amplitude")
ax.set_title("Reconstruído")

fig, ax = plt.subplots()
im = ax.imshow(diff)
cbar = adjustColorBar(fig, ax, im)
cbar.set_label("Amplitude")
ax.set_title("Diferença")

plt.show()