from Viewdata import plotting 
from survey import parameters

pmt = parameters("../inputs/Parameters.json")
plt = plotting(pmt)

# plt.viewModel(f"layer2")
# plt.viewSnapshot(f"../inputs/layer2vp_Nz301_Nx301.bin",f"{pmt.snapshotFolder}{pmt.approximation}_shot_1_Nx{pmt.nx}_Nz{pmt.nz}_Nt{pmt.nt}_frame_2500.bin")
# plt.movieSnapshot(f"{pmt.approximation}_shot_1_Nx{pmt.nx}_Nz{pmt.nz}_Nt{pmt.nt}_frame_", f"../inputs/layer2vp_Nz301_Nx301.bin", savegif = False)
# plt.viewSeismogram(f"{pmt.seismogramFolder}seismogram_shot_1_Nt{pmt.nt}_Nrec{pmt.Nrec}.bin", perc=95)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewMigratedImage(f"{pmt.migratedimageFolder}migrated_image_{pmt.approximation}_Nx{pmt.nx}_Nz{pmt.nz}.bin",laplacian=True,perc=99)
# plt.plotImageTrace(f"{pmt.migratedimageFolder}migrated_image_{pmt.approximation}_Nx{pmt.nx}_Nz{pmt.nz}.bin", f"../inputs/layer2vp_Nz{pmt.nz}_Nx{pmt.nx}.bin", laplacian = True, ix=None, perc=99)


# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# def adjustColorBar(fig,ax,im):
#     # Create a divider for the existing axes instance
#     divider = make_axes_locatable(ax)
#     # Append an axes to the right of the current axes, with the same height
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = fig.colorbar(im,cax=cax)
#     return cbar

# nx = 301
# nz = 301

# snap = np.fromfile("../outputs/snapshots/acousticcerjan_shot_1_Nx301_Nz301_Nt8001_frame_2400forward.bin", dtype=np.float32).reshape(nz, nx)
# snapcheck = np.fromfile("../outputs/snapshots/acousticcerjan_shot_1_Nx301_Nz301_Nt8001_frame_2400checkpoint.bin", dtype=np.float32).reshape(nz, nx)
# snapSB = np.fromfile("../outputs/snapshots/acousticcerjan_shot_1_Nx301_Nz301_Nt8001_frame_2400SB.bin", dtype=np.float32).reshape(nz, nx)
# snapRBC = np.fromfile("../outputs/snapshots/acousticcerjan_shot_1_Nx301_Nz301_Nt8001_frame_2400RBC.bin", dtype=np.float32).reshape(nz, nx)
# imgonthefly = np.fromfile("/home/juanmarques/workspace/SeismicModeling2D/outputs/migrated_image/migrated_image_acoustic_Nx301_Nz301onthefly.bin", dtype=np.float32).reshape(nz, nx)
# img = np.fromfile("../outputs/migrated_image/migrated_image_acoustic_Nx301_Nz301.bin", dtype=np.float32).reshape(nz, nx)
# diff = imgonthefly - img

# plt.figure()
# plt.plot(imgonthefly[:, nx//2], label="validação")
# plt.plot(img[:, nx//2], label="SB")
# plt.plot(diff[:, nx//2], label="diff")
# plt.legend()

# fig, ax = plt.subplots()
# im = ax.imshow(imgonthefly)
# cbar = adjustColorBar(fig, ax, im)
# cbar.set_label("Amplitude")
# ax.set_title("Validação")

# fig, ax = plt.subplots()
# im = ax.imshow(img)
# cbar = adjustColorBar(fig, ax, im)
# cbar.set_label("Amplitude")
# ax.set_title("SB")

# fig, ax = plt.subplots()
# im = ax.imshow(diff)
# cbar = adjustColorBar(fig, ax, im)
# cbar.set_label("Amplitude")
# ax.set_title("Diferença")

# plt.show()

# diff = snap - snapRBC

# plt.figure()
# plt.plot(snap[:, nx//2], label="validação")
# plt.plot(snapRBC[:, nx//2], label="check")
# plt.plot(diff[:, nx//2], label="diff")
# plt.legend()

# fig, ax = plt.subplots()
# im = ax.imshow(snap)
# cbar = adjustColorBar(fig, ax, im)
# cbar.set_label("Amplitude")
# ax.set_title("Validação")

# fig, ax = plt.subplots()
# im = ax.imshow(snapRBC)
# cbar = adjustColorBar(fig, ax, im)
# cbar.set_label("Amplitude")
# ax.set_title("check")

# fig, ax = plt.subplots()
# im = ax.imshow(diff)
# cbar = adjustColorBar(fig, ax, im)
# cbar.set_label("Amplitude")
# ax.set_title("Diferença")

# plt.show()
