from Viewdata import plotting 
from survey import parameters

pmt = parameters("../inputs/Parameters.json")
plt = plotting(pmt)

# plt.viewModel(f"fwi_vp_acoustic_Nx383_Nz141.bin")
# plt.viewModel(f"fwi_vp_smooth_acoustic_Nx383_Nz141.bin")
# plt.viewSnapshot(f"../inputs/layer2vp_Nz301_Nx301.bin",f"{pmt.approximation}backward_shot_1_Nx{pmt.nx}_Nz{pmt.nz}_Nt{pmt.nt}_frame_")
# plt.movieSnapshot(f"{pmt.approximation}forward_shot_1_Nx{pmt.nx}_Nz{pmt.nz}_Nt{pmt.nt}_frame_", f"../inputs/marmousi_vp_383x141.bin", savegif = False)
# plt.viewSeismogram(f"{pmt.seismogramFolder}seismogram_shot_20_Nt{pmt.nt}_Nrec{pmt.Nrec}.bin", perc=99)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewImage(f"../outputs/migrated_image/gradient_fwi_iter_1_acoustic_Nx383_Nz141.bin",laplacian=True,perc=99)
# plt.plotImageTrace(f"{pmt.migratedimageFolder}migrated_image_{pmt.approximation}_Nx{pmt.nx}_Nz{pmt.nz}.bin", f"../inputs/layer2vp_Nz{pmt.nz}_Nx{pmt.nx}.bin", laplacian = True, ix=None, perc=99)
# plt.movieImage(f"{pmt.approximation}_shot_1_Nx{pmt.nx}_Nz{pmt.nz}_frame_", f"../inputs/layer2vp_Nz301_Nx301.bin",laplacian = True, interval=200, savegif = False)

# import numpy as np

# model_smooth = np.fromfile("../inputs/fwi_vp_smooth_acoustic_Nx383_Nz141.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)
# model = np.fromfile("../inputs/fwi_vp_acoustic_Nx383_Nz141.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)

# dif = model_smooth - model

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(dif)
# plt.show()