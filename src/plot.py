from Viewdata import plotting 
from survey import parameters

pmt = parameters("../inputs/Parameters.json")
plt = plotting(pmt)

# plt.viewModel(f"fwi_vp_acoustic_Nx851_Nz351_itr1.bin")
# plt.viewModel(f"fwi_vp_smooth_acoustic_Nx383_Nz141.bin")
# plt.viewSnapshot("acousticforward_shot_3_Nx201_Nz201_Nt8001_frame_","../inputs/diffractorvp_Nz201_Nx201.bin")
# plt.movieSnapshot(f"acousticforward_shot_20_Nx1701_Nz351_Nt8001_frame", f"../inputs/models/vp_marmousi-ii_shape_(2801, 13601)_dh10m_Nz351_Nx1701.bin", savegif = False)
# plt.viewSeismogram(f"/home/processamento/SeismicModeling2D/outputs/seismograms/seismogram_shot_40_Nt8001_Nrec169.bin", perc=99)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewImage(f"../outputs/images/migrated_image_acoustic_Nx851_Nz351.bin",laplacian=True,perc=99)
# plt.plotImageTrace(f"{pmt.migratedimageFolder}migrated_image_{pmt.approximation}_Nx{pmt.nx}_Nz{pmt.nz}.bin", f"../inputs/layer2vp_Nz{pmt.nz}_Nx{pmt.nx}.bin", laplacian = True, ix=None, perc=99)
# plt.movieImage(f"{pmt.approximation}shot_1_Nx{pmt.nx}_Nz{pmt.nz}_frame", f"../inputs/layer2vp_Nz301_Nx301.bin",laplacian = True, interval=200, savegif = False)

# import numpy as np

# model_smooth = np.fromfile("/home/juanmarques/workspace/SeismicModeling2D/inputs/fwi_vp_smooth_acoustic_Nx201_Nz201.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)
# model = np.fromfile("../inputs/fwi_vp_acoustic_Nx201_Nz201_itr1.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)

# dif = model_smooth - model

# import matplotlib.pyplot as plt

# plt.figure()
# plt.imshow(dif)
# plt.show()