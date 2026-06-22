from Viewdata import plotting 
from survey import parameters

pmt = parameters("../inputs/Parameters.json")
plt = plotting(pmt)

# plt.viewModel(f"/home/juanmarques/workspace/SeismicModeling2D/inputs/models/fwi_epsilon_smooth_VTI_Nx751_Nz451.bin")
# plt.viewHistory()
# plt.viewSnapshot("acousticforward_shot_3_Nx201_Nz201_Nt8001_frame_","../inputs/diffractorvp_Nz201_Nx201.bin")
# plt.movieSnapshot(f"acousticforward_shot_10_Nx681_Nz141_Nt5001_frame", f"/home/juanmarques/workspace/SeismicModeling2D/inputs/models/diffractorvp_Nz141_Nx681.bin", savegif = True)
# plt.viewSeismogram(f"../outputs/seismograms/seismogram_shot_15_Nt2001_Nrec170_fcut9.0.bin", perc=99)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewImage(f"../outputs/gradients/delta_gradient_fwi_iter_1_VTI_Nx751_Nz451_freq9.0.bin",laplacian=True,perc=99)
# plt.plotImageTrace(f"{pmt.migratedimageFolder}migrated_image_{pmt.approximation}_Nx{pmt.nx}_Nz{pmt.nz}.bin", f"../inputs/layer2vp_Nz{pmt.nz}_Nx{pmt.nx}.bin", laplacian = True, ix=None, perc=99)
# plt.movieImage(f"{pmt.approximation}shot_1_Nx{pmt.nx}_Nz{pmt.nz}_frame", f"../inputs/layer2vp_Nz301_Nx301.bin",laplacian = True, interval=200, savegif = False)

import numpy as np
import matplotlib.pyplot as plt

model_smooth = np.fromfile("/home/juanmarques/workspace/SeismicModeling2D/inputs/models/fwi_delta_smooth_VTI_Nx751_Nz451.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)
model_ref = np.fromfile("/home/juanmarques/workspace/SeismicModeling2D/inputs/models/bptti2007/Delta_Model_shape_(1801, 12596)_dh25m_Nz451_Nx751.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)
# model_fwi = np.fromfile("../outputs/estimated_models/fwi_vp_acoustic_Nx681_Nz141_itr25_freq30.0.bin", dtype=np.float32).reshape(pmt.nz,pmt.nx)

D = np.linspace(0, pmt.nz * pmt.dz, pmt.nz, endpoint = False)
plt.figure()
plt.plot(model_ref[:,pmt.nx//2], D,label = "ref")
plt.plot(model_smooth[:,pmt.nx//2],D,label = "smooth")
# plt.plot(model_fwi[:,pmt.nx//2], D, label = "fwi")
plt.ylim(D[-1],0)
plt.legend()
plt.show()
