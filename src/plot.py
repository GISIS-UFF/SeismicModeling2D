from Viewdata import plotting 

plt = plotting("../inputs/Parameters.json")

# plt.viewModel("layeredvp_Nz101_Nx301.bin")
# plt.viewSnapshot("acousticcerjan_shot_1_Nx301_Nz101_Nt4001_frame_1200.bin", "../inputs/layeredvp_Nz101_Nx301.bin")
# plt.movieSnapshot("acousticcerjan_shot_1_Nx301_Nz101_Nt4001_frame_", "../inputs/layeredvp_Nz101_Nx301.bin", savegif = True)
# plt.viewSeismogram("../outputs/seismograms/acousticCPML_seismogram_shot_1_Nt4001_Nrec501.bin", perc=95)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewMigratedImage("../outputs/migrated_image/migrated_image_acoustic_Nx301_Nz101.bin",perc=99)
# plt.viewSnapshotAnalyticalComparison(1,"../outputs/snapshots/TTI_CPML_shot_1_Nx501_Nz201_Nt6000_frame_1000.bin")

# import matplotlib.pyplot as plt
# import numpy as np

# nx = 301
# nz = 101

# snaponthefly = np.fromfile("../outputs/snapshots/acousticcerjan_shot_1_Nx301_Nz101_Nt4001_frame_1200ONTHEFLY.bin", dtype=np.float32).reshape(nz, nx)
# snap = np.fromfile("../outputs/snapshots/acousticcerjan_shot_1_Nx301_Nz101_Nt4001_frame_1200.bin", dtype=np.float32).reshape(nz, nx)
# snapcheck = np.fromfile("../outputs/snapshots/acousticcerjan_shot_1_Nx301_Nz101_Nt4001_frame_1200CHECKPOINT.bin", dtype=np.float32).reshape(nz, nx)

# diff = snap - snapcheck
# plt.figure()
# plt.plot(snap[:, nx//2], label = "foward")
# plt.plot(snapcheck[:, nx//2], label = "reconstruido")
# plt.plot(diff[:, nx//2], label = "diff")
# plt.legend()
# plt.show()