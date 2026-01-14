from Viewdata import plotting 

plt = plotting("../inputs/Parameters.json")

# plt.viewModel("layeredvp_Nz383_Nx141.bin")
# plt.viewSnapshot("acousticcerjan_shot_1_Nx301_Nz301_Nt4001_frame_", "../inputs/layeredvp_Nz301_Nx301.bin")
# plt.movieSnapshot("acousticcerjan_shot_1_Nx301_Nz301_Nt4001_frame_", "../inputs/layeredvp_Nz301_Nx301.bin")
# plt.viewSeismogram("../outputs/seismograms/acousticcerjan_seismogram_shot_1_Nt4001_Nrec501.bin", perc=95)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewMigratedImage("../outputs/migrated_image/migrated_image_VTI_Nx301_Nz301.bin",perc=99)
# plt.viewSnapshotAnalyticalComparison(1,"../outputs/snapshots/TTI_CPML_shot_1_Nx501_Nz201_Nt6000_frame_1000.bin")

